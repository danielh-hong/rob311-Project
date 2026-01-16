import sys
import os
import uuid
import time
import random
import copy
import multiprocessing
from collections import Counter

# --- PATH SETUP ---
# Ensure we can import bazaar_ai AND your agent files
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from bazaar_ai.bazaar import BasicBazaar
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

# --- CRITICAL: IMPORT THE REAL OPPONENTS ---
# We do NOT define them here. We use the actual files you verified.
try:
    from shark_agent6 import SharkAgent6
    from shark_agent import SharkAgent
    from smart_agent import SmartAgent
    print("SUCCESS: Imported Real Agents (Shark6, Shark, Smart)")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import agents. Make sure shark_agent6.py etc are in the same folder. {e}")
    sys.exit(1)

# =========================================================
# THE LEARNER: SHARK AGENT OMEGA (Parametric)
# =========================================================
# This contains the V7 Logic (Deck Counting, Mercy Kill, Hoarding)
# But relies on the Genome passed during training.

TOTAL_CARDS = {
    GoodType.DIAMOND: 6, GoodType.GOLD: 6, GoodType.SILVER: 6, 
    GoodType.FABRIC: 8, GoodType.SPICE: 8, GoodType.LEATHER: 10, GoodType.CAMEL: 11
}

class GlobalStateTracker:
    def __init__(self):
        self.opp_confirmed = Counter(); self.opp_hand_size = 5; self.sold_cards = Counter()
        self.opp_camels = 0; self.opp_score_est = 0; self.last_action_id = None
    def update(self, obs):
        if obs.action is None or id(obs.action) == self.last_action_id: return
        self.last_action_id = id(obs.action); act = obs.action
        if act.trader_action_type.value == "Sell":
            self.opp_hand_size -= act._count; known = self.opp_confirmed[act._sell]
            self.opp_confirmed[act._sell] -= min(known, act._count); self.sold_cards[act._sell] += act._count
            val = 5 if act._sell in [GoodType.DIAMOND, GoodType.GOLD] else 2; self.opp_score_est += (act._count * val)
        elif act.trader_action_type.value == "Take":
            if act._take == GoodType.CAMEL: self.opp_camels += getattr(act, '_count', 1)
            else: self.opp_hand_size += 1; self.opp_confirmed[act._take] += 1
        elif act.trader_action_type.value == "Trade":
            for g in GoodType:
                if act.requested_goods[g] > 0: self.opp_confirmed[g] += act.requested_goods[g]
            for g in GoodType:
                if act.offered_goods[g] > 0:
                    if g == GoodType.CAMEL: self.opp_camels = max(0, self.opp_camels - act.offered_goods[g])
                    else: known = self.opp_confirmed[g]; self.opp_confirmed[g] -= min(known, act.offered_goods[g])
    def get_deck_remaining(self, obs):
        rem = copy.deepcopy(TOTAL_CARDS)
        for g in GoodType: rem[g] -= obs.market_goods[g]
        for g in GoodType: rem[g] -= obs.actor_goods[g]
        for g in rem: rem[g] -= self.opp_confirmed[g]
        for g in rem: rem[g] -= self.sold_cards[g]
        for g in rem: rem[g] = max(0, rem[g])
        return rem

class ParametricSharkOmega(Trader):
    def __init__(self, seed, name, genome=None):
        super().__init__(seed, name)
        self.tracker = GlobalStateTracker()
        if not hasattr(self, 'uuid'): self.uuid = uuid.uuid4()
        self.genome = genome

    def select_action(self, actions, observation, simulate_action_fnc):
        self.tracker.update(observation)
        opp_confirmed = self.tracker.opp_confirmed
        deck_remaining = self.tracker.get_deck_remaining(observation)
        opp_locked = (self.tracker.opp_hand_size >= 7)
        cards_in_deck = observation.market_reserved_goods_count
        empty_piles = sum(1 for g in GoodType if not observation.market_goods_coins.get(g, []))
        is_endgame = (cards_in_deck <= 8) or (empty_piles >= 2)
        my_score = sum(sum(c) for c in observation.actor_goods_coins.values())
        am_i_winning = (my_score > self.tracker.opp_score_est + 10)
        hand_size = observation.actor_goods.count(include_camels=False)
        limit = observation.max_player_goods_count
        pressure = 20 * self.genome['pressure_weight'] if hand_size >= limit else (5 * self.genome['pressure_weight'] if hand_size >= limit-1 else 0)

        best_action = None; best_score = float('-inf')
        for action in actions:
            score = 0
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure, opp_confirmed, deck_remaining, is_endgame, am_i_winning)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure, opp_confirmed, opp_locked, is_endgame, deck_remaining)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size, deck_remaining)
            score += random.random() * 0.1
            if score > best_score: best_score = score; best_action = action
        return best_action

    def _get_val(self, good, count, obs):
        t = obs.market_goods_coins.get(good, [])
        return sum(t[-min(count, len(t)):]) if t else 0

    def _calculate_opponent_potential(self, good, opp_confirmed_count, deck_remaining, obs):
        max_possible = opp_confirmed_count + deck_remaining[good] + obs.market_goods[good]
        potential_count = opp_confirmed_count + 1
        if potential_count > max_possible: return 0 
        if potential_count < 3: return 0 
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(potential_count, len(tokens))
        return sum(tokens[-take_n:]) + (2.0 if potential_count == 3 else 5.0)

    def _score_sell(self, action, obs, pressure, opp_confirmed, deck, is_endgame, winning):
        g, c, p = action._sell, action._count, self.genome
        pts = self._get_val(g, c, obs)
        avail = len(obs.market_goods_coins.get(g, []))
        waste = p['waste_penalty'] * (c - avail) if c > avail else 0
        impossible = (deck[g] == 0 and opp_confirmed[g] == 0 and obs.market_goods[g] == 0)
        kill = p['mercy_kill_bonus'] if (winning and c >= avail and sum(1 for x in GoodType if not obs.market_goods_coins.get(x, [])) >= 2) else 0
        bonus = p['bonus_3_est'] if c==3 else (p['bonus_4_est'] if c==4 else (p['bonus_5_est'] if c>=5 else 0))
        race = 8.0 if (g in [GoodType.DIAMOND, GoodType.GOLD] and opp_confirmed[g] >= 2 and c >= 3) else 0
        if impossible: race += p['impossible_sell_bonus']
        if is_endgame and c >= 3: race += p['endgame_rush_bonus']
        total = pts + bonus + race + kill - waste
        if g in [GoodType.DIAMOND, GoodType.GOLD]: return (total * p['luxury_mult']) + pressure
        if g in [GoodType.LEATHER, GoodType.SPICE]:
            if c >= 5: return (total * p['cheap_mult']) + pressure + 10
            if is_endgame and c >= 3: return total + pressure
            if c <= 2 and pressure < 10: return -50
        return total + pressure

    def _score_take(self, action, obs, hand_size, pressure, opp_confirmed, locked, is_endgame, deck):
        g, p = action._take, self.genome
        if g == GoodType.CAMEL:
            wanted = sum(deck[t] for t in GoodType if t!=GoodType.CAMEL and obs.actor_goods[t]>=2)
            deck_total = sum(deck.values())
            fish = (wanted/deck_total * p['fishing_bonus']) if deck_total > 0 else 0
            if is_endgame: return p['endgame_camel_value']
            if obs.actor_goods[GoodType.CAMEL] < 2: return p['camel_min_util']
            return p['camel_take_val'] + fish
        t = obs.market_goods_coins.get(g, []); score = t[-1] if t else 1
        in_hand = obs.actor_goods[g]
        if deck[g] == 0: score += p['scarcity_bonus']
        elif deck[g] == 1: score += p['scarcity_bonus'] / 2
        if in_hand == 3: score += 15
        if in_hand == 4: score += 20
        if g in [GoodType.DIAMOND, GoodType.GOLD]: score += p['luxury_take_add']
        
        opp_c = opp_confirmed[g]
        threat = self._calculate_opponent_potential(g, opp_c, deck, obs)
        if threat > 0: score += (threat * p['denial_weight'])

        if locked and score < 20: score -= 2.0
        return score - pressure

    def _score_trade(self, action, obs, hand_size, deck):
        req, off, p = action.requested_goods, action.offered_goods, self.genome
        val_in = 0; complete = False
        for g in GoodType:
            if req[g] > 0:
                t = obs.market_goods_coins.get(g, []); val_in += t[-1] if t else 0
                if obs.actor_goods[g] + req[g] >= 5: val_in += p['trade_set_bonus']; complete = True
                elif obs.actor_goods[g] + req[g] == 4 and deck[g] == 0: val_in += p['trade_set_bonus']
        val_out = 0
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL: val_out += 2
                else:
                    t = obs.market_goods_coins.get(g, []); val_out += t[-1] if t else 0
                    if obs.actor_goods[g] >= 3:
                        val_out += p['set_break_penalty'] if g in [GoodType.DIAMOND, GoodType.GOLD] else 2.0
        if hand_size >= 6 and (req.count(False) - off.count(False)) < 0: val_in += 10
        if complete: return 100 + (val_in - val_out)
        return val_in - val_out

# =========================================================
# 3. PRODUCTION TRAINING LOOP
# =========================================================
GENERATIONS = 200         
POPULATION_SIZE = 50      
GAMES_PER_MATCH = 400     

GENOME_KEYS = [
    'bonus_3_est', 'bonus_4_est', 'bonus_5_est', 'luxury_mult', 'cheap_mult', 
    'pressure_weight', 'camel_min_util', 'camel_take_val', 'trade_set_bonus', 
    'luxury_take_add', 'set_break_penalty', 'denial_weight',
    'impossible_sell_bonus', 'scarcity_bonus', 'waste_penalty', 
    'endgame_rush_bonus', 'endgame_camel_value', 'mercy_kill_bonus', 'fishing_bonus'
]

# START FROM THE GENOME THAT "THOUGHT" IT WON 93% (It's a good baseline)
CHAMPION_GENOME = {
    'bonus_3_est': 0.440, 'bonus_4_est': 17.168, 'bonus_5_est': 26.994, 
    'luxury_mult': 0.185, 'cheap_mult': 0.523, 'pressure_weight': 0.114, 
    'camel_min_util': 4.239, 'camel_take_val': 0.008, 'trade_set_bonus': 4.199, 
    'luxury_take_add': 0.159, 'set_break_penalty': 37.262, 'denial_weight': 0.061, 
    'impossible_sell_bonus': 17.481, 'scarcity_bonus': 8.710, 'waste_penalty': 2.905, 
    'endgame_rush_bonus': 20.479, 'endgame_camel_value': 14.644, 
    'mercy_kill_bonus': 11.161, 'fishing_bonus': 1.989
}

def create_jittered_genome():
    g = copy.deepcopy(CHAMPION_GENOME)
    for k in g: g[k] *= random.uniform(0.8, 1.2)
    return g

def create_pure_random_genome():
    return {
        'bonus_3_est': random.uniform(0.1, 5.0),
        'bonus_4_est': random.uniform(5.0, 25.0),
        'bonus_5_est': random.uniform(10.0, 60.0), 
        'luxury_mult': random.uniform(0.1, 5.0),   
        'cheap_mult': random.uniform(0.1, 5.0),
        'pressure_weight': random.uniform(0.1, 10.0),
        'camel_min_util': random.uniform(1.0, 10.0),
        'camel_take_val': random.uniform(-5.0, 5.0), 
        'trade_set_bonus': random.uniform(2.0, 25.0),
        'luxury_take_add': random.uniform(0.0, 20.0),
        'set_break_penalty': random.uniform(5.0, 60.0), 
        'denial_weight': random.uniform(0.0, 3.0),
        'impossible_sell_bonus': random.uniform(5.0, 100.0),
        'scarcity_bonus': random.uniform(2.0, 30.0),
        'waste_penalty': random.uniform(1.0, 20.0),
        'endgame_rush_bonus': random.uniform(5.0, 50.0),
        'endgame_camel_value': random.uniform(2.0, 25.0),
        'mercy_kill_bonus': random.uniform(20.0, 200.0),
        'fishing_bonus': random.uniform(0.0, 20.0)
    }

def play_match(seed, genome):
    hero = ParametricSharkOmega(seed, "Hero", genome)
    roll = random.random()
    # 60% REAL SharkAgent6, 30% REAL SharkAgent, 10% REAL SmartAgent
    if roll < 0.6: villain = SharkAgent6(seed, "Boss_V6")
    elif roll < 0.9: villain = SharkAgent(seed, "Aggressor_Shark")
    else: villain = SmartAgent(seed, "Teacher_Smart")
    
    if not hasattr(hero, 'uuid'): hero.uuid = uuid.uuid4()
    if not hasattr(villain, 'uuid'): villain.uuid = uuid.uuid4()
    
    players = {hero.uuid: hero, villain.uuid: villain}
    order = [hero, villain] if (seed % 2 == 0) else [villain, hero]
    
    try:
        game = BasicBazaar(seed=seed, players=order)
        state = game.state
        while not game.terminal(state):
            actor = state.actor
            actions = game.all_actions(actor, state)
            if not actions: break
            observation = game.observe(actor, state)
            chosen_action = actor.select_action(actions, observation, lambda a: None)
            state = game.apply_action(state, chosen_action)
            game.state = state 
        scores = {pid: game.calculate_reward(players[pid], state, state) for pid in players}
        return 1 if scores[hero.uuid] > scores[villain.uuid] else 0
    except Exception: return 0

def run_match_task(args): return play_match(*args)

def evaluate_population(population):
    tasks = []
    for i, genome in enumerate(population):
        seed_start = i * GAMES_PER_MATCH
        for j in range(GAMES_PER_MATCH): tasks.append((seed_start + j, genome))
    
    cpu_count = min(32, multiprocessing.cpu_count())
    with multiprocessing.Pool(cpu_count) as pool:
        results = []
        total = len(tasks)
        print(f"  > REAL LEAGUE BATTLE ({total} games)...")
        for i, res in enumerate(pool.imap(run_match_task, tasks)):
            results.append(res)
            if i % (total // 10) == 0: print(f"  > {int(i/total*100)}%...", end="\r"); sys.stdout.flush()
    print("")
    wins_per_agent = [0] * len(population)
    for idx, win in enumerate(results): wins_per_agent[idx // GAMES_PER_MATCH] += win
    return wins_per_agent

def mutate(genome, rate, strength, catastrophic_rate):
    new_genome = copy.deepcopy(genome)
    for k in GENOME_KEYS:
        if random.random() < catastrophic_rate:
            new_genome[k] = create_pure_random_genome()[k]
        elif random.random() < rate:
            change = 1.0 + random.uniform(-strength, strength)
            new_genome[k] *= change
    return new_genome

def run_evolution():
    population = []
    # 50/50 Start
    for _ in range(int(POPULATION_SIZE / 2)): population.append(create_jittered_genome())
    for _ in range(int(POPULATION_SIZE / 2)): population.append(create_pure_random_genome())
    best_all_time = (0, None)

    for gen in range(GENERATIONS):
        progress = gen / GENERATIONS
        current_rate = 0.5 - (0.4 * progress)
        current_strength = 0.8 - (0.6 * progress)
        current_catastrophic = 0.10 - (0.09 * progress)
        
        print(f"\n--- GEN {gen+1}/{GENERATIONS} [Mut:{current_rate:.2f} Str:{current_strength:.2f} Cat:{current_catastrophic:.2f}] ---")
        start_t = time.time()
        
        wins = evaluate_population(population)
        scored_pop = list(zip(wins, population))
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        top_wins, top_genome = scored_pop[0]
        duration = time.time() - start_t
        print(f"Gen Time: {duration:.1f}s")
        print(f"Best: {top_wins}/{GAMES_PER_MATCH} ({top_wins/GAMES_PER_MATCH*100:.1f}%)")
        print(f"Top Genome: {top_genome}")
        
        if top_wins > best_all_time[0]:
            best_all_time = (top_wins, top_genome)
            print(f"** NEW RECORD **")

        cutoff = int(POPULATION_SIZE * 0.2)
        survivors = [x[1] for x in scored_pop[:cutoff]]
        
        new_pop = survivors[:]
        while len(new_pop) < POPULATION_SIZE:
            parent = random.choice(survivors)
            child = mutate(parent, current_rate, current_strength, current_catastrophic)
            new_pop.append(child)
        population = new_pop

    print(f"\nFINAL BEST: {best_all_time[0]} wins\n{best_all_time[1]}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_evolution()