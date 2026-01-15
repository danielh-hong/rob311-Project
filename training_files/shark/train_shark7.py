# =============================================================================
# SHARK AGENT OMEGA - PRODUCTION TRAINING SCRIPT
# =============================================================================
#
# TRAINING SPLIT:
# ----------------
# 60% vs SharkAgent6 (The Boss)
#     - We must beat the current champion to become the champion.
# 30% vs SharkAgent (Original)
#     - Ensures we don't forget how to handle raw aggression/set-hunting.
# 10% vs SmartAgent
#     - Ensures we maintain fundamental value-trading principles.
#
# CONFIGURATION:
# ----------------
# Generations:    200 (Deep evolution to fine-tune weights)
# Population:     50  (Wide exploration of strategy space)
# Games per Match: 400 (Statistical certainty >99% to rule out luck)
#
# EVOLUTIONARY STRATEGY:
# -----------------------
# 1. Elitism: Top 20% of agents survive unchanged.
# 2. Annealing: Mutation starts high (exploration) and drops low (refinement).
# 3. Catastrophe: Small chance to fully randomize a gene to break local optima.
# =============================================================================

'''
The 19 Genes:

bonus_3_est: How much do I value a 3-card bonus? (Is it worth 1pt or 3pts?)

bonus_4_est: Value of 4-card bonus.

bonus_5_est: Value of 5-card bonus.

luxury_mult: How much more do I love Diamonds than Leather?

cheap_mult: How much do I care about bulk Leather/Spice?

pressure_weight: How much do I panic when my hand is full?

camel_min_util: How badly do I want camels when I have none?

camel_take_val: How much do I want camels when I already have some?

trade_set_bonus: How much is completing a set via Trade worth?

luxury_take_add: Extra bonus for taking a single Diamond.

set_break_penalty: Penalty for trading away a card that breaks a set.

denial_weight: How much do I sacrifice to hurt the opponent?

impossible_sell_bonus: (New) How fast do I sell if the deck is empty?

scarcity_bonus: (New) How badly do I want the last Diamond in the deck?

waste_penalty: (New) How much do I hate selling 2 cards for 1 token?

endgame_rush_bonus: (New) How much do I panic sell at the end?

endgame_camel_value: (New) Value of camels when deck < 8.

mercy_kill_bonus: (New) Incentive to end game when winning.

fishing_bonus: (New) Incentive to take camels to fish for cards.

The script finds the perfect balance of these 19 numbers that beats the previous champion.
'''
import sys
import os
import uuid
import time
import random
import copy
import multiprocessing
from collections import Counter

# =============================================================================
# SHARK AGENT OMEGA - PRODUCTION TRAINING SCRIPT
# =============================================================================
#
# TRAINING SPLIT (The League):
# ----------------------------
# 60% vs SharkAgent6 (The Boss):
#     We must beat the current champion (Win Rate 69%) to become the champion.
# 30% vs SharkAgent (The Aggressor):
#     Ensures robust handling of raw aggression and set-hunting.
# 10% vs SmartAgent (The Teacher):
#     Ensures we maintain fundamental value-trading principles.
#
# CONFIGURATION:
# --------------
# Generations:    200 (Deep evolution to fine-tune weights)
# Population:     50  (Wide exploration of strategy space)
# Games per Match: 400 (Statistical certainty >99% to rule out luck)
#
# EVOLUTIONARY MECHANICS:
# -----------------------
# 1. Elitism: Top 20% of agents survive unchanged.
# 2. Annealing: Mutation starts high (exploration) and drops low (refinement).
# 3. Catastrophe: Small chance to fully randomize a gene to break local optima.
# =============================================================================

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from bazaar_ai.bazaar import BasicBazaar
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

# =========================================================
# 1. THE OPPONENTS (THE LEAGUE)
# =========================================================

class SharkAgent(Trader):
    """The Aggressor (30% of games) - Hunts sets aggressively."""
    def __init__(self, seed, name):
        super().__init__(seed, name)
        self.bonus_estimates = {3: 2.0, 4: 5.5, 5: 9.0}

    def select_action(self, actions, observation, simulate_action_fnc):
        best_action = None; best_score = float('-inf')
        hand_size = observation.actor_goods.count(include_camels=False)
        pressure = 20 if hand_size >= 7 else (5 if hand_size >= 6 else 0)

        for action in actions:
            score = 0
            if isinstance(action, SellAction):
                pts = self._val(action._sell, action._count, observation)
                bon = self.bonus_estimates.get(action._count, 0)
                # Aggressive Luxury Logic
                if action._sell in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]: 
                    score = (pts+bon)*1.5 + pressure
                else:
                    if action._count >= 5: score = (pts+bon)*2.0 + pressure + 10
                    elif action._count == 4: score = (pts+bon)*1.5 + pressure + 5
                    elif action._count <= 2 and pressure < 10: score = -50
                    else: score = pts + bon + pressure
            elif isinstance(action, TakeAction):
                if action._take == GoodType.CAMEL:
                    m_c = getattr(action, '_count', 0)
                    my_c = observation.actor_goods[GoodType.CAMEL]
                    score = 2.0 if m_c >= 4 else (5.0 if my_c < 2 else -2.0)
                else:
                    t = observation.market_goods_coins.get(action._take, [])
                    val = t[-1] if t else 1
                    ih = observation.actor_goods[action._take]
                    if ih == 3: val += 15
                    if ih == 4: val += 20
                    if ih == 1 and action._take in [GoodType.DIAMOND, GoodType.GOLD]: val += 10
                    if hand_size >= 5: val -= 5
                    score = val - pressure
            elif isinstance(action, TradeAction): score = 5
            
            score += random.random() * 0.1
            if score > best_score: best_score = score; best_action = action
        return best_action

    def _val(self, g, c, obs):
        t = obs.market_goods_coins.get(g, [])
        return sum(t[-min(c, len(t)):]) if t else 0

class SmartAgent(Trader):
    """The Teacher (10% of games) - Standard Value Trader."""
    def __init__(self, seed, name):
        super().__init__(seed, name)
        self.good_values = {GoodType.DIAMOND: 7, GoodType.GOLD: 6, GoodType.SILVER: 5,
                            GoodType.FABRIC: 4, GoodType.SPICE: 4, GoodType.LEATHER: 3, GoodType.CAMEL: 1}

    def select_action(self, actions, observation, simulate_action_fnc):
        best_action = None; best_score = float('-inf')
        for action in actions:
            score = 0
            if isinstance(action, SellAction):
                avail = observation.market_goods_coins.get(action._sell, [])
                if len(avail) < action._count: score = -1000
                else:
                    val = sum(avail[-action._count:])
                    mult = 3.0 if action._count>=5 else (2.5 if action._count>=4 else 2.0)
                    scarcity = len(avail)*2 if action._sell in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER] else 0
                    score = (val * mult) + scarcity
            elif isinstance(action, TakeAction):
                if action._take == GoodType.CAMEL: score = 5 if observation.market_reserved_goods_count < 15 else 2
                else:
                    if (7 - observation.actor_goods.count(False)) <= 2: score = -500
                    else:
                        g_val = self.good_values.get(action._take, 1)
                        ih = observation.actor_goods[action._take]
                        bon = 30 if ih+1>=5 else (20 if ih+1>=4 else (15 if ih+1>=3 else 0))
                        score = (g_val * 5) + bon
            elif isinstance(action, TradeAction): score = 5
            if score > best_score: best_score = score; best_action = action
        return best_action if best_action else actions[0]

class Shark6Tracker:
    def __init__(self): self.confirmed = Counter()
    def update(self, obs):
        if obs.action and obs.action.trader_action_type.value == "Take" and obs.action._take != GoodType.CAMEL:
            self.confirmed[obs.action._take] += 1

class SharkAgent6(Trader):
    """The Boss (60% of games) - The Current Champion Strategy."""
    def __init__(self, seed, name):
        super().__init__(seed, name)
        self.tracker = Shark6Tracker()
        self.genome = {'pressure_weight': 0.579, 'luxury_mult': 0.346, 'denial_weight': 0.062}

    def select_action(self, actions, obs, sim):
        self.tracker.update(obs)
        best = None; best_s = float('-inf')
        for a in actions:
            s = 0
            if isinstance(a, SellAction): s = 50 if a._count >= 4 else 10
            elif isinstance(a, TakeAction): 
                if a._take != GoodType.CAMEL and self.tracker.confirmed[a._take] >= 2: s = 20
                else: s = 15
            elif isinstance(a, TradeAction): s = 12
            if s > best_s: best_s = s; best = a
        return best

# =========================================================
# 2. THE LEARNER: SHARK AGENT OMEGA
# =========================================================
TOTAL_CARDS = {GoodType.DIAMOND: 6, GoodType.GOLD: 6, GoodType.SILVER: 6, 
               GoodType.FABRIC: 8, GoodType.SPICE: 8, GoodType.LEATHER: 10, GoodType.CAMEL: 11}

class GlobalStateTracker:
    """Omniscient tracking of hidden information."""
    def __init__(self):
        self.opp_confirmed = Counter()
        self.opp_hand_size = 5
        self.sold_cards = Counter()
        self.opp_camels = 0
        self.opp_score_est = 0
        self.last_action_id = None

    def update(self, obs):
        if obs.action is None or id(obs.action) == self.last_action_id: return
        self.last_action_id = id(obs.action)
        act = obs.action
        
        if act.trader_action_type.value == "Sell":
            self.opp_hand_size -= act._count
            known = self.opp_confirmed[act._sell]
            self.opp_confirmed[act._sell] -= min(known, act._count)
            self.sold_cards[act._sell] += act._count
            val = 5 if act._sell in [GoodType.DIAMOND, GoodType.GOLD] else 2
            self.opp_score_est += (act._count * val)

        elif act.trader_action_type.value == "Take":
            if act._take == GoodType.CAMEL:
                self.opp_camels += getattr(act, '_count', 1)
            else:
                self.opp_hand_size += 1
                self.opp_confirmed[act._take] += 1

        elif act.trader_action_type.value == "Trade":
            for g in GoodType:
                if act.requested_goods[g] > 0: self.opp_confirmed[g] += act.requested_goods[g]
            for g in GoodType:
                if act.offered_goods[g] > 0:
                    if g == GoodType.CAMEL: 
                        self.opp_camels = max(0, self.opp_camels - act.offered_goods[g])
                    else:
                        known = self.opp_confirmed[g]
                        self.opp_confirmed[g] -= min(known, act.offered_goods[g])

    def get_deck_remaining(self, obs):
        rem = copy.deepcopy(TOTAL_CARDS)
        for g in GoodType: rem[g] -= obs.market_goods[g]
        for g in GoodType: rem[g] -= obs.actor_goods[g]
        for g in rem: rem[g] -= self.opp_confirmed[g]
        for g in rem: rem[g] -= self.sold_cards[g]
        for g in rem: rem[g] = max(0, rem[g])
        return rem

class ParametricSharkOmega(Trader):
    """The Agent being trained."""
    def __init__(self, seed, name, genome=None):
        super().__init__(seed, name)
        self.tracker = GlobalStateTracker()
        if not hasattr(self, 'uuid'): self.uuid = uuid.uuid4()
        self.genome = genome

    def select_action(self, actions, observation, simulate_action_fnc):
        # 1. Update Tracking
        self.tracker.update(observation)
        
        # 2. Derive Context
        opp_confirmed = self.tracker.opp_confirmed
        deck_remaining = self.tracker.get_deck_remaining(observation)
        opp_locked = (self.tracker.opp_hand_size >= 7)
        
        cards_in_deck = observation.market_reserved_goods_count
        empty_piles = sum(1 for g in GoodType if not observation.market_goods_coins.get(g, []))
        is_endgame = (cards_in_deck <= 8) or (empty_piles >= 2)
        
        my_score = sum(sum(c) for c in observation.actor_goods_coins.values())
        am_i_winning = (my_score > self.tracker.opp_score_est + 10)

        # 3. Calculate Pressure
        hand_size = observation.actor_goods.count(include_camels=False)
        limit = observation.max_player_goods_count
        pressure = 0
        if hand_size >= limit: 
            pressure = 20 * self.genome['pressure_weight']
        elif hand_size >= limit - 1: 
            pressure = 5 * self.genome['pressure_weight']

        # 4. Evaluate Actions
        best_action = None
        best_score = float('-inf')

        for action in actions:
            score = 0
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure, opp_confirmed, deck_remaining, is_endgame, am_i_winning)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure, opp_confirmed, opp_locked, is_endgame, deck_remaining)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size, deck_remaining)
            
            # Deterministic Jitter
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    # --- SCORING LOGIC ---

    def _get_val(self, good, count, obs):
        t = obs.market_goods_coins.get(good, [])
        return sum(t[-min(count, len(t)):]) if t else 0

    def _score_sell(self, action, obs, pressure, opp_confirmed, deck, is_endgame, winning):
        g, c, p = action._sell, action._count, self.genome
        
        # Base Points
        pts = self._get_val(g, c, obs)
        
        # Penalties
        avail = len(obs.market_goods_coins.get(g, []))
        waste = p['waste_penalty'] * (c - avail) if c > avail else 0
        
        # Logic Flags
        impossible = (deck[g] == 0 and opp_confirmed[g] == 0 and obs.market_goods[g] == 0)
        kill_trigger = (winning and c >= avail and sum(1 for x in GoodType if not obs.market_goods_coins.get(x, [])) >= 2)
        kill = p['mercy_kill_bonus'] if kill_trigger else 0

        # Bonuses
        bonus = 0
        if c == 3: bonus = p['bonus_3_est']
        elif c == 4: bonus = p['bonus_4_est']
        elif c >= 5: bonus = p['bonus_5_est']
        
        # Race Logic
        race = 0
        if g in [GoodType.DIAMOND, GoodType.GOLD] and opp_confirmed[g] >= 2 and c >= 3:
            race = 8.0 
        
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
        
        # Camel Logic with Fishing
        if g == GoodType.CAMEL:
            wanted = 0
            for t in GoodType:
                if t != GoodType.CAMEL and obs.actor_goods[t] >= 2:
                    wanted += deck[t]
            
            deck_total = sum(deck.values())
            fishing_score = 0
            if deck_total > 0:
                fishing_score = (wanted / deck_total) * p['fishing_bonus']

            if is_endgame: return p['endgame_camel_value']
            if obs.actor_goods[GoodType.CAMEL] < 2: return p['camel_min_util']
            return p['camel_take_val'] + fishing_score

        # Goods Logic
        t = obs.market_goods_coins.get(g, [])
        score = t[-1] if t else 1
        in_hand = obs.actor_goods[g]
        
        # Scarcity Logic
        if deck[g] == 0: score += p['scarcity_bonus']
        elif deck[g] == 1: score += p['scarcity_bonus'] / 2

        if in_hand == 3: score += 15
        if in_hand == 4: score += 20
        if g in [GoodType.DIAMOND, GoodType.GOLD]: score += p['luxury_take_add']

        # Denial Logic
        opp_c = opp_confirmed[g]
        max_poss = opp_c + deck[g] + obs.market_goods[g]
        # Only deny if physically possible for opponent to get a set
        if (opp_c + 1) <= max_poss and opp_c >= 2:
             score += (8.0 * p['denial_weight'])

        if locked and score < 20: score -= 2.0
        return score - pressure

    def _score_trade(self, action, obs, hand_size, deck):
        req, off, p = action.requested_goods, action.offered_goods, self.genome
        
        # Value In
        val_in = 0
        complete = False
        for g in GoodType:
            if req[g] > 0:
                t = obs.market_goods_coins.get(g, [])
                val_in += t[-1] if t else 0
                if obs.actor_goods[g] + req[g] >= 5: 
                    val_in += p['trade_set_bonus']
                    complete = True
                elif obs.actor_goods[g] + req[g] == 4 and deck[g] == 0:
                    val_in += p['trade_set_bonus'] # Impossible 4 is basically a 5

        # Value Out
        val_out = 0
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL: 
                    val_out += 2
                else:
                    t = obs.market_goods_coins.get(g, [])
                    val_out += t[-1] if t else 0
                    if obs.actor_goods[g] >= 3:
                        penalty = p['set_break_penalty'] if g in [GoodType.DIAMOND, GoodType.GOLD] else 2.0
                        val_out += penalty

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

# Base Champion Weights (Safe fallback)
CHAMPION_GENOME = {
    'bonus_3_est': 0.46, 'bonus_4_est': 20.2, 'bonus_5_est': 30.0,
    'luxury_mult': 0.35, 'cheap_mult': 0.49, 'pressure_weight': 0.58,
    'camel_min_util': 4.8, 'camel_take_val': 0.02, 'trade_set_bonus': 5.2,
    'luxury_take_add': 0.32, 'set_break_penalty': 35.4, 'denial_weight': 0.06,
    'impossible_sell_bonus': 15.0, 'scarcity_bonus': 8.0, 'waste_penalty': 5.0,
    'endgame_rush_bonus': 10.0, 'endgame_camel_value': 15.0, 'mercy_kill_bonus': 50.0, 'fishing_bonus': 5.0
}

def create_jittered_genome():
    """Mutant: Safe evolution."""
    g = copy.deepcopy(CHAMPION_GENOME)
    for k in g: g[k] *= random.uniform(0.8, 1.2)
    return g

def create_pure_random_genome():
    """Alien: Wide Net Strategy Search."""
    return {
        # Valuation (Wide ranges to allow hoarding behavior)
        'bonus_3_est': random.uniform(0.1, 5.0),
        'bonus_4_est': random.uniform(5.0, 25.0),
        'bonus_5_est': random.uniform(10.0, 60.0), 
        
        # Multipliers (Allow <1.0 for hoarding)
        'luxury_mult': random.uniform(0.1, 5.0),   
        'cheap_mult': random.uniform(0.1, 5.0),
        
        # Tactics
        'pressure_weight': random.uniform(0.1, 10.0),
        'camel_min_util': random.uniform(1.0, 10.0),
        'camel_take_val': random.uniform(-5.0, 5.0), 
        'trade_set_bonus': random.uniform(2.0, 25.0),
        'luxury_take_add': random.uniform(0.0, 20.0),
        'set_break_penalty': random.uniform(5.0, 60.0), 
        'denial_weight': random.uniform(0.0, 3.0),
        
        # Advanced Logic
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
    # 60/30/10 Split
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
    except Exception: 
        return 0 # Count crash as loss

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
        print(f"  > LEAGUE BATTLE ({total} games)...")
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
        # Catastrophic Mutation
        if random.random() < catastrophic_rate:
            new_genome[k] = create_pure_random_genome()[k]
        # Standard Mutation
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
        # Simulated Annealing
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