# Save this as: train_shark6.py
import sys
import os
import uuid
import time
import random
import copy
import multiprocessing
from collections import Counter

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from bazaar_ai.bazaar import BasicBazaar
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

# --- IMPORT OPPONENTS ---
try:
    from agents.shark_agent import SharkAgent
except ImportError:
    print("ERROR: agents/shark_agent.py not found. Please save File 1 first.")
    sys.exit(1)

try:
    from agents.smart_agent import SmartAgent
except ImportError:
    print("WARNING: agents/smart_agent.py not found. Using RandomAgent as the 'Teacher' placeholder.")
    from agents.random_agent import RandomAgent as SmartAgent

# =========================================================
# 1. THE LEARNER: PARAMETRIC SUPER SHARK (Logic V6)
# =========================================================
class PerfectTracker:
    """Tracks opponent hand via public action logs."""
    def __init__(self):
        self.confirmed_hand = Counter()
        self.unknown_cards = 5
        self.hand_size = 5
        self.last_action_id = None

    def update(self, obs):
        if obs.action is None: return
        if id(obs.action) == self.last_action_id: return
        self.last_action_id = id(obs.action)
        act = obs.action
        
        if act.trader_action_type.value == "Sell":
            good = act._sell
            count = act._count
            self.hand_size -= count
            known_count = self.confirmed_hand[good]
            remove_from_known = min(known_count, count)
            remove_from_unknown = count - remove_from_known
            self.confirmed_hand[good] -= remove_from_known
            self.unknown_cards = max(0, self.unknown_cards - remove_from_unknown)

        elif act.trader_action_type.value == "Take":
            good = act._take
            if good != GoodType.CAMEL:
                self.hand_size += 1
                self.confirmed_hand[good] += 1

        elif act.trader_action_type.value == "Trade":
            for g in GoodType:
                if act.requested_goods[g] > 0: self.confirmed_hand[g] += act.requested_goods[g]
            for g in GoodType:
                count_out = act.offered_goods[g]
                if count_out > 0:
                    known_count = self.confirmed_hand[g]
                    remove_from_known = min(known_count, count_out)
                    remove_from_unknown = count_out - remove_from_known
                    self.confirmed_hand[g] -= remove_from_known
                    self.unknown_cards = max(0, self.unknown_cards - remove_from_unknown)

class ParametricSuperShark(Trader):
    def __init__(self, seed, name, genome=None):
        super().__init__(seed, name)
        self.tracker = PerfectTracker()
        if not hasattr(self, 'uuid'): self.uuid = uuid.uuid4()
        self.genome = genome # Genome passed from training loop

    def select_action(self, actions, observation, simulate_action_fnc):
        # 1. State Estimation
        self.tracker.update(observation)
        confirmed = self.tracker.confirmed_hand
        opp_hand_size = self.tracker.hand_size
        opponent_locked = (opp_hand_size >= 7)

        best_action = None
        best_score = float('-inf')

        hand = observation.actor_goods
        hand_size = hand.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        
        pressure = 0
        if hand_size >= hand_limit: pressure = 20 * self.genome['pressure_weight']
        elif hand_size >= hand_limit - 1: pressure = 5 * self.genome['pressure_weight']

        for action in actions:
            score = 0
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure, confirmed)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure, confirmed, opponent_locked)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size)
            
            score += random.random() * 0.1
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _get_token_value(self, good, count, obs):
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(count, len(tokens))
        return sum(tokens[-take_n:])

    def _calculate_opponent_potential(self, good, opp_confirmed_count, obs):
        potential_count = opp_confirmed_count + 1
        if potential_count < 3: return 0
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(potential_count, len(tokens))
        token_points = sum(tokens[-take_n:])
        bonus_points = 0
        if potential_count == 3: bonus_points = 2.0  
        elif potential_count == 4: bonus_points = 5.0  
        elif potential_count >= 5: bonus_points = 9.0  
        return token_points + bonus_points

    def _score_sell(self, action, obs, pressure, opp_confirmed_hand):
        good = action._sell
        count = action._count
        params = self.genome
        points = self._get_token_value(good, count, obs)
        
        opp_has_good = opp_confirmed_hand[good]
        is_luxury = good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]
        
        race_bonus = 0
        if is_luxury and opp_has_good >= 2 and count >= 3: race_bonus = 8.0 
        if opp_has_good == 0 and count == 4 and pressure < 5: race_bonus = -5.0

        bonus = 0
        if count == 3: bonus = params['bonus_3_est']
        elif count == 4: bonus = params['bonus_4_est']
        elif count >= 5: bonus = params['bonus_5_est']
        
        total = points + bonus + race_bonus
        if is_luxury: return (total * params['luxury_mult']) + pressure
        if good in [GoodType.LEATHER, GoodType.SPICE, GoodType.FABRIC]:
            if count >= 5: return (total * params['cheap_mult']) + pressure + 10
            if count == 4: return (total * (params['cheap_mult']*0.75)) + pressure + 5
            if count <= 2 and pressure < 10: return -50
        return total + pressure

    def _score_take(self, action, obs, current_hand_size, pressure, opp_confirmed_hand, opponent_locked):
        good = action._take
        params = self.genome
        if good == GoodType.CAMEL:
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2: return params['camel_min_util']
            return params['camel_take_val']

        tokens = obs.market_goods_coins.get(good, [])
        top_token_val = tokens[-1] if tokens else 1
        in_hand = obs.actor_goods[good]
        score = top_token_val
        if in_hand == 3: score += 15
        if in_hand == 4: score += 20
        if good in [GoodType.DIAMOND, GoodType.GOLD]: score += params['luxury_take_add']

        opp_count = opp_confirmed_hand[good]
        threat_value = self._calculate_opponent_potential(good, opp_count, obs)
        if threat_value > 0: score += (threat_value * params['denial_weight'])

        if opponent_locked and score < 20: score -= 2.0 
        return score - pressure

    def _score_trade(self, action, obs, current_hand_size):
        req = action.requested_goods
        off = action.offered_goods
        params = self.genome
        value_in = 0
        completes_set = False
        for g in GoodType:
            if req[g] > 0:
                tokens = obs.market_goods_coins.get(g, [])
                val = tokens[-1] if tokens else 0
                if obs.actor_goods[g] + req[g] >= 5:
                    value_in += params['trade_set_bonus']
                    completes_set = True
                else: value_in += val
        value_out = 0
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL: value_out += 2
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    val = tokens[-1] if tokens else 0
                    value_out += val
                    if obs.actor_goods[g] >= 3: 
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]: value_out += params['set_break_penalty']
                        else: value_out += 2.0
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_change = count_in - count_out
        if current_hand_size >= 6 and space_change < 0: value_in += 10
        if completes_set: return 100 + (value_in - value_out)
        return value_in - value_out

# =========================================================
# 2. TRAINING CONFIGURATION
# =========================================================
GENERATIONS = 100         # Run overnight
POPULATION_SIZE = 30      # Wide search
GAMES_PER_MATCH = 300     # Statistical significance
MUTATION_RATE = 0.35      # Aggressive evolution
MUTATION_STRENGTH = 0.5   # Large jumps in strategy

GENOME_KEYS = [
    'bonus_3_est', 'bonus_4_est', 'bonus_5_est',
    'luxury_mult', 'cheap_mult', 'pressure_weight',
    'camel_min_util', 'camel_take_val', 'trade_set_bonus',
    'luxury_take_add', 'set_break_penalty',
    'denial_weight' 
]

def create_random_genome():
    return {
        'bonus_3_est': random.uniform(1.0, 5.0),
        'bonus_4_est': random.uniform(3.0, 10.0),
        'bonus_5_est': random.uniform(8.0, 18.0),
        'luxury_mult': random.uniform(1.0, 5.0),
        'cheap_mult': random.uniform(1.0, 4.0),
        'pressure_weight': random.uniform(0.5, 4.0),
        'camel_min_util': random.uniform(2.0, 12.0),
        'camel_take_val': random.uniform(-2.0, 6.0),
        'trade_set_bonus': random.uniform(15.0, 80.0), 
        'luxury_take_add': random.uniform(5.0, 30.0),
        'set_break_penalty': random.uniform(5.0, 35.0),
        'denial_weight': random.uniform(0.0, 2.0) 
    }

def play_match(seed, genome):
    # 1. CREATE HERO
    hero = ParametricSuperShark(seed, "Hero", genome)
    
    # 2. SELECT OPPONENT (League Mode)
    # 30% SmartAgent (The Teacher), 70% SharkAgent (The Boss)
    if random.random() < 0.3:
        villain = SmartAgent(seed, "SmartAgent")
    else:
        villain = SharkAgent(seed, "SharkAgent")
    
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
        return 0 

def run_match_task(args):
    return play_match(*args)

def evaluate_population(population):
    tasks = []
    for i, genome in enumerate(population):
        seed_start = i * GAMES_PER_MATCH
        for j in range(GAMES_PER_MATCH):
            tasks.append((seed_start + j, genome))
            
    cpu_count = min(32, multiprocessing.cpu_count())
    
    with multiprocessing.Pool(cpu_count) as pool:
        results = []
        total = len(tasks)
        print(f"  > Eval: {total} games against Mixed League (70% Shark / 30% Smart)...")
        for i, res in enumerate(pool.imap(run_match_task, tasks)):
            results.append(res)
            if i % (total // 10) == 0:
                print(f"  > {int(i/total*100)}%...")
                sys.stdout.flush()

    wins_per_agent = [0] * len(population)
    for idx, win in enumerate(results):
        agent_idx = idx // GAMES_PER_MATCH
        wins_per_agent[agent_idx] += win
    return wins_per_agent

def mutate(genome):
    new_genome = copy.deepcopy(genome)
    for k in GENOME_KEYS:
        if random.random() < MUTATION_RATE:
            change = 1.0 + random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
            new_genome[k] *= change
    return new_genome

def run_evolution():
    population = [create_random_genome() for _ in range(POPULATION_SIZE)]
    best_all_time = (0, None)

    for gen in range(GENERATIONS):
        print(f"\n--- GENERATION {gen+1} ---")
        start_t = time.time()
        
        wins = evaluate_population(population)
        scored_pop = list(zip(wins, population))
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        top_wins, top_genome = scored_pop[0]
        
        duration = time.time() - start_t
        print(f"Gen Complete in {duration:.1f}s")
        print(f"Best: {top_wins}/{GAMES_PER_MATCH} ({top_wins/GAMES_PER_MATCH*100:.1f}%)")
        print(f"Genome: {top_genome}")
        
        if top_wins > best_all_time[0]:
            best_all_time = (top_wins, top_genome)
            print(f"** NEW RECORD **")

        cutoff = max(2, int(POPULATION_SIZE * 0.25))
        survivors = [x[1] for x in scored_pop[:cutoff]]
        
        new_pop = survivors[:]
        while len(new_pop) < POPULATION_SIZE:
            parent = random.choice(survivors)
            child = mutate(parent)
            new_pop.append(child)
        population = new_pop

    print("\n\n=== TRAINING COMPLETE ===")
    print(f"Best Win Rate: {best_all_time[0]}/{GAMES_PER_MATCH}")
    print(best_all_time[1])

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_evolution()