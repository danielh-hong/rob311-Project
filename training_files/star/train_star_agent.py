# train_star_agent.py
#
# Evolves the genome weights inside StarAgent by running hundreds of games
# and keeping the parameter sets that win the most.
#
# HOW TO RUN:
#   python train_star_agent.py
#
# When it finishes, copy the printed genome dict into star_agent.py
# inside the self.base_genome = { ... } block.

import sys
import os
import uuid
import time
import random
import copy
import multiprocessing
from collections import Counter

# ---------------------------------------------------------------------------
# PATH SETUP
# This file lives at: ROB311-PROJECT/training_files/star/train_star_agent.py
# Two dirname() calls brings us up to ROB311-PROJECT/ where bazaar_ai lives.
# ---------------------------------------------------------------------------
# Hardcoded to your exact project location
project_root = r"C:\NicoleSchool\ROB311\rob311-Project"
bazaar_root = os.path.join(project_root, "..", "bazaar-ai", "src", "bazaar-ai")   # bazaar_ai lives inside bazaar-ai/
agents_dir = os.path.join(project_root, "agents")

sys.path.insert(0, project_root)
sys.path.insert(0, bazaar_root)
sys.path.insert(0, agents_dir)

from backend.bazaar import BasicBazaar
from backend.trader import Trader, SellAction, TakeAction, TradeAction
from backend.goods import GoodType

# ---------------------------------------------------------------------------
# IMPORT OPPONENTS TO TRAIN AGAINST
# These files must be inside ROB311-PROJECT/agents/
# ---------------------------------------------------------------------------
try:
    from shark_agent6 import SharkAgent6
except ImportError:
    print("ERROR: Could not find agents/shark_agent6.py")
    print("Make sure shark_agent6.py is inside ROB311-PROJECT/agents/")
    sys.exit(1)

try:
    from smart_agent import SmartAgent
except ImportError:
    print("WARNING: Could not find agents/smart_agent.py — will only train vs SharkAgent6")
    SmartAgent = None


# ===========================================================================
# TRAINABLE VERSION OF STAR AGENT
# This is identical to star_agent.py but accepts a genome= argument
# so the training loop can inject different parameter sets.
# ===========================================================================

class EnhancedOpponentTracker:
    def __init__(self):
        self.confirmed_hand = Counter()
        self.unknown_cards = 5
        self.hand_size = 5
        self.last_action_id = None
        self.action_history = []
        self.goods_taken = Counter()
        self.goods_sold = Counter()

    def update(self, obs):
        if obs.action is None:
            return
        if id(obs.action) == self.last_action_id:
            return
        self.last_action_id = id(obs.action)
        act = obs.action
        self.action_history.append(act.trader_action_type.value)

        if act.trader_action_type.value == "Sell":
            good = act._sell
            count = act._count
            self.hand_size -= count
            self.goods_sold[good] += count
            known_count = self.confirmed_hand[good]
            remove_from_known = min(known_count, count)
            remove_from_unknown = count - remove_from_known
            self.confirmed_hand[good] -= remove_from_known
            self.unknown_cards = max(0, self.unknown_cards - remove_from_unknown)

        elif act.trader_action_type.value == "Take":
            good = act._take
            self.goods_taken[good] += 1
            if good != GoodType.CAMEL:
                self.hand_size += 1
                self.confirmed_hand[good] += 1

        elif act.trader_action_type.value == "Trade":
            for g in GoodType:
                count_in = act.requested_goods[g]
                if count_in > 0:
                    self.confirmed_hand[g] += count_in
                    self.goods_taken[g] += count_in
                count_out = act.offered_goods[g]
                if count_out > 0:
                    known_count = self.confirmed_hand[g]
                    remove_from_known = min(known_count, count_out)
                    remove_from_unknown = count_out - remove_from_known
                    self.confirmed_hand[g] -= remove_from_known
                    self.unknown_cards = max(0, self.unknown_cards - remove_from_unknown)

    def get_threat_level(self, good_type):
        confirmed = self.confirmed_hand[good_type]
        if confirmed >= 2:
            return 3
        elif confirmed == 1 and self.goods_taken[good_type] >= 2:
            return 2
        return 1


class GamePhaseAnalyzer:
    def __init__(self):
        pass

    def analyze_phase(self, obs):
        depleted_types = 0
        total_tokens_remaining = 0
        for good_type in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER,
                          GoodType.FABRIC, GoodType.SPICE, GoodType.LEATHER]:
            tokens = obs.market_goods_coins.get(good_type, [])
            if len(tokens) == 0:
                depleted_types += 1
            total_tokens_remaining += len(tokens)
        deck_remaining = obs.market_reserved_goods_count
        if depleted_types >= 2:
            return 'late', 0.9
        elif depleted_types >= 1 or deck_remaining < 15:
            return 'mid', 0.6
        elif total_tokens_remaining < 20 or deck_remaining < 25:
            return 'mid', 0.4
        else:
            return 'early', 0.2

    def get_phase_priorities(self, phase):
        if phase == 'early':
            return {'collect_luxury': 1.2, 'build_sets': 1.0, 'deny_opponent': 0.7, 'sell_pressure': 0.5}
        elif phase == 'mid':
            return {'collect_luxury': 1.0, 'build_sets': 1.2, 'deny_opponent': 1.0, 'sell_pressure': 0.8}
        else:
            return {'collect_luxury': 0.7, 'build_sets': 0.8, 'deny_opponent': 1.3, 'sell_pressure': 1.5}


class TrainableStarAgent(Trader):
    """StarAgent that accepts an injected genome for training."""

    def __init__(self, seed, name, genome=None):
        super().__init__(seed, name)
        if not hasattr(self, 'uuid'):
            self.uuid = uuid.uuid4()

        self.tracker = EnhancedOpponentTracker()
        self.phase_analyzer = GamePhaseAnalyzer()

        # Default genome (SharkAgent6 trained values as starting point)
        self.base_genome = {
            'bonus_3_est': 0.461,
            'bonus_4_est': 20.2,
            'bonus_5_est': 30.0,
            'luxury_mult': 0.346,
            'cheap_mult': 0.488,
            'pressure_weight': 0.579,
            'camel_min_util': 4.8,
            'camel_take_val': 2.0,
            'trade_set_bonus': 5.23,
            'luxury_take_add': 0.316,
            'set_break_penalty': 35.4,
            'denial_weight': 0.45,
        }

        # Override with evolved genome if provided
        if genome is not None:
            self.base_genome.update(genome)

        self.strategy_params = {
            'denial_threshold': 15.0,
        }

    def select_action(self, actions, observation, simulate_action_fnc):
        self.tracker.update(observation)
        phase, urgency = self.phase_analyzer.analyze_phase(observation)

        hand = observation.actor_goods
        hand_size = hand.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        opp_confirmed = self.tracker.confirmed_hand
        opp_hand_size = self.tracker.hand_size
        opponent_locked = (opp_hand_size >= 7)

        pressure = self._calculate_pressure(hand_size, hand_limit, urgency)
        phase_mods = self.phase_analyzer.get_phase_priorities(phase)

        best_action = None
        best_score = float('-inf')

        for action in actions:
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure, opp_confirmed, phase, phase_mods)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure, opp_confirmed, opponent_locked, phase, phase_mods)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size, phase, phase_mods)
            else:
                score = float('-inf')

            score += random.random() * 0.01

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _calculate_pressure(self, hand_size, hand_limit, urgency):
        base_pressure = 0
        if hand_size >= hand_limit:
            base_pressure = 25
        elif hand_size >= hand_limit - 1:
            base_pressure = 10
        elif hand_size >= hand_limit - 2:
            base_pressure = 3
        return base_pressure * (1 + urgency * 0.5)

    def _score_sell(self, action, obs, pressure, opp_confirmed_hand, phase, phase_mods):
        good = action._sell
        count = action._count
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens:
            return -1000
        take_n = min(count, len(tokens))
        coins_value = sum(tokens[-take_n:])
        bonus_value = 0
        if count == 3:   bonus_value = self.base_genome['bonus_3_est']
        elif count == 4: bonus_value = self.base_genome['bonus_4_est']
        elif count >= 5: bonus_value = self.base_genome['bonus_5_est']
        total = coins_value + bonus_value
        is_luxury = good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]

        if is_luxury:
            opp_has = opp_confirmed_hand[good]
            threat_level = self.tracker.get_threat_level(good)
            race_bonus = 0
            if opp_has >= 3 and threat_level >= 3:   race_bonus = 15.0
            elif opp_has >= 2 and threat_level >= 2: race_bonus = 8.0
            elif opp_has == 0 and count == 4 and phase == 'early': race_bonus = -8.0
            total += race_bonus
            late_mult = phase_mods['sell_pressure'] if phase == 'late' else 1.0
            return (total * self.base_genome['luxury_mult'] * late_mult) + pressure

        if count >= 5:
            return (total * self.base_genome['cheap_mult'] * phase_mods['sell_pressure']) + pressure + 15
        if count == 4:
            return (total * self.base_genome['cheap_mult'] * 0.75 * phase_mods['sell_pressure']) + pressure + 8
        if count <= 2 and pressure < 10:
            return -50
        return total * phase_mods['sell_pressure'] + pressure

    def _score_take(self, action, obs, current_hand_size, pressure,
                    opp_confirmed_hand, opponent_locked, phase, phase_mods):
        good = action._take
        if good == GoodType.CAMEL:
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2: return self.base_genome['camel_min_util']
            return self.base_genome['camel_take_val']
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens:
            return -10
        top_token = tokens[-1]
        in_hand = obs.actor_goods[good]
        score = top_token
        if in_hand == 4: score += 20
        elif in_hand == 3: score += 15
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            score += self.base_genome['luxury_take_add']
            score *= phase_mods['collect_luxury']
        opp_count = opp_confirmed_hand[good]
        threat_value = self._calculate_threat_value(good, opp_count, obs)
        if threat_value > self.strategy_params['denial_threshold']:
            score += threat_value * self.base_genome['denial_weight'] * phase_mods['deny_opponent']
        elif threat_value > 0:
            score += threat_value * 0.2
        if opponent_locked and score < 20: score -= 2.0
        elif not opponent_locked and threat_value > 10: score += 5.0
        if phase == 'late' and in_hand == 0: score -= 8.0
        return score - pressure

    def _score_trade(self, action, obs, current_hand_size, phase, phase_mods):
        req = action.requested_goods
        off = action.offered_goods
        value_in = 0
        completes_valuable_set = False
        for g in GoodType:
            if req[g] > 0:
                tokens = obs.market_goods_coins.get(g, [])
                token_val = tokens[-1] if tokens else 1
                current_count = obs.actor_goods[g]
                future_count = current_count + req[g]
                if future_count >= 5:
                    value_in += self.base_genome['trade_set_bonus'] * 2
                    completes_valuable_set = True
                elif future_count >= 4:
                    value_in += self.base_genome['trade_set_bonus'] * 1.2
                elif future_count >= 3:
                    value_in += self.base_genome['trade_set_bonus']
                else:
                    value_in += token_val * 3
        value_out = 0
        breaking_luxury = False
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL:
                    value_out += 3
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    token_val = tokens[-1] if tokens else 1
                    value_out += token_val * 2
                    current_count = obs.actor_goods[g]
                    if current_count >= 4:
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                            value_out += self.base_genome['set_break_penalty']
                            breaking_luxury = True
                        else:
                            value_out += 15.0
                    elif current_count >= 3:
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                            value_out += 20.0
                        else:
                            value_out += 8.0
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_change = count_in - count_out
        if current_hand_size >= 6 and space_change < 0:
            value_in += 15
        if phase == 'late' and not completes_valuable_set:
            value_out += 10
        if completes_valuable_set and not breaking_luxury:
            return 120 + (value_in - value_out)
        return (value_in - value_out) * phase_mods['build_sets']

    def _calculate_threat_value(self, good, opp_count, obs):
        potential_count = opp_count + 1
        if potential_count < 3: return 0
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(potential_count, len(tokens))
        token_points = sum(tokens[-take_n:])
        bonus_points = 0
        if potential_count == 3:   bonus_points = 2.0
        elif potential_count == 4: bonus_points = 5.0
        elif potential_count >= 5: bonus_points = 9.0
        return token_points + bonus_points

    def calculate_reward(self, old_observation, new_observation, has_acted, environment_reward):
        pass


# ===========================================================================
# TRAINING CONFIGURATION
# Tune these numbers based on how long you want to wait.
# ===========================================================================
GENERATIONS     = 20   # Number of evolution rounds. More = better but slower.
POPULATION_SIZE = 16   # Genomes tested per generation.
GAMES_PER_MATCH = 50  # Games per genome. More = more accurate win rate.
MUTATION_RATE   = 0.3  # Chance each gene mutates (0.0 to 1.0).
MUTATION_STRENGTH = 0.3  # How much each gene changes when it mutates.

# Which genes to evolve (all keys in base_genome)
GENOME_KEYS = [
    'bonus_3_est', 'bonus_4_est', 'bonus_5_est',
    'luxury_mult', 'cheap_mult', 'pressure_weight',
    'camel_min_util', 'camel_take_val', 'trade_set_bonus',
    'luxury_take_add', 'set_break_penalty', 'denial_weight',
]

# Starting genome (SharkAgent6 trained values)
STARTING_GENOME = {
    'bonus_3_est': 0.461,
    'bonus_4_est': 20.2,
    'bonus_5_est': 30.0,
    'luxury_mult': 0.346,
    'cheap_mult': 0.488,
    'pressure_weight': 0.579,
    'camel_min_util': 4.8,
    'camel_take_val': 2.0,
    'trade_set_bonus': 5.23,
    'luxury_take_add': 0.316,
    'set_break_penalty': 35.4,
    'denial_weight': 0.45,
}


# ===========================================================================
# GAME RUNNER
# Plays one game between TrainableStarAgent (hero) and an opponent (villain).
# Returns 1 if hero wins, 0 otherwise.
# ===========================================================================
def play_one_game(args):
    seed, genome = args

    hero = TrainableStarAgent(seed, "StarAgent", genome=genome)

    # 70% of games vs SharkAgent6, 30% vs SmartAgent (if available)
    if SmartAgent is not None and random.random() < 0.3:
        villain = SmartAgent(seed + 5000, "SmartAgent")
    else:
        villain = SharkAgent6(seed + 9999, "SharkAgent6")

    if not hasattr(hero, 'uuid'):    hero.uuid = uuid.uuid4()
    if not hasattr(villain, 'uuid'): villain.uuid = uuid.uuid4()

    players = {hero.uuid: hero, villain.uuid: villain}
    order = [hero, villain] if (seed % 2 == 0) else [villain, hero]

    try:
        game = BasicBazaar(seed=seed, players=order)
        state = game.state

        while not game.terminal(state):
            actor = state.actor
            actions = game.all_actions(actor, state)
            if not actions:
                break
            observation = game.observe(actor, state)
            chosen = actor.select_action(actions, observation, lambda a: None)
            state = game.apply_action(state, chosen)
            game.state = state

        scores = {pid: game.calculate_reward(players[pid], state, state) for pid in players}
        return 1 if scores[hero.uuid] > scores[villain.uuid] else 0

    except Exception as e:
        return 0  # count crashes as losses


# ===========================================================================
# GENETIC ALGORITHM HELPERS
# ===========================================================================
def mutate(genome):
    """Return a slightly modified copy of the genome."""
    new_genome = copy.deepcopy(genome)
    for key in GENOME_KEYS:
        if random.random() < MUTATION_RATE:
            factor = 1.0 + random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
            new_genome[key] = new_genome[key] * factor
    return new_genome


def evaluate_population(population):
    """Run GAMES_PER_MATCH games for every genome in parallel. Returns list of win counts."""
    tasks = []
    for genome_idx, genome in enumerate(population):
        seed_base = genome_idx * GAMES_PER_MATCH
        for game_idx in range(GAMES_PER_MATCH):
            tasks.append((seed_base + game_idx, genome))

    cpu_count = min(multiprocessing.cpu_count(), 8)  # cap at 8 to avoid memory issues
    print(f"  Running {len(tasks)} games across {cpu_count} CPU cores...")

    with multiprocessing.Pool(cpu_count) as pool:
        results = pool.map(play_one_game, tasks)

    # Tally wins per genome
    wins_per_genome = [0] * len(population)
    for task_idx, win in enumerate(results):
        genome_idx = task_idx // GAMES_PER_MATCH
        wins_per_genome[genome_idx] += win

    return wins_per_genome


# ===========================================================================
# MAIN TRAINING LOOP
# ===========================================================================
def run_training():
    print("=" * 60)
    print("STAR AGENT GENETIC TRAINING")
    print(f"  Generations:      {GENERATIONS}")
    print(f"  Population size:  {POPULATION_SIZE}")
    print(f"  Games per genome: {GAMES_PER_MATCH}")
    print(f"  Total games:      {GENERATIONS * POPULATION_SIZE * GAMES_PER_MATCH:,}")
    print("=" * 60)

    # --- INITIALISE POPULATION ---
    # Start with mutations of the known-good genome so we don't search from scratch
    population = [mutate(STARTING_GENOME) for _ in range(POPULATION_SIZE)]
    population[0] = copy.deepcopy(STARTING_GENOME)  # always keep one unmodified copy

    best_ever_wins = 0
    best_ever_genome = copy.deepcopy(STARTING_GENOME)

    for gen in range(GENERATIONS):
        t_start = time.time()
        print(f"\n--- Generation {gen + 1} / {GENERATIONS} ---")

        wins = evaluate_population(population)

        # Sort genomes best → worst
        ranked = sorted(zip(wins, population), key=lambda x: x[0], reverse=True)
        top_wins, top_genome = ranked[0]
        elapsed = time.time() - t_start

        win_rate = top_wins / GAMES_PER_MATCH * 100
        print(f"  Best this gen:  {top_wins}/{GAMES_PER_MATCH} ({win_rate:.1f}% win rate)")
        print(f"  Time:           {elapsed:.1f}s")

        if top_wins > best_ever_wins:
            best_ever_wins = top_wins
            best_ever_genome = copy.deepcopy(top_genome)
            print(f"  *** NEW BEST GENOME! ***")

        # --- SELECTION: keep top 25% ---
        num_elites = max(2, POPULATION_SIZE // 4)
        elites = [genome for _, genome in ranked[:num_elites]]

        # --- REPRODUCTION: fill rest with mutations of elites ---
        next_gen = copy.deepcopy(elites)
        while len(next_gen) < POPULATION_SIZE:
            parent = random.choice(elites)
            next_gen.append(mutate(parent))

        population = next_gen

    # --- PRINT FINAL RESULT ---
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best win rate: {best_ever_wins}/{GAMES_PER_MATCH} "
          f"({best_ever_wins / GAMES_PER_MATCH * 100:.1f}%)")
    print("=" * 60)
    print("\nCopy this genome into star_agent.py inside self.base_genome = { ... }:\n")
    print("self.base_genome = {")
    for key, val in best_ever_genome.items():
        print(f"    '{key}': {val},")
    print("}")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed on Windows
    run_training()
