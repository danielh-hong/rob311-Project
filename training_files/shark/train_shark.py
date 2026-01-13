import sys
import os
import uuid
import time
import random
import copy
import multiprocessing

# ---------------------------------------------------------
# PATH FIX: Ensures imports work from anywhere
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
# ---------------------------------------------------------

from bazaar_ai.bazaar import BasicBazaar
from agents.smart_agent import SmartAgent 
from agents.random_agent import RandomAgent
from parametric_shark import ParametricShark 

# ---------------------------------------------------------
# SETTINGS: PRODUCTION MODE
# ---------------------------------------------------------
GENERATIONS = 15          
POPULATION_SIZE = 20      
GAMES_PER_MATCH = 200     # High number to eliminate luck
MUTATION_RATE = 0.2       
MUTATION_STRENGTH = 0.3   

GENOME_KEYS = [
    'bonus_3_est', 'bonus_4_est', 'bonus_5_est',
    'luxury_mult', 'cheap_mult', 'pressure_weight',
    'camel_min_util', 'camel_take_val', 'trade_set_bonus',
    'luxury_take_add', 'set_break_penalty'
]

# ---------------------------------------------------------
# SAFE WRAPPERS (Prevents UUID Crashes)
# ---------------------------------------------------------
class SafeSmartAgent(SmartAgent):
    def __init__(self, seed, name):
        super().__init__(seed, name)
        if not hasattr(self, 'uuid'): self.uuid = uuid.uuid4()

class SafeRandomAgent(RandomAgent):
    def __init__(self, seed, name):
        super().__init__(seed, name)
        if not hasattr(self, 'uuid'): self.uuid = uuid.uuid4()

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def create_random_genome():
    return {
        'bonus_3_est': random.uniform(1.0, 4.0),
        'bonus_4_est': random.uniform(3.0, 8.0),
        'bonus_5_est': random.uniform(6.0, 15.0),
        'luxury_mult': random.uniform(1.0, 4.0),
        'cheap_mult': random.uniform(1.0, 3.0),
        'pressure_weight': random.uniform(0.5, 3.0),
        'camel_min_util': random.uniform(2.0, 10.0),
        'camel_take_val': random.uniform(-2.0, 5.0),
        'trade_set_bonus': random.uniform(15.0, 50.0),
        'luxury_take_add': random.uniform(5.0, 25.0),
        'set_break_penalty': random.uniform(5.0, 30.0)
    }

def play_game_safe(seed, genome):
    hero = ParametricShark(seed=seed, name="Hero", genome=genome)
    
    # --- 90% Smart Agent / 10% Random Agent ---
    if random.random() < 0.90:
        villain = SafeSmartAgent(seed=seed, name="SmartVillain")
    else:
        villain = SafeRandomAgent(seed=seed, name="RandomVillain")
    # ------------------------------------------

    # Ensure Hero has UUID
    if not hasattr(hero, 'uuid'): hero.uuid = uuid.uuid4()
    
    # Setup Match
    p1_starts = (seed % 2 == 0)
    players = {hero.uuid: hero, villain.uuid: villain}
    order = [hero, villain] if p1_starts else [villain, hero]
    
    game = BasicBazaar(seed=seed, players=order)
    state = game.state
    
    # Manual Loop (Stable)
    while not game.terminal(state):
        actor = state.actor
        actions = game.all_actions(actor, state)
        if not actions: break
        
        try:
            observation = game.observe(actor, state)
            chosen_action = actor.select_action(actions, observation, lambda a: None)
        except Exception:
            chosen_action = random.choice(actions)
            
        state = game.apply_action(state, chosen_action)
        game.state = state 
    
    scores = {pid: game.calculate_reward(players[pid], state, state) for pid in players}
    return 1 if scores[hero.uuid] > scores[villain.uuid] else 0

# Helper to unpack arguments for imap
def run_match_task(args):
    return play_game_safe(*args)

def evaluate_population(population):
    tasks = []
    for i, genome in enumerate(population):
        seed_start = i * GAMES_PER_MATCH
        for j in range(GAMES_PER_MATCH):
            tasks.append((seed_start + j, genome)) # args tuple
            
    # Use max cores
    cpu_count = min(20, multiprocessing.cpu_count())
    
    with multiprocessing.Pool(cpu_count) as pool:
        results = []
        total = len(tasks)
        
        # Use imap to print progress in REAL TIME
        for i, res in enumerate(pool.imap(run_match_task, tasks)):
            results.append(res)
            # Print every 5%
            if i % (total // 20) == 0:
                print(f"  > Progress: {i}/{total} games finished...")
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
    # 1. Random population
    population = [create_random_genome() for _ in range(POPULATION_SIZE)]
    
    # 2. SEED WITH YOUR 29/30 WINNER (Don't lose progress!)
    population[0] = {
        'bonus_3_est': 2.0, 'bonus_4_est': 5.5, 'bonus_5_est': 9.0, 
        'luxury_mult': 1.5, 'cheap_mult': 2.0, 'pressure_weight': 1.0, 
        'camel_min_util': 5.0, 'camel_take_val': 2.0, 'trade_set_bonus': 25.0, 
        'luxury_take_add': 10.0, 'set_break_penalty': 15.0
    }
    
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
        print(f"Best: {top_wins}/{GAMES_PER_MATCH} wins ({top_wins/GAMES_PER_MATCH*100:.1f}%)")
        print(f"Genome: {top_genome}")
        
        if top_wins > best_all_time[0]:
            best_all_time = (top_wins, top_genome)
            print(f"** NEW RECORD **")

        # Selection: Top 25%
        cutoff = max(2, int(POPULATION_SIZE * 0.25))
        survivors = [x[1] for x in scored_pop[:cutoff]]
        
        # Mutation
        new_pop = survivors[:]
        while len(new_pop) < POPULATION_SIZE:
            parent = random.choice(survivors)
            child = mutate(parent)
            new_pop.append(child)
        population = new_pop

    print("\n\n=== TRAINING COMPLETE ===")
    print(f"Best Win Rate: {best_all_time[0]}/{GAMES_PER_MATCH}")
    print("Paste this into your final shark agent:")
    print(best_all_time[1])

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_evolution()