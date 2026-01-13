import random
import time
import multiprocessing
import json
from copy import deepcopy

# IMPORTS (Ensure these files exist in the same folder)
try:
    from trainable_expert_agent import TrainableExpertAgent
    from smart_agent import SmartAgent
    from bazaar_ai.bazaar import BasicBazaar
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Ensure trainable_agent.py and smart_agent.py are in this folder.")
    exit()

# CONFIGURATION
POPULATION_SIZE = 20
GENERATIONS = 10
GAMES_PER_EVAL = 50   # Games per genome per generation
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.2

# BASELINE PARAMETERS
DEFAULT_PARAMS = {
    'val_diamond': 7.0, 'val_gold': 6.0, 'val_silver': 5.0,
    'val_fabric': 4.0, 'val_spice': 4.0, 'val_leather': 1.5,
    'val_camel': 0.5,
    'sell_luxury_mult': 2.0, 'sell_endgame_mult': 5.0,
    'bonus_5_add': 30.0, 'bonus_4_add': 15.0,
    'hand_pressure_high': 20.0, 'camel_min_utility': 25.0,
}

def evaluate_genome(args):
    """Run a mini-tournament for one genome."""
    genome, genome_id, seed_base = args
    total_margin = 0
    wins = 0
    
    for i in range(GAMES_PER_EVAL):
        seed = seed_base + i
        
        # Create Agents
        hero = TrainableExpertAgent(seed, "Hero", genome=genome)
        villain = SmartAgent(seed + 9999, "Villain")
        
        # Setup Game
        game = BasicBazaar(seed, [hero, villain])
        state = game.state
        
        # FAST GAME LOOP (No UI, No Print)
        while not game.terminal(state):
            try:
                # Hero Select
                if state.actor == hero:
                    action = hero.select_action(game.all_actions(hero, state), game.observe(hero, state), None)
                else:
                    action = villain.select_action(game.all_actions(villain, state), game.observe(villain, state), None)
            except:
                break # Crash logic (should not happen)
                
            state = game.apply_action(state, action)
            game.state = state  # *** CRITICAL FIX ***
            
        # Scoring
        s1 = game.calculate_reward(hero, state, state)
        s2 = game.calculate_reward(villain, state, state)
        
        total_margin += (s1 - s2)
        if s1 > s2: wins += 1

    return (genome_id, total_margin / GAMES_PER_EVAL, wins)

def mutate(genome):
    """Create a variation of the genome."""
    new_genome = deepcopy(genome)
    for key in new_genome:
        if random.random() < MUTATION_RATE:
            # Change value by +/- 20%
            change = 1.0 + random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
            new_genome[key] *= change
    return new_genome

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print(f"Starting Genetic Evolution: {GENERATIONS} Gens, {POPULATION_SIZE} Pop")
    
    # 1. Initialize Population
    population = [mutate(DEFAULT_PARAMS) for _ in range(POPULATION_SIZE)]
    # Add one pure default to ensure we don't get worse
    population[0] = DEFAULT_PARAMS 
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    for gen in range(GENERATIONS):
        t0 = time.time()
        
        # Prepare jobs
        seed_base = gen * 1000
        jobs = [(pop, i, seed_base) for i, pop in enumerate(population)]
        
        # Run Evaluation
        results = pool.map(evaluate_genome, jobs)
        
        # Sort by Margin (Score Difference)
        results.sort(key=lambda x: x[1], reverse=True)
        
        best_id, best_margin, best_wins = results[0]
        best_genome = population[best_id]
        
        print(f"Gen {gen+1}: Best Margin +{best_margin:5.1f} | Win Rate {best_wins}/{GAMES_PER_EVAL} | Time {time.time()-t0:.1f}s")
        
        # Selection (Survival of the Fittest)
        # Keep top 20% (Elites)
        num_elites = max(2, int(POPULATION_SIZE * 0.2))
        next_gen = [population[res[0]] for res in results[:num_elites]]
        
        # Fill rest with mutations of elites
        while len(next_gen) < POPULATION_SIZE:
            parent = random.choice(next_gen[:num_elites])
            next_gen.append(mutate(parent))
            
        population = next_gen

    print("\n" + "="*40)
    print("TRAINING COMPLETE. COPY THESE PARAMS:")
    print("="*40)
    print(json.dumps(best_genome, indent=4))