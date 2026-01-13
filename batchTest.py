# batchTest.py
# batch test two agents with parallel processing
# venv\Scripts\Activate.ps1
# python batchTest.py

from bazaar_ai.bazaar import BasicBazaar
from bazaar_ai.goods import GoodType
from agents.expert_heuristic_agent import ExpertHeuristicAgent
from agents.smart_agent import SmartAgent
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def play_single_game(agent1_class, agent2_class, game_num, seed):
    """
    Play a single game and return results.
    This runs in a separate process.
    """
    # Create agents
    agent1 = agent1_class(seed=seed, name=f"{agent1_class.__name__}")
    agent2 = agent2_class(seed=seed+1000, name=f"{agent2_class.__name__}")
    
    # Create and play game
    players = [agent1, agent2]
    game = BasicBazaar(seed=seed, players=players)
    game.play()
    
    # Get final scores
    final_state = game.state
    
    score1 = 0
    for good_coins in final_state.player_coins[agent1].goods_coins.values():
        score1 += sum(good_coins)
    for bonus_coins in final_state.player_coins[agent1].bonus_coins.values():
        score1 += sum(bonus_coins)
    
    score2 = 0
    for good_coins in final_state.player_coins[agent2].goods_coins.values():
        score2 += sum(good_coins)
    for bonus_coins in final_state.player_coins[agent2].bonus_coins.values():
        score2 += sum(bonus_coins)
    
    # Camel bonus
    camel1 = final_state.player_goods[agent1][GoodType.CAMEL]
    camel2 = final_state.player_goods[agent2][GoodType.CAMEL]
    if camel1 > camel2:
        score1 += 5
    elif camel2 > camel1:
        score2 += 5
    
    # Determine winner
    if score1 > score2:
        result = "WIN"
    elif score2 > score1:
        result = "LOSS"
    else:
        result = "TIE"
    
    return {
        'game_num': game_num,
        'score1': score1,
        'score2': score2,
        'result': result
    }

def run_games(agent1_class, agent2_class, num_games=100, seed_start=0, max_workers=None):
    """
    Run multiple games in parallel between two agents.
    
    Args:
        agent1_class: First agent class
        agent2_class: Second agent class
        num_games: Number of games to play
        seed_start: Starting seed for reproducibility
        max_workers: Number of parallel processes (default: CPU count)
    """
    
    agent1_wins = 0
    agent2_wins = 0
    ties = 0
    
    agent1_total_score = 0
    agent2_total_score = 0
    
    print(f"\n{'='*60}")
    print(f"Battle: {agent1_class.__name__} vs {agent2_class.__name__}")
    print(f"Playing {num_games} games in parallel...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Create list of game tasks
    tasks = [(agent1_class, agent2_class, game_num, seed_start + game_num) 
             for game_num in range(num_games)]
    
    # Run games in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all games
        futures = {executor.submit(play_single_game, *task): task for task in tasks}
        
        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            
            game_num = result['game_num']
            score1 = result['score1']
            score2 = result['score2']
            outcome = result['result']
            
            # Update stats
            agent1_total_score += score1
            agent2_total_score += score2
            
            if outcome == "WIN":
                agent1_wins += 1
            elif outcome == "LOSS":
                agent2_wins += 1
            else:
                ties += 1
            
            # Print game result
            print(f"Game {game_num + 1}/{num_games} finished: "
                  f"{agent1_class.__name__} {score1}-{score2} {agent2_class.__name__} "
                  f"({outcome})")
    
    elapsed_time = time.time() - start_time
    
    # Final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Time taken: {elapsed_time:.1f} seconds ({elapsed_time/num_games:.2f}s per game)")
    print(f"\n{agent1_class.__name__}: {agent1_wins} wins ({agent1_wins/num_games*100:.1f}%)")
    print(f"{agent2_class.__name__}: {agent2_wins} wins ({agent2_wins/num_games*100:.1f}%)")
    print(f"Ties: {ties}")
    print(f"\nAverage Scores:")
    print(f"{agent1_class.__name__}: {agent1_total_score/num_games:.1f}")
    print(f"{agent2_class.__name__}: {agent2_total_score/num_games:.1f}")
    print(f"{'='*60}\n")
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'ties': ties,
        'agent1_avg_score': agent1_total_score / num_games,
        'agent2_avg_score': agent2_total_score / num_games,
        'time_elapsed': elapsed_time
    }

if __name__ == "__main__":
    # Run with automatic CPU count (uses all cores)
    results = run_games(ExpertHeuristicAgent, SmartAgent, num_games=100)
    
    # Or specify number of workers:
    # results = run_games(ExpertHeuristicAgent, SmartAgent, num_games=100, max_workers=4)
