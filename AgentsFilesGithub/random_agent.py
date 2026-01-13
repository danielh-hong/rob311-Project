from bazaar_ai.trader import Trader

class RandomAgent(Trader):
    """
    A simple random agent that mirrors the base Trader class.
    Selects actions completely at random with no strategy.
    """
    
    def __init__(self, seed, name):
        super().__init__(seed, name)
    
    def select_action(self, actions, observation, simulate_action_fnc):
        """Randomly select an action from available options"""
        return self.rng.choice(actions)
    
    def calculate_reward(self, old_observation, new_observation, has_acted, environment_reward):
        """No reward calculation needed for random agent"""
        pass