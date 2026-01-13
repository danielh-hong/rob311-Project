from bazaar_ai.trader import Trader, TraderAction
from bazaar_ai.market import MarketObservation
from typing import Optional, Callable
import random

class TemplateAgent(Trader):
    """
    Template agent - basic working agent you can modify.
    """
    def __init__(self, seed, name):
        super().__init__(seed, name)
        # Initialize any agent-specific data structures
        self.memory = []
        random.seed(seed)

    def select_action(self,
                      actions: list[TraderAction],
                      observation: MarketObservation,
                      simulate_action_fnc: Callable[[TraderAction], MarketObservation]) -> TraderAction:
        """
        Choose an action based on the current market state.

        Args:
            actions: List of legal actions available
            observation: Current view of the market (your hand, market cards, etc.)
            simulate_action_fnc: Function to simulate what happens if you take an action

        Returns:
            The action you want to take
        """
        # Simple strategy: evaluate each action and pick the best
        best_action = None
        best_score = float('-inf')
        
        for action in actions:
            # Simulate what happens if we take this action
            future_state = simulate_action_fnc(action)
            
            # Evaluate this future state
            score = self.evaluate_state(future_state)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else random.choice(actions)
    
    def evaluate_state(self, observation: MarketObservation) -> float:
        """
        Simple heuristic evaluation of a game state.
        Returns a score (higher = better for us).
        """
        score = 0.0
        
        try:
            # Try to access various attributes
            # (We don't know exact attribute names yet, so this is a guess)
            
            # Prefer states where we have more points
            if hasattr(observation, 'my_score'):
                score += observation.my_score
            
            if hasattr(observation, 'opponent_score'):
                score -= observation.opponent_score
            
            # Prefer having cards in hand
            if hasattr(observation, 'my_hand'):
                score += len(observation.my_hand) * 0.5
            
        except Exception:
            # If we can't evaluate, return neutral score
            score = 0.0
        
        return score

    def calculate_reward(self,
                        old_observation: MarketObservation,
                        new_observation: MarketObservation,
                        has_acted: bool,
                        environment_reward: Optional[float]):
        """
        Calculate rewards and update any internal state.
        
        This is called after every turn (yours and your opponent's).
        Use it to update value estimates, store experiences, etc.

        Args:
            old_observation: Market state before the action
            new_observation: Market state after the action
            has_acted: True if this was your turn
            environment_reward: Optional reward from the game (e.g., points scored)
        """
        # Store experience for later analysis
        reward = environment_reward if environment_reward else 0.0
        self.memory.append({
            'old_obs': old_observation,
            'new_obs': new_observation,
            'was_our_turn': has_acted,
            'reward': reward
        })