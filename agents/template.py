from bazaar_ai.trader import Trader, TraderAction
from bazaar_ai.market import MarketObservation
from typing import Optional, Callable

class CustomAgent(Trader):
    def __init__(self, seed, name):
        super().__init__(seed, name)
        # Initialize any agent-specific data structures
        self.memory = []

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
        # Implement your decision logic here
        # You can use simulate_action_fnc to look ahead!
        
        for action in actions:
            future_state = simulate_action_fnc(action)
            # Evaluate this future state...
        
        return best_action

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
        # Update your agent's learning signals
        reward = self._compute_reward(old_observation, new_observation)
        self.memory.append((old_observation, new_observation, reward))