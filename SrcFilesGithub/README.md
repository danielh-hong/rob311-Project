# Custom Agents Workspace

This folder is your workspace for creating custom Bazaar-AI agents.

## Quick Start

1. Create a new file in this folder (e.g., `my_agent.py`)
2. Define your agent class that extends `Trader`
3. Run the simulator: `bazaar-simulate`
4. Select your agent from the dropdown menu in the UI

## Minimal Agent Example

```python
from bazaar_ai.trader import Trader, TraderAction
from bazaar_ai.market import MarketObservation
from typing import Callable

class MyAgent(Trader):
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
    def select_action(self,
                      actions: list[TraderAction],
                      observation: MarketObservation,
                      simulate_action_fnc: Callable[[TraderAction], MarketObservation]) -> TraderAction:
        """Choose an action based on the current game state."""
        # Simple random strategy
        return self.rng.choice(actions)
    
    def calculate_reward(self,
                        old_observation: MarketObservation,
                        new_observation: MarketObservation,
                        has_acted: bool,
                        environment_reward: float):
        """Optional: update internal state for learning."""
        pass
```

## What You Have Access To

### In `observation` (MarketObservation):
- `observation.actor_goods` - Your hand of cards
- `observation.market_goods` - Cards currently in the market
- `observation.actor_goods_coins` - Your coin stacks (points earned)
- `observation.market_goods_coins` - Available coin stacks in the market
- `observation.actor_bonus_coins_counts` - Your bonus tokens earned
- `observation.market_bonus_coins_counts` - Available bonus tokens
- `observation.market_reserved_goods_count` - Cards remaining in the deck
- `observation.max_player_goods_count` - Max cards you can hold (usually 7)
- `observation.max_market_goods_count` - Max cards in market (usually 5)

### The `actions` parameter:
A list of all legal moves you can make. Each action is one of:
- `TakeAction` - Take 1 card or all camels from the market
- `SellAction` - Sell cards from your hand for coins
- `TradeAction` - Exchange cards with the market

### The `simulate_action_fnc` function:
Preview what the game state would look like after an action:
```python
for action in actions:
    future_state = simulate_action_fnc(action)
    # Evaluate this future state to pick the best action
```

## Strategy Ideas

1. **Random Baseline** - Pick actions randomly (see `RandomAgent`)
2. **Greedy** - Always sell when profitable, take high-value cards
3. **Lookahead** - Use `simulate_action_fnc` to evaluate future states
4. **Heuristic-Based** - Score actions based on rules (card value, timing, etc.)
5. **Learning-Based** - Use Q-learning, MCTS, or neural networks

## Built-in Examples

Check out the built-in agents for reference:
- `src/bazaar_ai/simulator/agents/random_agent.py` - Random baseline
- `src/bazaar_ai/simulator/agents/simple_agent.py` - Simple heuristics

## Tips

- Start simple (random or greedy) and add complexity gradually
- Use the simulator UI to watch your agent play and debug
- The `calculate_reward` method is optional but useful for learning agents
- Your agent files here are gitignored - they won't be committed to the repo

## Need Help?

- Check the [documentation](../docs/)
- Look at the built-in example agents
- Open an issue on GitHub

Happy trading! 🐪
