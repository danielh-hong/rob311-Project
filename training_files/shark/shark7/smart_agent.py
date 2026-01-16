from bazaar_ai.trader import Trader, TraderAction, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

class SmartAgent(Trader):
    """
    A strategic Bazaar agent that prioritizes:
    1. Selling high-value goods when profitable
    2. Collecting bonus tokens (3+, 4+, 5+ cards)
    3. Taking valuable goods from market
    4. Trading efficiently to maintain hand size
    """
    
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
        # Good value priorities (higher = more valuable)
        self.good_values = {
            GoodType.DIAMOND: 7,
            GoodType.GOLD: 6,
            GoodType.SILVER: 5,
            GoodType.FABRIC: 4,
            GoodType.SPICE: 4,
            GoodType.LEATHER: 3,
            GoodType.CAMEL: 1
        }
    
    def select_action(self, actions, observation, simulate_action_fnc):
        """Select the best action based on strategic evaluation"""
        
        # Separate actions by type
        sell_actions = [a for a in actions if isinstance(a, SellAction)]
        take_actions = [a for a in actions if isinstance(a, TakeAction)]
        trade_actions = [a for a in actions if isinstance(a, TradeAction)]
        
        # Evaluate all actions and pick the best
        best_action = None
        best_score = float('-inf')
        
        # 1. Evaluate selling actions (highest priority if we can get bonus)
        for action in sell_actions:
            score = self._evaluate_sell_action(action, observation)
            if score > best_score:
                best_score = score
                best_action = action
        
        # 2. Evaluate taking actions
        for action in take_actions:
            score = self._evaluate_take_action(action, observation)
            if score > best_score:
                best_score = score
                best_action = action
        
        # 3. Evaluate trading actions (lower priority)
        for action in trade_actions:
            score = self._evaluate_trade_action(action, observation)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else self.rng.choice(actions)
    
    def _evaluate_sell_action(self, action, observation):
        """Evaluate the value of a sell action"""
        good_type = action._sell
        count = action._count
        
        # Get the coin values we'd receive
        available_coins = observation.market_goods_coins.get(good_type, [])
        if len(available_coins) < count:
            return -1000  # Can't sell if not enough coins
        
        # Calculate immediate coin value
        coins_to_receive = available_coins[-count:]
        coin_value = sum(coins_to_receive)
        
        # Bonus for selling 3+, 4+, or 5+ cards (bonus tokens)
        bonus_multiplier = 1.0
        if count >= 5:
            bonus_multiplier = 3.0  # Highest priority for 5-card bonus
        elif count >= 4:
            bonus_multiplier = 2.5
        elif count >= 3:
            bonus_multiplier = 2.0
        
        # Premium goods (diamond, gold, silver) should be sold quickly
        # as their coins deplete fast
        scarcity_bonus = 0
        if good_type in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            scarcity_bonus = len(available_coins) * 2  # More valuable when more coins left
        
        # Calculate total score
        score = (coin_value * bonus_multiplier) + scarcity_bonus
        
        return score
    
    def _evaluate_take_action(self, action, observation):
        """Evaluate the value of a take action"""
        good_type = action._take
        count = action._count
        
        # Taking camels
        if good_type == GoodType.CAMEL:
            # Camels are useful but low priority unless we're close to end
            camel_value = 5 if observation.market_reserved_goods_count < 15 else 2
            return camel_value
        
        # Don't take if hand is too full (need room for selling later)
        hand_space = observation.max_player_goods_count - observation.actor_non_camel_goods_count
        if hand_space <= 2:
            return -500  # Avoid filling hand completely
        
        # Value based on good type
        good_value = self.good_values.get(good_type, 1)
        
        # Check how many coins are left in market for this good
        available_coins = observation.market_goods_coins.get(good_type, [])
        coins_remaining = len(available_coins)
        
        # Higher value if we're close to having enough to sell for bonus
        actor_goods = observation.actor_goods
        cards_of_type = actor_goods[good_type]
        
        bonus_potential = 0
        if cards_of_type + count >= 5:
            bonus_potential = 30
        elif cards_of_type + count >= 4:
            bonus_potential = 20
        elif cards_of_type + count >= 3:
            bonus_potential = 15
        
        # High-value goods with many coins left are prioritized
        coin_availability_bonus = coins_remaining * 2
        
        score = (good_value * 5) + bonus_potential + coin_availability_bonus
        return score
    
    def _evaluate_trade_action(self, action, observation):
        """Evaluate the value of a trade action"""
        requested = action.requested_goods
        offered = action.offered_goods
        
        # Calculate value difference
        value_gained = sum(
            self.good_values.get(good_type, 0) * requested[good_type]
            for good_type in GoodType
        )
        
        value_lost = sum(
            self.good_values.get(good_type, 0) * offered[good_type]
            for good_type in GoodType
        )
        
        net_value = value_gained - value_lost
        
        # Penalize trades that give away camels (we need them for end bonus)
        if offered[GoodType.CAMEL] > 0:
            net_value -= offered[GoodType.CAMEL] * 3
        
        # Bonus for getting closer to a set for selling
        actor_goods = observation.actor_goods
        set_bonus = 0
        
        for good_type in GoodType:
            if good_type == GoodType.CAMEL:
                continue
            
            new_count = actor_goods[good_type] - offered[good_type] + requested[good_type]
            
            # Reward if we get closer to 3, 4, or 5 cards
            if new_count >= 5:
                set_bonus += 15
            elif new_count >= 4:
                set_bonus += 10
            elif new_count >= 3:
                set_bonus += 5
        
        # Trades are generally lower priority than direct takes or sells
        score = (net_value * 3) + set_bonus - 5  # Small penalty for trading complexity
        
        return score
    
    def calculate_reward(self, old_observation, new_observation, has_acted, environment_reward):
        """Calculate reward (not used in this greedy agent, but required by interface)"""
        pass