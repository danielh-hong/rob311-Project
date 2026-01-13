from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType
from bazaar_ai.coins import BonusType

class SearchAgent(Trader):
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
    def select_action(self, actions, observation, simulate_action_fnc):
        # DON'T use simulate - just evaluate actions directly like SmartAgent does
        
        best_action = None
        best_score = float('-inf')
        
        for action in actions:
            # Evaluate action WITHOUT simulating
            score = self.score_action(action, observation)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else actions[0]
    
    def score_action(self, action, obs):
        """Score an action based on what it does"""
        score = 0.0
        
        # SELL actions - immediate points!
        if isinstance(action, SellAction):
            good_type = action._sell
            count = action._count
            
            # Get coins we'd receive
            available = obs.market_goods_coins.get(good_type, [])
            if len(available) >= count:
                coins_to_get = available[-count:]
                score += sum(coins_to_get) * 5  # Points are valuable!
                
                # Bonus for selling 3+
                if count >= 5:
                    score += 50
                elif count >= 4:
                    score += 30
                elif count >= 3:
                    score += 15
        
        # TAKE actions - get valuable cards
        elif isinstance(action, TakeAction):
            good_type = action._take
            
            if good_type == GoodType.DIAMOND:
                score += 35
            elif good_type == GoodType.GOLD:
                score += 30
            elif good_type == GoodType.SILVER:
                score += 25
            elif good_type == GoodType.FABRIC:
                score += 15
            elif good_type == GoodType.SPICE:
                score += 15
            elif good_type == GoodType.LEATHER:
                score += 10
            elif good_type == GoodType.CAMEL:
                score += 5
        
        # TRADE actions - lower priority
        elif isinstance(action, TradeAction):
            # Check if we're getting valuable stuff
            requested = action.requested_goods
            score += requested[GoodType.DIAMOND] * 20
            score += requested[GoodType.GOLD] * 18
            score += requested[GoodType.SILVER] * 15
            score += requested[GoodType.FABRIC] * 10
            score += requested[GoodType.SPICE] * 10
            score += requested[GoodType.LEATHER] * 5
        
        return score
    
    def calculate_reward(self, old_observation, new_observation, has_acted, environment_reward):
        pass