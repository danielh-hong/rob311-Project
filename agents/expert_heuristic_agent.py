# trained a bit
import random
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType
from bazaar_ai.market import MarketObservation

class ExpertHeuristicAgent(Trader):
    """
    The 'Production' Agent.
    
    Strategy: Evolved Heuristic (Gen 10).
    Performance: ~65-70% Win Rate vs SmartAgent.
    
    Key Attributes:
    - High Valuation on Gold/Silver (Counter-Meta).
    - Massive weight on 5-card bonuses (+33.5 pts).
    - Low panic threshold (Hand Pressure ~17).
    """
    
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
        # --- GEN 10 EVOLVED WEIGHTS ---
        self.params = {
            'val_diamond': 5.8499, # Lower than base (Let opponent fight for them)
            'val_gold': 6.5379,    # Higher than base (We corner this market)
            'val_silver': 6.0079,
            'val_fabric': 4.0697,
            'val_spice': 4.7912,
            'val_leather': 1.5,
            'val_camel': 0.5,
            
            'sell_luxury_mult': 2.0,
            'sell_endgame_mult': 5.0,
            'bonus_5_add': 33.5813, # The secret sauce: Wait for 5 cards
            'bonus_4_add': 10.1414,
            'hand_pressure_high': 17.3894,
            'camel_min_utility': 16.5249
        }
        # ------------------------------

        # Map for speed
        self.base_values = {
            GoodType.DIAMOND: self.params['val_diamond'],
            GoodType.GOLD: self.params['val_gold'],
            GoodType.SILVER: self.params['val_silver'],
            GoodType.FABRIC: self.params['val_fabric'],
            GoodType.SPICE: self.params['val_spice'],
            GoodType.LEATHER: self.params['val_leather'],
            GoodType.CAMEL: self.params['val_camel']
        }

    def select_action(self, actions, observation, simulate_action_fnc):
        # 1. ANALYZE STATE
        current_hand_size = observation.actor_goods.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        cards_left = observation.market_reserved_goods_count
        is_endgame = cards_left <= 4
        
        # Calculate Pressure (Urgency to clear hand)
        pressure = 0
        if current_hand_size >= hand_limit:
            pressure = self.params['hand_pressure_high'] * 5 # Panic!
        elif current_hand_size >= hand_limit - 1:
            pressure = self.params['hand_pressure_high']

        best_action = None
        best_score = float('-inf')

        # 2. SCORE ACTIONS
        for action in actions:
            score = 0
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, is_endgame, pressure)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, is_endgame, pressure)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, is_endgame)
            
            if score > best_score:
                best_score = score
                best_action = action
                
        # 3. SAFETY FALLBACK
        # If no action was scored (should be impossible), pick safest: Sell -> Take -> Random
        if not best_action:
            sells = [a for a in actions if isinstance(a, SellAction)]
            takes = [a for a in actions if isinstance(a, TakeAction)]
            if sells: return sells[0]
            if takes: return takes[0]
            return self.rng.choice(actions)

        return best_action

    def _score_sell(self, action, obs, is_endgame, pressure):
        good = action._sell
        count = action._count
        
        # Token Valuation
        tokens = obs.market_goods_coins.get(good, [])
        if len(tokens) >= count:
            points = sum(tokens[-count:])
            top_token = tokens[-1]
        else:
            points = sum(tokens)
            top_token = 0

        # Logic
        if is_endgame: 
            return points * self.params['sell_endgame_mult']

        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            # Multiplier + Pressure ensures we bank these points
            return (points * self.params['sell_luxury_mult']) + pressure + 10

        # The "Big Bonus" Strategy
        if count >= 5: 
            return (points * 1.5) + self.params['bonus_5_add'] + pressure
        if count == 4: 
            return (points * 1.5) + self.params['bonus_4_add'] + pressure
        
        # Sniper: Sell 2 if it steals a 5+ point token
        if count == 2 and top_token >= 5: 
            return (points * 2.0) + 10 + pressure
            
        return points + pressure - 5.0

    def _score_take(self, action, obs, is_endgame, pressure):
        good = action._take
        
        if good == GoodType.CAMEL:
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2: return self.params['camel_min_utility']
            if my_camels > 5: return -10.0
            if action._count >= 4: return 5.0 # Caution with big camel takes
            return 10.0 + action._count

        current_hand = obs.actor_goods[good]
        base_val = self.base_values[good]
        tight_hand_penalty = pressure if pressure > 0 else 0
        
        # Always take Luxury
        if good in [GoodType.DIAMOND, GoodType.GOLD]:
            return 40.0 - tight_hand_penalty
            
        # Set Building
        if current_hand == 3: return 35.0 - tight_hand_penalty # Target 4
        if current_hand == 4: return 45.0 - tight_hand_penalty # Target 5
        
        # Speculative Take (Start new set)
        if current_hand == 1 and good != GoodType.LEATHER: 
            return 20.0 - tight_hand_penalty
        
        return base_val - tight_hand_penalty

    def _score_trade(self, action, obs, is_endgame):
        req = action.requested_goods
        off = action.offered_goods
        
        val_in = sum(self.base_values[g] * req[g] for g in GoodType)
        val_out = sum(self.base_values[g] * off[g] for g in GoodType)
        
        # Safety Veto
        if (off[GoodType.DIAMOND] + off[GoodType.GOLD]) > 0:
            return -100

        # Space Management
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_created = count_out - count_in
        
        # If hand is full, trading to create space is valuable
        current_hand_size = obs.actor_goods.count(include_camels=False)
        if current_hand_size >= 6:
            val_in += (space_created * 15.0)
            
        return val_in - val_out - 5.0