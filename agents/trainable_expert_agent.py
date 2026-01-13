import random
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

class TrainableExpertAgent(Trader):
    """
    The 'Learning' version of your Expert Agent.
    It accepts a 'genome' dictionary to tune its strategy.
    """
    def __init__(self, seed, name, genome=None):
        super().__init__(seed, name)
        
        # DEFAULT WEIGHTS (Starting Point)
        self.params = {
            'val_diamond': 7.0, 'val_gold': 6.0, 'val_silver': 5.0,
            'val_fabric': 4.0, 'val_spice': 4.0, 'val_leather': 1.5,
            'val_camel': 0.5,
            'sell_luxury_mult': 2.0,
            'sell_endgame_mult': 5.0,
            'bonus_5_add': 30.0,
            'bonus_4_add': 15.0,
            'hand_pressure_high': 20.0,
            'camel_min_utility': 25.0,
        }
        
        # Override defaults with evolved genes
        if genome:
            self.params.update(genome)
            
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
        # 1. CONTEXT ANALYSIS
        current_hand_size = observation.actor_goods.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        cards_left = observation.market_reserved_goods_count
        is_endgame = cards_left <= 4
        
        pressure = 0
        if current_hand_size >= hand_limit:
            pressure = self.params['hand_pressure_high'] * 5 
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
                
        return best_action if best_action else self.rng.choice(actions)

    def _score_sell(self, action, obs, is_endgame, pressure):
        good = action._sell
        count = action._count
        tokens = obs.market_goods_coins.get(good, [])
        points = sum(tokens[-count:]) if len(tokens) >= count else sum(tokens)
        top_token = tokens[-1] if tokens else 0

        if is_endgame: return points * self.params['sell_endgame_mult']
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            return (points * self.params['sell_luxury_mult']) + pressure + 10

        if count >= 5: return (points * 1.5) + self.params['bonus_5_add'] + pressure
        if count == 4: return (points * 1.5) + self.params['bonus_4_add'] + pressure
        if count == 2 and top_token >= 5: return (points * 2.0) + 10 + pressure
        return points + pressure - 5.0

    def _score_take(self, action, obs, is_endgame, pressure):
        good = action._take
        if good == GoodType.CAMEL:
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2: return self.params['camel_min_utility']
            if my_camels > 5: return -10.0
            if action._count >= 4: return 5.0 
            return 10.0 + action._count

        current_hand = obs.actor_goods[good]
        base_val = self.base_values[good]
        tight_hand_penalty = pressure if pressure > 0 else 0
        
        if good in [GoodType.DIAMOND, GoodType.GOLD]: return 40.0 - tight_hand_penalty
        if current_hand == 3: return 35.0 - tight_hand_penalty
        if current_hand == 4: return 45.0 - tight_hand_penalty
        if current_hand == 1 and good != GoodType.LEATHER: return 20.0 - tight_hand_penalty
        return base_val - tight_hand_penalty

    def _score_trade(self, action, obs, is_endgame):
        req = action.requested_goods
        off = action.offered_goods
        val_in = sum(self.base_values[g] * req[g] for g in GoodType)
        val_out = sum(self.base_values[g] * off[g] for g in GoodType)
        
        # Veto giving away luxury
        if (off[GoodType.DIAMOND] + off[GoodType.GOLD]) > 0: return -100
        
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_created = count_out - count_in
        
        if obs.actor_goods.count(include_camels=False) >= 6:
            val_in += (space_created * 15.0)
            
        return val_in - val_out - 5.0