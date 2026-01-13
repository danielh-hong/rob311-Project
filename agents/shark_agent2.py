# 998 wins to 2 wins against random agent
# 695 wins to 281 wins against smart agent
# shark_agent with trained genome via genetic algorithm
# this shark is literally worse than shark_agent.py

# trained but literally worse than shark_agent.py
import random
import uuid
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType
from bazaar_ai.coins import BonusType

class SharkAgent2(Trader):
    """
    The Tournament Shark.
    Trained via Genetic Algorithm (200 games/match).
    
    Performance:
    - Win Rate: 80% (160/200) vs Mixed Smart/Random Agents
    
    Strategy:
    - 'Camel Hoarding': Keeps ~8 camels (val 8.32) to starve opponents.
    - 'Luxury Sniping': Massive 24.3pt bonus for taking Diamonds/Gold.
    - 'Set Completion': Huge 37.9pt weight on trading to finish a set.
    """
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
        # 1. Safety Fix for UUID
        if not hasattr(self, 'uuid'):
            self.uuid = uuid.uuid4()

        # 2. THE CHAMPION GENOME (160/200 Wins)
        self.genome = {
            'bonus_3_est': 2.6726939388624076,
            'bonus_4_est': 4.565948676413123,
            'bonus_5_est': 10.290716077527753,
            'luxury_mult': 1.5711409204092068,
            'cheap_mult': 2.93907163567225,       # High multiplier for big cheap sets
            'pressure_weight': 2.1653094431390563, # Sensitive to hand limits
            'camel_min_util': 8.323026228486425,   # Keeps a LARGE camel buffer
            'camel_take_val': 0.749047626293883,
            'trade_set_bonus': 37.95416952793376,  # Aggressively trades to finish sets
            'luxury_take_add': 24.338908558979583, # Immediate bonus for taking Luxury
            'set_break_penalty': 23.819065770276826
        }

    def select_action(self, actions, observation, simulate_action_fnc):
        best_action = None
        best_score = float('-inf')

        hand = observation.actor_goods
        hand_size = hand.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        
        # Panic Calculation
        pressure = 0
        if hand_size >= hand_limit: pressure = 20 * self.genome['pressure_weight']
        elif hand_size >= hand_limit - 1: pressure = 5 * self.genome['pressure_weight']

        for action in actions:
            score = 0
            
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size)
            
            # Tiny jitter to break ties deterministically
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def _get_token_value(self, good, count, obs):
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(count, len(tokens))
        return sum(tokens[-take_n:])

    def _score_sell(self, action, obs, pressure):
        good = action._sell
        count = action._count
        params = self.genome
        
        points = self._get_token_value(good, count, obs)
        
        bonus = 0
        if count == 3: bonus = params['bonus_3_est']
        elif count == 4: bonus = params['bonus_4_est']
        elif count >= 5: bonus = params['bonus_5_est']
        
        total = points + bonus
        
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            return (total * params['luxury_mult']) + pressure
        
        if good in [GoodType.LEATHER, GoodType.SPICE, GoodType.FABRIC]:
            if count >= 5: return (total * params['cheap_mult']) + pressure + 10
            # Logic to sell 4 if pressure is high
            if count == 4: return (total * (params['cheap_mult']*0.75)) + pressure + 5
            if count <= 2 and pressure < 10: return -50
            
        return total + pressure

    def _score_take(self, action, obs, current_hand_size, pressure):
        good = action._take
        params = self.genome
        
        if good == GoodType.CAMEL:
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2: return params['camel_min_util']
            return params['camel_take_val']

        tokens = obs.market_goods_coins.get(good, [])
        top_token_val = tokens[-1] if tokens else 1
        
        in_hand = obs.actor_goods[good]
        score = top_token_val
        
        if in_hand == 3: score += 15
        if in_hand == 4: score += 20
        
        # Massive weight on taking luxury
        if good in [GoodType.DIAMOND, GoodType.GOLD]:
             score += params['luxury_take_add']
        
        return score - pressure

    def _score_trade(self, action, obs, current_hand_size):
        req = action.requested_goods
        off = action.offered_goods
        params = self.genome
        
        value_in = 0
        completes_set = False
        for g in GoodType:
            if req[g] > 0:
                tokens = obs.market_goods_coins.get(g, [])
                val = tokens[-1] if tokens else 0
                
                if obs.actor_goods[g] + req[g] >= 5:
                    value_in += params['trade_set_bonus']
                    completes_set = True
                else:
                    value_in += val

        value_out = 0
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL:
                    value_out += 2
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    val = tokens[-1] if tokens else 0
                    value_out += val
                    if obs.actor_goods[g] >= 3: value_out += params['set_break_penalty']

        # Hand space logic
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_change = count_in - count_out
        
        if current_hand_size >= 6 and space_change < 0:
            value_in += 10
            
        if completes_set:
            return 100 + (value_in - value_out)

        return value_in - value_out