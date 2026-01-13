import random
import uuid
import copy
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

class OpponentTracker:
    """
    STATE ESTIMATION ENGINE
    -----------------------
    Tracks the opponent's hand size by observing public market changes.
    Uses strictly the API defined in goods.py and market.py.
    """
    def __init__(self):
        self.hand_size = 5  # Everyone starts with 5 cards
        self.last_tokens = {}
        self.last_market_camels = 0
        self.first_turn = True

    def update(self, obs):
        """
        Called every turn. Compares current observation to the last recorded state.
        """
        # 1. Initialize on first turn
        if self.first_turn:
            self._record_state(obs)
            self.first_turn = False
            return

        # 2. DETECT SALES (Did token stacks shrink?)
        # Access: obs.market_goods_coins (dict[GoodType, list[int]])
        cards_sold = 0
        current_tokens = obs.market_goods_coins
        
        for good, tokens in current_tokens.items():
            # If we haven't seen this good before, assume full stack
            prev_count = len(self.last_tokens.get(good, []))
            curr_count = len(tokens)
            
            # If tokens disappeared, they sold cards
            if curr_count < prev_count:
                cards_sold += (prev_count - curr_count)
        
        if cards_sold > 0:
            # Logic: Selling removes cards from hand
            self.hand_size = max(0, self.hand_size - cards_sold)
            self._record_state(obs)
            return

        # 3. DETECT CAMEL TAKING
        # Access: obs.market_goods (Goods object) -> Use [GoodType.CAMEL]
        curr_market_camels = obs.market_goods[GoodType.CAMEL]
        prev_market_camels = self.last_market_camels
        
        if curr_market_camels < prev_market_camels:
            # Logic: Taking camels does NOT change hand size (goods count)
            self._record_state(obs)
            return

        # 4. DETECT TAKE/TRADE
        # If they didn't sell or take camels, they likely took a card.
        # Heuristic: Assume +1 card unless they are at the limit (7), then +0.
        if self.hand_size < 7:
            self.hand_size += 1
        
        self._record_state(obs)

    def _record_state(self, obs):
        # Deepcopy to ensure we store value, not reference
        self.last_tokens = copy.deepcopy(obs.market_goods_coins)
        # Store just the integer count for camels
        self.last_market_camels = obs.market_goods[GoodType.CAMEL]


class SharkAgent4(Trader):
    """
    SHARK AGENT V4 (Fixed & Optimized)
    ----------------------------------
    - Uses state estimation (OpponentTracker) to guess opponent's hand.
    - Uses 80% win-rate genetic weights.
    - Fully compatible with Goods/Market API.
    """
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
        self.tracker = OpponentTracker()
        
        # Ensure UUID exists
        if not hasattr(self, 'uuid'): 
            self.uuid = uuid.uuid4()

        # The Proven Weights
        self.genome = {
            'bonus_3_est': 2.0, 'bonus_4_est': 5.5, 'bonus_5_est': 9.0,      
            'luxury_mult': 1.5, 'cheap_mult': 2.0, 'pressure_weight': 1.0,  
            'camel_min_util': 5.0, 'camel_take_val': 2.0, 'trade_set_bonus': 25.0, 
            'luxury_take_add': 10.0, 'set_break_penalty': 15.0 
        }

    def select_action(self, actions, observation, simulate_action_fnc):
        # 1. Update Tracker
        self.tracker.update(observation)
        
        # 2. Logic: Threat Detection
        opp_hand_est = self.tracker.hand_size
        opponent_threatening = (opp_hand_est >= 6)

        best_action = None
        best_score = float('-inf')

        # 3. Logic: Self Analysis
        # Access: obs.actor_goods is a Goods object
        # Use .count(include_camels=False)
        hand = observation.actor_goods
        hand_size = hand.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        
        pressure = 0
        if hand_size >= hand_limit: 
            pressure = 20 * self.genome['pressure_weight']
        elif hand_size >= hand_limit - 1: 
            pressure = 5 * self.genome['pressure_weight']

        # 4. Score Actions
        for action in actions:
            score = 0
            
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure, opponent_threatening)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size)
            
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def _get_token_value(self, good, count, obs):
        # Access: obs.market_goods_coins is a dict
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(count, len(tokens))
        return sum(tokens[-take_n:])

    def _score_sell(self, action, obs, pressure, opponent_threatening):
        # SellAction attributes: _sell (GoodType), _count (int)
        good = action._sell
        count = action._count
        params = self.genome
        
        points = self._get_token_value(good, count, obs)
        
        # RACE CONDITION: If they are threatening, panic sell 3+ sets
        race_bonus = 0
        if opponent_threatening and count >= 3:
            race_bonus = 5.0

        bonus = 0
        if count == 3: bonus = params['bonus_3_est']
        elif count == 4: bonus = params['bonus_4_est']
        elif count >= 5: bonus = params['bonus_5_est']
        
        total = points + bonus + race_bonus
        
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            return (total * params['luxury_mult']) + pressure
        
        if good in [GoodType.LEATHER, GoodType.SPICE, GoodType.FABRIC]:
            if count >= 5: return (total * params['cheap_mult']) + pressure + 10
            if count == 4: return (total * (params['cheap_mult']*0.75)) + pressure + 5
            if count <= 2 and pressure < 10: return -50
            
        return total + pressure

    def _score_take(self, action, obs, current_hand_size, pressure):
        # TakeAction attributes: _take (GoodType)
        good = action._take
        params = self.genome
        
        if good == GoodType.CAMEL:
            # Goods object access
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2: return params['camel_min_util']
            return params['camel_take_val']

        tokens = obs.market_goods_coins.get(good, [])
        top_token_val = tokens[-1] if tokens else 1
        
        in_hand = obs.actor_goods[good]
        score = top_token_val
        
        if in_hand == 3: score += 15
        if in_hand == 4: score += 20
        
        if good in [GoodType.DIAMOND, GoodType.GOLD]:
             score += params['luxury_take_add']
        
        return score - pressure

    def _score_trade(self, action, obs, current_hand_size):
        req = action.requested_goods
        off = action.offered_goods
        params = self.genome
        
        # 1. Calculate Value GAINED
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

        # 2. Calculate Value LOST
        value_out = 0
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL:
                    value_out += 2 
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    val = tokens[-1] if tokens else 0
                    value_out += val
                    
                    # --- THE FIX IS HERE ---
                    # Only apply huge penalty if we are breaking a GOOD set
                    if obs.actor_goods[g] >= 3: 
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                            value_out += params['set_break_penalty'] # Keep the -15 for luxury
                        else:
                            value_out += 2.0 # Only a tiny penalty for breaking junk sets

        # 3. Hand Management
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_change = count_in - count_out
        
        if current_hand_size >= 6 and space_change < 0:
            value_in += 10
            
        if completes_set:
            return 100 + (value_in - value_out)

        return value_in - value_out