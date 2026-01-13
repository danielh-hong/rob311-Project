import random
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType
from bazaar_ai.coins import BonusType

class ApexAgent(Trader):
    """
    The 'Apex' Agent.
    
    Strategy: MAXIMUM TEMPO.
    1. Always takes Luxury (Diamond/Gold) if available.
    2. Sells 'Cheap' sets (Leather/Spice) fast (3-4 cards) to clear hand.
    3. Trades Camels aggressively to steal point cards.
    """
    
    def __init__(self, seed, name):
        super().__init__(seed, name)

    def select_action(self, actions, observation, simulate_action_fnc):
        # --- 1. PRE-COMPUTE STATE ---
        hand = observation.actor_goods
        hand_size = hand.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        
        # Check if 5-bonus exists (from Grandmaster logic)
        bonus_5_exists = observation.market_bonus_coins_counts[BonusType.FIVE] > 0

        best_action = None
        best_score = float('-inf')

        # --- 2. ACTION SCORING ---
        for action in actions:
            score = 0
            
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, bonus_5_exists)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, hand_limit)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size, bonus_5_exists)
            
            # Tiny random jitter to prevent stalemate loops
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def _get_stack_value(self, good, count, obs):
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(count, len(tokens))
        return sum(tokens[-take_n:])

    def _score_sell(self, action, obs, bonus_5_exists):
        good = action._sell
        count = action._count
        
        # 1. Base Value
        points = self._get_stack_value(good, count, obs)
        
        # 2. Bonus Estimate
        bonus = 0
        if count == 3: bonus = 2
        elif count == 4: bonus = 5
        elif count >= 5: bonus = 9
        
        total = points + bonus
        
        # --- STRATEGY ---
        
        # LUXURY: SELL IMMEDIATELY
        # If we have 2 Diamonds, that is 14 points. 
        # Waiting for a 3rd is risky (might never come). Bank it.
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            return (total * 3.0) + 10.0
            
        # CHEAP GOODS: SPEED IS LIFE
        # If we have 5, sell immediately.
        if count >= 5: return (total * 2.0) + 20.0
        
        # If we have 4, usually sell immediately unless we are SURE we can get a 5th.
        # Grandmaster waited. Apex sells.
        if count == 4: 
            if bonus_5_exists: return (total * 2.0) + 5.0 # Slight hesitation
            return (total * 2.0) + 15.0 # Sell now!
            
        # If we have 3, sell if we need space or if points are decent.
        if count == 3:
            return total + 5.0
            
        # Don't sell 1 or 2 cheap cards. Waste of turn.
        return -100.0

    def _score_take(self, action, obs, hand_size, hand_limit):
        good = action._take
        
        # --- CAMELS: THE "LAST RESORT" ---
        if good == GoodType.CAMEL:
            num_camels = action._count
            my_camels = obs.actor_goods[GoodType.CAMEL]
            
            # Only take camels if:
            # 1. We are completely out (need ammo)
            # 2. There is NOTHING else good on the board
            if my_camels == 0: return 10.0
            
            # Low priority otherwise. We want cards.
            return 1.0

        # --- CARDS: THE PRIORITY ---
        tokens = obs.market_goods_coins.get(good, [])
        val = tokens[-1] if tokens else 0
        
        # 1. LUXURY PRIORITY (Diamond/Gold)
        # Even if token value is low, taking it denies opponent.
        if good in [GoodType.DIAMOND, GoodType.GOLD]:
            return (val * 4.0) + 20.0 # MASSIVE WEIGHT. TAKE IT.
            
        # 2. SET BUILDING
        current = obs.actor_goods[good]
        synergy = 0
        if current == 3: synergy = 15 # Grab 4th
        if current == 4: synergy = 25 # Grab 5th
        
        # 3. HAND MANAGEMENT
        # If hand is full (7), we physically can't take (filtered by engine),
        # But if 6, taking is risky unless it's Diamond.
        space_penalty = 0
        if hand_size >= hand_limit - 1:
            if good not in [GoodType.DIAMOND, GoodType.GOLD]:
                space_penalty = 50.0 # Don't clog hand with leather
                
        return (val * 2.0) + synergy - space_penalty

    def _score_trade(self, action, obs, hand_size, bonus_5_exists):
        req = action.requested_goods
        off = action.offered_goods
        
        val_in = 0
        val_out = 0
        
        # CALCULATE IN (Greedy)
        for g in GoodType:
            if req[g] > 0:
                tokens = obs.market_goods_coins.get(g, [])
                val_in += tokens[-1] if tokens else 0
                
                # Bonus weight for luxury
                if g in [GoodType.DIAMOND, GoodType.GOLD]:
                    val_in += 10.0
                
                # Bonus for completing sets
                if obs.actor_goods[g] + req[g] >= 5: val_in += 30.0
        
        # CALCULATE OUT (Dump Camels)
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL:
                    val_out += 0.5 # Camels are trash to us, spend them!
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    val_out += tokens[-1] if tokens else 0
                    # Hate giving away real cards
                    val_out += 10.0 

        # SPACE CHECK
        # Trading Camels -> Goods fills hand.
        cards_in = req.count(include_camels=False)
        cards_out = off.count(include_camels=False)
        space_diff = cards_in - cards_out # Positive means filling hand
        
        if hand_size + space_diff > 7: return -1000 # Illegal/Bad
        if hand_size >= 6 and space_diff > 0: return -50 # Don't fill last slot via trade

        return val_in - val_out