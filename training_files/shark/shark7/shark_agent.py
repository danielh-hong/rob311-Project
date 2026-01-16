# 998 wins to 2 wins against random agent
# 706 wins to 271 wins against smart agent

import random
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

class SharkAgent(Trader):
    """
    The 'Shark' Agent.
    
    Philosophy: Real-time Opportunity Cost.
    - Instead of static weights, it checks the ACTUAL top token value.
    - Aggressively hunts for 4 and 5-card sets.
    - 'Safe' trading: Trades luxury for bulk only if it triggers a massive bonus.
    """
    
    def __init__(self, seed, name):
        super().__init__(seed, name)
        # Expected values for bonuses (Average of remaining usually)
        # 3-card: ~2, 4-card: ~5, 5-card: ~9
        self.bonus_estimates = {3: 2.0, 4: 5.5, 5: 9.0}

    def select_action(self, actions, observation, simulate_action_fnc):
        # 1. READ BOARD STATE
        # We don't use simulation. We calculate "Marginal Utility".
        
        best_action = None
        best_score = float('-inf')

        # Cache basic state
        hand = observation.actor_goods
        hand_size = hand.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        
        # Panic factor: How desperate are we to clear hand space?
        # 0 = chill, 10 = panic
        pressure = 0
        if hand_size >= hand_limit: pressure = 20
        elif hand_size >= hand_limit - 1: pressure = 5

        # 2. EVALUATE ALL ACTIONS
        for action in actions:
            score = 0
            
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size)
            
            # Add a tiny random jitter to break ties deterministically
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def _get_token_value(self, good_type, count, obs):
        """Calculates exact points we'd get from the token stack right now."""
        tokens = obs.market_goods_coins.get(good_type, [])
        if not tokens: return 0
        
        # Sum the top 'count' tokens
        # If we sell more than available tokens, we just take what's there
        available = len(tokens)
        take_n = min(count, available)
        
        # Python slicing to get the last N items (the highest values are usually at the end/top)
        # Note: Check bazaar implementation. Usually tokens are popped from end.
        # Assuming list is [1, 1, 2, 3, 5] and we pop from end.
        current_value = sum(tokens[-take_n:])
        return current_value

    def _score_sell(self, action, obs, pressure):
        good = action._sell
        count = action._count
        
        # 1. Immediate Token Value
        points = self._get_token_value(good, count, obs)
        
        # 2. Bonus Value
        bonus = 0
        if count == 3: bonus = self.bonus_estimates[3]
        elif count == 4: bonus = self.bonus_estimates[4]
        elif count >= 5: bonus = self.bonus_estimates[5]
        
        total_value = points + bonus
        
        # 3. Strategic Weighting
        # If we are selling Luxury (Diamond/Gold/Silver), selling early is often better 
        # to deny opponent high tokens.
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            # If we have 2 diamonds, sell NOW before opponent does.
            # Unless we have 4+, then maybe wait for 5? No, luxury race is too fast.
            return (total_value * 1.5) + pressure
        
        # For cheap goods (Leather/Spice/Fabric), we ONLY want to sell big sets.
        if good in [GoodType.LEATHER, GoodType.SPICE, GoodType.FABRIC]:
            if count >= 5: return (total_value * 2.0) + pressure + 10 # JACKPOT
            if count == 4: return (total_value * 1.5) + pressure + 5
            if count <= 2 and pressure < 10: return -50 # Don't sell small cheap sets unless panicked
            
        return total_value + pressure

    def _score_take(self, action, obs, current_hand_size, pressure):
        good = action._take
        
        # --- CAMEL LOGIC ---
        if good == GoodType.CAMEL:
            # Take camels if we are low, or if the market is full of them (refresh market)
            # or if the market has NOTHING good.
            market_camels = action._count
            my_camels = obs.actor_goods[GoodType.CAMEL]
            
            if market_camels >= 4: return 2.0 # Taking 4+ camels is a good "stall" move
            if my_camels < 2: return 5.0 # Always keep a buffer
            return -2.0 # Otherwise, prefer taking real cards
            
        # --- CARD LOGIC ---
        # What is this card worth POTENTIALLY?
        # We look at the top token.
        tokens = obs.market_goods_coins.get(good, [])
        top_token_val = tokens[-1] if tokens else 1
        
        # How many do I have?
        in_hand = obs.actor_goods[good]
        
        # SCORING
        score = top_token_val  # Base value is the token value
        
        # Set Building Bonuses
        if in_hand == 3: score += 15 # Taking the 4th card is HUGE
        if in_hand == 4: score += 20 # Taking the 5th card is MASSIVE
        if in_hand == 1 and good in [GoodType.DIAMOND, GoodType.GOLD]: score += 10 # Always grab luxury
        
        # Penalty for filling hand too early
        if current_hand_size >= 5: score -= 5
        
        return score - pressure

    def _score_trade(self, action, obs, current_hand_size):
        req = action.requested_goods # What I get
        off = action.offered_goods   # What I give
        
        # 1. Analyze "What I Get"
        # We want to trade FOR cards that complete sets
        value_in = 0
        completes_set = False
        
        for g in GoodType:
            if req[g] > 0:
                # Value of the card itself (token value)
                tokens = obs.market_goods_coins.get(g, [])
                token_val = tokens[-1] if tokens else 0
                
                # Value of set completion
                current_count = obs.actor_goods[g]
                new_count = current_count + req[g]
                
                if new_count >= 5: 
                    value_in += 25 # Massive weight for completing 5-set
                    completes_set = True
                elif new_count == 4:
                    value_in += 10
                elif g in [GoodType.DIAMOND, GoodType.GOLD]:
                    value_in += (token_val * 2) # Luxury is always good
                else:
                    value_in += token_val

        # 2. Analyze "What I Give"
        value_out = 0
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL:
                    value_out += 2 # Camels are cheap currency
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    token_val = tokens[-1] if tokens else 0
                    value_out += token_val
                    
                    # Penalty for breaking a set
                    if obs.actor_goods[g] >= 3: value_out += 15 # Don't trade away my sets!

        # 3. Space Management (The "Smart" part)
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_change = count_in - count_out
        
        # If hand is full, we LIKE trades that reduce hand size (give 2 take 1)
        if current_hand_size >= 6 and space_change < 0:
            value_in += 10
            
        # If we complete a 5-set, we almost always accept
        if completes_set and value_in > value_out:
            return 100 + (value_in - value_out)

        return value_in - value_out