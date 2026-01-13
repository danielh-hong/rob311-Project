import random
from bazaar_ai.trader import Trader, TraderAction, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType
from bazaar_ai.coins import BonusType
from bazaar_ai.market import MarketObservation

class ExpertHeuristicAgent(Trader):
    """
    The 'Forum Expert' Agent.
    
    Core Philosophy:
    1. SPEED: Sell Diamond/Gold/Silver INSTANTLY (Pairs are fine).
    2. GREED: Hoard Leather/Spice/Fabric for 4/5-card bonuses...
    3. SAFETY: ...UNLESS our hand is full. Then panic sell to free space.
    4. SNIPING: Sell a single card if it steals a high-value (5+) token.
    """
    
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
        # Inherent value of cards (Mental model)
        self.base_values = {
            GoodType.DIAMOND: 7.0,
            GoodType.GOLD: 6.0,
            GoodType.SILVER: 5.0,
            GoodType.FABRIC: 4.0, 
            GoodType.SPICE: 4.0,
            GoodType.LEATHER: 1.5,
            GoodType.CAMEL: 0.5 
        }

    def select_action(self, actions, observation, simulate_action_fnc):
        
        # --- CONTEXT AWARENESS ---
        
        # 1. How full is our hand? (Max is usually 7)
        # We need to exclude camels from hand count as they don't count towards limit
        current_hand_size = observation.actor_goods.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        
        # Pressure grows exponentially: 0 at 4 cards, 10 at 6 cards, 50 at 7 cards.
        hand_pressure = 0
        if current_hand_size >= hand_limit:
            hand_pressure = 100.0 # EMERGENCY: We MUST sell or we can't Take
        elif current_hand_size >= hand_limit - 1:
            hand_pressure = 20.0  # High pressure
        elif current_hand_size >= hand_limit - 2:
            hand_pressure = 5.0   # Mild pressure
            
        # 2. Is the game ending?
        cards_left = observation.market_reserved_goods_count
        is_endgame = cards_left <= 4
        
        # --- SCORE ACTIONS ---
        best_action = None
        best_score = float('-inf')
        
        for action in actions:
            score = 0
            
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, is_endgame, hand_pressure)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, is_endgame, hand_pressure)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, is_endgame)
                
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else self.rng.choice(actions)

    def _score_sell(self, action: SellAction, obs: MarketObservation, is_endgame: bool, pressure: float) -> float:
        good = action._sell
        count = action._count
        
        # 1. BASE VALUE (Immediate Points)
        # Get the actual token values we would win
        tokens = obs.market_goods_coins.get(good, [])
        # If tokens are [1, 1, 2, 5], and we sell 2, we take 5 and 2 (from end)
        if len(tokens) >= count:
            points = sum(tokens[-count:])
            top_token = tokens[-1]
        else:
            points = sum(tokens)
            top_token = 0
            
        # 2. STRATEGIC MULTIPLIERS
        
        # A. ENDGAME: Sell everything. Points now > Points later.
        if is_endgame:
            return points * 5.0
            
        # B. LUXURY (Diamond/Gold/Silver): Sell ASAP.
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            # If we sell 2 diamonds for 14 pts, that's huge. 
            # Add 'pressure' to ensure we prioritize selling over taking weak cards.
            return (points * 2.0) + pressure + 10
            
        # C. COMMON (Leather/Spice/Fabric): The delicate balance.
        else:
            # Bonus Values: 3 cards=+1, 4 cards=+4, 5 cards=+8 (approx)
            
            # CASE 1: The "Jackpot" (5 cards)
            if count >= 5: return (points * 1.5) + 30 + pressure
            
            # CASE 2: The "Solid Play" (4 cards)
            if count == 4: return (points * 1.5) + 15 + pressure
            
            # CASE 3: The "Early Sniper" (2 cards)
            # If we can snag a 5-point token with just a pair, DO IT.
            if count == 2 and top_token >= 5:
                return (points * 2.0) + 10 + pressure

            # CASE 4: The "Hand Dump" (1-3 cards, low value)
            # Normally we hate this. But if hand_pressure is high, we MUST do it.
            # If pressure is 100 (Full Hand), we essentially return 100+points, 
            # which beats any Trade/Take action.
            return points + pressure - 5.0 # -5 penalty acts as "reluctance"

    def _score_take(self, action: TakeAction, obs: MarketObservation, is_endgame: bool, pressure: float) -> float:
        # If hand is full, we literally cannot take (unless it's camels, maybe).
        # But usually the simulator filters these out.
        # If we are close to full, we should hate taking unless it's amazing.
        
        good = action._take
        
        if good == GoodType.CAMEL:
            camel_count = action._count
            my_camels = obs.actor_goods[GoodType.CAMEL]
            
            # Panic grab camels if we are broke
            if my_camels < 2: return 25.0
            # Don't grab if we have plenty
            if my_camels > 5: return -10.0
            # Don't grab big piles if market is dangerous
            if camel_count >= 4: return 5.0 
            
            return 10.0 + camel_count

        # Taking Goods
        current_hand = obs.actor_goods[good]
        base_val = self.base_values[good]
        
        # MODIFIER: If hand is tight, taking a card MUST complete a set or be high value.
        tight_hand_penalty = 0
        if pressure > 10: tight_hand_penalty = 20 # Strongly discourage taking when full
        
        # 1. Luxury is always good
        if good in [GoodType.DIAMOND, GoodType.GOLD]:
            return 40.0 - tight_hand_penalty
            
        # 2. Completing a Set (The "Magic Numbers")
        # If we have 3, taking 1 makes 4 (Bonus territory).
        if current_hand == 3: return 35.0 - tight_hand_penalty
        # If we have 4, taking 1 makes 5 (Jackpot).
        if current_hand == 4: return 45.0 - tight_hand_penalty
        # If we have 1 and it's Spice/Fabric, making a pair opens up the "Sniper" option.
        if current_hand == 1 and good != GoodType.LEATHER: return 20.0 - tight_hand_penalty
        
        # 3. Speculative Taking
        return base_val - tight_hand_penalty

    def _score_trade(self, action: TradeAction, obs: MarketObservation, is_endgame: bool) -> float:
        # Trading is the safety valve.
        # If hand is full, trading 2 junk cards for 1 good card is AMAZING.
        
        req = action.requested_goods
        off = action.offered_goods
        
        # Calculate net value
        val_in = sum(self.base_values[g] * req[g] for g in GoodType)
        val_out = sum(self.base_values[g] * off[g] for g in GoodType)
        
        # Did we get Luxury?
        got_luxury = (req[GoodType.DIAMOND] + req[GoodType.GOLD]) > 0
        
        # Did we give Luxury?
        gave_luxury = (off[GoodType.DIAMOND] + off[GoodType.GOLD]) > 0
        if gave_luxury: return -100 # Never
        
        # HAND MANAGEMENT BONUS
        # If we give more cards than we take, we free up space.
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_created = count_out - count_in
        
        # If hand is full, creating space is worth GOLD.
        current_hand_size = obs.actor_goods.count(include_camels=False)
        if current_hand_size >= 6:
            val_in += (space_created * 15.0)
            
        # Trade Penalty (Time cost)
        # If we are just swapping 1 for 1 with no value gain, it's a waste.
        time_penalty = 5.0
        
        return val_in - val_out - time_penalty