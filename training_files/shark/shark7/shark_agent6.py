# prior to training: 408 wins to 565 wins against shark agent (worse than shark agent)
# prior to training: 609 wins to 363 wins against smart agent (better than smart agent)

# post training: 793 wins to 186 wins (21 draws) against smart agent (much better than smart agent)
# post training: 603 wins to 373 wins (24 draws) against shark agent (better than shark agent)

import random
import uuid
import copy
from collections import Counter
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

class PerfectTracker:
    """
    STATE ESTIMATION ENGINE
    -----------------------
    Tracks the opponent's hand contents by reading the public action log.
    
    How it works:
    - We start knowing nothing (5 Unknown Cards).
    - When they TAKE: We confirm exactly what card they have.
    - When they TRADE: We confirm what they took, and remove what they gave.
    - When they SELL: We remove those cards from our confirmed/unknown pools.
    """
    def __init__(self):
        self.confirmed_hand = Counter() # Specific cards we know they have
        self.unknown_cards = 5          # Cards from start of game we haven't seen
        self.hand_size = 5
        self.last_action_id = None      # To prevent double-counting the same turn

    def update(self, obs):
        # 1. Validation: Is there a new action to analyze?
        if obs.action is None:
            return

        # Prevent processing the same action twice (update is called every sim step)
        if id(obs.action) == self.last_action_id:
            return
        self.last_action_id = id(obs.action)

        # 2. Analyze the Move
        act = obs.action
        
        # CASE A: THEY SOLD
        if act.trader_action_type.value == "Sell":
            good = act._sell
            count = act._count
            
            self.hand_size -= count
            
            # Remove from our mental model
            # Logic: We remove from 'confirmed' first, then 'unknowns'
            known_count = self.confirmed_hand[good]
            remove_from_known = min(known_count, count)
            remove_from_unknown = count - remove_from_known
            
            self.confirmed_hand[good] -= remove_from_known
            self.unknown_cards = max(0, self.unknown_cards - remove_from_unknown)

        # CASE B: THEY TOOK (Single Card or Camels)
        elif act.trader_action_type.value == "Take":
            good = act._take
            if good == GoodType.CAMEL:
                # Taking camels does not increase hand size (goods)
                pass 
            else:
                self.hand_size += 1
                self.confirmed_hand[good] += 1

        # CASE C: THEY TRADED
        elif act.trader_action_type.value == "Trade":
            # Add what they TOOK (Requested) to confirmed hand
            for g in GoodType:
                count_in = act.requested_goods[g]
                if count_in > 0:
                    self.confirmed_hand[g] += count_in
            
            # Remove what they GAVE (Offered)
            for g in GoodType:
                count_out = act.offered_goods[g]
                if count_out > 0:
                    known_count = self.confirmed_hand[g]
                    remove_from_known = min(known_count, count_out)
                    remove_from_unknown = count_out - remove_from_known
                    
                    self.confirmed_hand[g] -= remove_from_known
                    self.unknown_cards = max(0, self.unknown_cards - remove_from_unknown)


class SharkAgent6(Trader):
    """
    SHARK AGENT V5 (The Grandmaster)
    --------------------------------
    Strategy:
    1. Perfect Tracking: Knows exactly what cards the opponent has.
    2. Calculated Denial: Calculates the exact point swing (EV) if the 
       opponent is allowed to take a specific card.
    3. Smart Trading: Will aggressively trade away 'Junk' sets but 
       hoard 'Luxury' sets.
    """
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
        self.tracker = PerfectTracker()
        
        if not hasattr(self, 'uuid'): 
            self.uuid = uuid.uuid4()

        # The "Balanced" Genome (Robust Baseline)
        # The "Evolved" Genome (Best from Gen 92 - 67.3% Win Rate)
        self.genome = {
            'bonus_3_est': 0.46101521927513794,
            'bonus_4_est': 20.195191234717484,
            'bonus_5_est': 30.00124098364798,
            'luxury_mult': 0.34644316548584847,
            'cheap_mult': 0.48812771349567025,
            'pressure_weight': 0.5787146757167929,
            'camel_min_util': 4.7995497582532805,
            'camel_take_val': 0.017704558639876328,
            'trade_set_bonus': 5.231116644985265,
            'luxury_take_add': 0.3161569808518768,
            'set_break_penalty': 35.389358948276374,
            'denial_weight': 0.061804398633300416
        }

    def select_action(self, actions, observation, simulate_action_fnc):
        # 1. Update Tracker
        self.tracker.update(observation)
        
        # 2. Analyze Opponent State
        opp_confirmed = self.tracker.confirmed_hand
        opp_hand_size = self.tracker.hand_size
        
        # Logic: If opponent is full (7 cards), they CANNOT 'Take'.
        # They MUST 'Trade' or 'Sell'. Trading is expensive, so the market is slightly safer.
        opponent_locked = (opp_hand_size >= 7)

        best_action = None
        best_score = float('-inf')

        # 3. Analyze Self
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
                score = self._score_sell(action, observation, pressure, opp_confirmed)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure, opp_confirmed, opponent_locked)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size)
            
            # Tiny jitter for determinism breaking
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    # --- SCORING LOGIC ---

    def _get_token_value(self, good, count, obs):
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(count, len(tokens))
        return sum(tokens[-take_n:])
    
    def _calculate_opponent_potential(self, good, opp_confirmed_count, obs):
        """
        Calculates the mathematical THREAT VALUE (Points) the opponent scores
        if they take this card and complete a set.
        """
        potential_count = opp_confirmed_count + 1
        
        # Only threatening if this lets them sell a set (3+)
        if potential_count < 3: return 0
            
        # Value of tokens they would take
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(potential_count, len(tokens))
        token_points = sum(tokens[-take_n:])
        
        # Value of bonus they would get
        bonus_points = 0
        if potential_count == 3: bonus_points = 2.0  
        elif potential_count == 4: bonus_points = 5.0  
        elif potential_count >= 5: bonus_points = 9.0  
        
        return token_points + bonus_points

    def _score_sell(self, action, obs, pressure, opp_confirmed_hand):
        good = action._sell
        count = action._count
        params = self.genome
        
        points = self._get_token_value(good, count, obs)
        
        # RACE LOGIC:
        # Check if opponent is competing for this exact good.
        opp_has_good = opp_confirmed_hand[good]
        is_luxury = good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]
        
        race_bonus = 0
        # Threat: They have 2+ cards and are likely to snipe the tokens.
        # Response: Sell NOW (if we have a valid set) to beat them.
        if is_luxury and opp_has_good >= 2 and count >= 3:
            race_bonus = 8.0 
        
        # Greed: They have 0 cards. We are safe to hoard for a bigger set.
        # (Only if we aren't under pressure)
        if opp_has_good == 0 and count == 4 and pressure < 5:
            race_bonus = -5.0

        bonus = 0
        if count == 3: bonus = params['bonus_3_est']
        elif count == 4: bonus = params['bonus_4_est']
        elif count >= 5: bonus = params['bonus_5_est']
        
        total = points + bonus + race_bonus
        
        if is_luxury:
            return (total * params['luxury_mult']) + pressure
        
        if good in [GoodType.LEATHER, GoodType.SPICE, GoodType.FABRIC]:
            if count >= 5: return (total * params['cheap_mult']) + pressure + 10
            if count == 4: return (total * (params['cheap_mult']*0.75)) + pressure + 5
            if count <= 2 and pressure < 10: return -50
            
        return total + pressure

    def _score_take(self, action, obs, current_hand_size, pressure, opp_confirmed_hand, opponent_locked):
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
        
        # Self Incentives
        if in_hand == 3: score += 15
        if in_hand == 4: score += 20
        if good in [GoodType.DIAMOND, GoodType.GOLD]:
             score += params['luxury_take_add']

        # DENIAL LOGIC (The "Mean" Part)
        # Calculate exactly what they score if we DON'T take this.
        opp_count = opp_confirmed_hand[good]
        threat_value = self._calculate_opponent_potential(good, opp_count, obs)
        
        # Add a fraction of their potential points to our decision weight.
        if threat_value > 0:
            score += (threat_value * 0.8)

        # LOGISTICS
        # If they are locked (7 cards), they can't 'Take'.
        # They CAN 'Trade', but it's costly. We are safer leaving this card on table.
        if opponent_locked and score < 20:
            score -= 2.0 
        
        return score - pressure

    def _score_trade(self, action, obs, current_hand_size):
        req = action.requested_goods
        off = action.offered_goods
        params = self.genome
        
        # Value In
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

        # Value Out
        value_out = 0
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL:
                    value_out += 2
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    val = tokens[-1] if tokens else 0
                    value_out += val
                    
                    # SMART PENALTY:
                    # Breaking a Luxury set (Gold) -> BAD (-15)
                    # Breaking a Junk set (Leather) to get Gold -> GOOD (-2)
                    if obs.actor_goods[g] >= 3: 
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                            value_out += params['set_break_penalty']
                        else:
                            value_out += 2.0 # Tiny penalty for junk

        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_change = count_in - count_out
        
        if current_hand_size >= 6 and space_change < 0:
            value_in += 10
            
        if completes_set:
            return 100 + (value_in - value_out)

        return value_in - value_out