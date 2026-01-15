import random
import uuid
import copy
from collections import Counter
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType

# --- CONSTANTS ---
# Exact card distribution from Jaipur Rulebook.
TOTAL_CARDS = {
    GoodType.DIAMOND: 6,
    GoodType.GOLD: 6,
    GoodType.SILVER: 6,
    GoodType.FABRIC: 8,
    GoodType.SPICE: 8,
    GoodType.LEATHER: 10,
    GoodType.CAMEL: 11
}

class GlobalStateTracker:
    """
    OMNISCIENT STATE ENGINE
    Tracks hidden information by deducing it from the action log history.
    """
    def __init__(self):
        self.opp_confirmed = Counter() 
        self.opp_hand_size = 5         
        self.opp_camels = 0            
        self.opp_score_est = 0         
        self.sold_cards = Counter()    
        self.last_action_id = None     

    def update(self, obs):
        if obs.action is None: return
        if id(obs.action) == self.last_action_id: return
        self.last_action_id = id(obs.action)
        
        act = obs.action
        
        # --- CASE A: OPPONENT SOLD CARDS ---
        if act.trader_action_type.value == "Sell":
            good = act._sell
            count = act._count
            
            # Update Hand Size & Knowledge
            self.opp_hand_size -= count
            known = self.opp_confirmed[good]
            self.opp_confirmed[good] -= min(known, count)
            
            # Update Dead Pile (Gone forever)
            self.sold_cards[good] += count
            
            # Update Score Estimate
            avg_val = 5 if good in [GoodType.DIAMOND, GoodType.GOLD] else 2
            bonus = 0
            if count == 3: bonus = 2
            elif count == 4: bonus = 5
            elif count >= 5: bonus = 9
            self.opp_score_est += (count * avg_val) + bonus

        # --- CASE B: OPPONENT TOOK CARDS ---
        elif act.trader_action_type.value == "Take":
            good = act._take
            
            if good == GoodType.CAMEL:
                # We trust the action object or assume at least 1
                count = getattr(act, '_count', 1)
                self.opp_camels += count
            else:
                self.opp_hand_size += 1
                self.opp_confirmed[good] += 1

        # --- CASE C: OPPONENT TRADED ---
        elif act.trader_action_type.value == "Trade":
            for g in GoodType:
                if act.requested_goods[g] > 0: 
                    self.opp_confirmed[g] += act.requested_goods[g]
            
            for g in GoodType:
                qty = act.offered_goods[g]
                if qty > 0:
                    if g == GoodType.CAMEL:
                        self.opp_camels = max(0, self.opp_camels - qty)
                    else:
                        known = self.opp_confirmed[g]
                        self.opp_confirmed[g] -= min(known, qty)
    
    def get_deck_remaining(self, obs):
        """Calculates cards remaining in the draw pile."""
        remaining = copy.deepcopy(TOTAL_CARDS)
        
        for g in GoodType: remaining[g] -= obs.market_goods[g]
        for g in GoodType: remaining[g] -= obs.actor_goods[g]
        for g in remaining: remaining[g] -= self.opp_confirmed[g]
        for g in remaining: remaining[g] -= self.sold_cards[g]
        
        for g in remaining: remaining[g] = max(0, remaining[g])
        return remaining


class SharkAgent7(Trader):
    """
    SHARK AGENT 7 (Omega Variant)
    -----------------------------
    Configuration: 93.2% Win Rate Genome against Mixed League.
    Strategy: Hoarder / Mercy Killer / Deck Counter.
    """
    def __init__(self, seed, name):
        super().__init__(seed, name)
        
        self.tracker = GlobalStateTracker()
        
        if not hasattr(self, 'uuid'): self.uuid = uuid.uuid4()

        # --- THE 93.2% RECORD GENOME ---
        self.genome = {
            'bonus_3_est': 0.44093175922854905, 
            'bonus_4_est': 17.168219969548513, 
            'bonus_5_est': 26.993858784408033, 
            'luxury_mult': 0.18531452549948751, 
            'cheap_mult': 0.5234153816218508, 
            'pressure_weight': 0.1143520517829709, 
            'camel_min_util': 4.238755563598485, 
            'camel_take_val': 0.008100828264704409, 
            'trade_set_bonus': 4.198623573899503, 
            'luxury_take_add': 0.15866538889263251, 
            'set_break_penalty': 37.26189099078606, 
            'denial_weight': 0.06079607741591933, 
            'impossible_sell_bonus': 17.4808323336054, 
            'scarcity_bonus': 8.710491419776499, 
            'waste_penalty': 2.904686199332339, 
            'endgame_rush_bonus': 20.47920950880446, 
            'endgame_camel_value': 14.644475736790667, 
            'mercy_kill_bonus': 11.161215221685692, 
            'fishing_bonus': 1.9886034346621024
        }

    def select_action(self, actions, observation, simulate_action_fnc):
        # 1. Update Global Knowledge
        self.tracker.update(observation)
        
        # 2. Derive Context
        opp_confirmed = self.tracker.opp_confirmed
        deck_remaining = self.tracker.get_deck_remaining(observation)
        opp_locked = (self.tracker.opp_hand_size >= 7)
        
        # 3. Detect Game End Conditions
        cards_in_deck = observation.market_reserved_goods_count
        empty_piles = 0
        for g in GoodType:
            if not observation.market_goods_coins.get(g, []):
                empty_piles += 1
        
        is_endgame = (cards_in_deck <= 8) or (empty_piles >= 2)

        # 4. Score Check
        my_score = self._calculate_my_current_score(observation)
        opp_score = self.tracker.opp_score_est
        am_i_winning = (my_score > opp_score + 10)

        # 5. Hand Pressure
        hand = observation.actor_goods
        hand_size = hand.count(include_camels=False)
        limit = observation.max_player_goods_count
        
        pressure = 0
        if hand_size >= limit: pressure = 20 * self.genome['pressure_weight']
        elif hand_size >= limit - 1: pressure = 5 * self.genome['pressure_weight']

        # 6. Evaluate All Actions
        best_action = None
        best_score = float('-inf')

        for action in actions:
            score = 0
            
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure, opp_confirmed, deck_remaining, is_endgame, am_i_winning)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure, opp_confirmed, opp_locked, is_endgame, deck_remaining)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size, deck_remaining)
            
            # Deterministic Jitter
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action      
        return best_action

    def _calculate_my_current_score(self, obs):
        total = 0
        for coins in obs.actor_goods_coins.values():
            total += sum(coins)
        return total

    # --- SCORING ENGINE ---

    def _get_exact_value(self, good, count, obs):
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        real_sales = min(count, len(tokens))
        return sum(tokens[-real_sales:])

    def _calculate_opponent_potential(self, good, opp_confirmed_count, deck_remaining, obs):
        max_possible = opp_confirmed_count + deck_remaining[good] + obs.market_goods[good]
        potential_count = opp_confirmed_count + 1
        
        if potential_count > max_possible: return 0 
        if potential_count < 3: return 0 
        
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens: return 0
        take_n = min(potential_count, len(tokens))
        return sum(tokens[-take_n:]) + (2.0 if potential_count == 3 else 5.0)

    def _score_sell(self, action, obs, pressure, opp_confirmed, deck, is_endgame, am_i_winning):
        good = action._sell
        count = action._count
        params = self.genome
        
        points = self._get_exact_value(good, count, obs)
        tokens_available = len(obs.market_goods_coins.get(good, []))
        
        waste_penalty = 0
        if count > tokens_available:
            waste_penalty = params['waste_penalty'] * (count - tokens_available)

        impossible = (deck[good] == 0 and opp_confirmed[good] == 0 and obs.market_goods[good] == 0)
        
        # Mercy Kill
        mercy_kill_bonus = 0
        if am_i_winning and tokens_available <= count:
            empty_piles = sum(1 for g in GoodType if not obs.market_goods_coins.get(g, []))
            if empty_piles >= 2:
                mercy_kill_bonus = params['mercy_kill_bonus']

        bonus = 0
        if count == 3: bonus = params['bonus_3_est']
        elif count == 4: bonus = params['bonus_4_est']
        elif count >= 5: bonus = params['bonus_5_est']
        
        opp_has_good = opp_confirmed[good]
        is_luxury = good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]
        
        race_bonus = 0
        if is_luxury and opp_has_good >= 2 and count >= 3: 
            race_bonus = 8.0 
        
        if impossible: race_bonus += params['impossible_sell_bonus']
        if is_endgame and count >= 3: race_bonus += params['endgame_rush_bonus']

        total = points + bonus + race_bonus + mercy_kill_bonus - waste_penalty
        
        if is_luxury: return (total * params['luxury_mult']) + pressure
        if good in [GoodType.LEATHER, GoodType.SPICE, GoodType.FABRIC]:
            if count >= 5: return (total * params['cheap_mult']) + pressure + 10
            if count == 4: return (total * (params['cheap_mult']*0.75)) + pressure + 5
            if is_endgame and count >= 3: return total + pressure
            if count <= 2 and pressure < 10: return -50
        return total + pressure

    def _score_take(self, action, obs, current_hand_size, pressure, opp_confirmed, opp_locked, is_endgame, deck):
        good = action._take
        params = self.genome
        
        if good == GoodType.CAMEL:
            # Fishing Logic
            wanted = 0
            for t in GoodType:
                if t != GoodType.CAMEL and obs.actor_goods[t] >= 2:
                    wanted += deck[t]
            deck_total = sum(deck.values())
            fishing_score = 0
            if deck_total > 0:
                fishing_score = (wanted / deck_total) * params['fishing_bonus']

            if is_endgame: return params['endgame_camel_value']
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2: return params['camel_min_util']
            return params['camel_take_val'] + fishing_score

        tokens = obs.market_goods_coins.get(good, [])
        score = tokens[-1] if tokens else 1
        
        in_hand = obs.actor_goods[good]
        
        scarcity_bonus = 0
        if deck[good] == 0: scarcity_bonus = params['scarcity_bonus']
        elif deck[good] == 1: scarcity_bonus = params['scarcity_bonus'] / 2

        if in_hand == 3: score += 15
        if in_hand == 4: score += 20
        if good in [GoodType.DIAMOND, GoodType.GOLD]: score += params['luxury_take_add']

        opp_count = opp_confirmed[good]
        threat_value = self._calculate_opponent_potential(good, opp_count, deck, obs)
        if threat_value > 0: score += (threat_value * params['denial_weight'])

        if opp_locked and score < 20: score -= 2.0 
        return score - pressure

    def _score_trade(self, action, obs, current_hand_size, deck):
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
                elif obs.actor_goods[g] + req[g] == 4 and deck[g] == 0:
                     value_in += params['trade_set_bonus'] 
                else:
                    value_in += val

        value_out = 0
        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL: value_out += 2
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    val = tokens[-1] if tokens else 0
                    value_out += val
                    if obs.actor_goods[g] >= 3: 
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                            value_out += params['set_break_penalty']
                        else: value_out += 2.0 

        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_change = count_in - count_out
        
        if current_hand_size >= 6 and space_change < 0: value_in += 10
        if completes_set: return 100 + (value_in - value_out)
        return value_in - value_out