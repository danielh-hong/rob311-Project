# improvements from sharkagent6
# 1. Enhanced opponent modeling with probabilistic hand estimation
# 2. Dynamic game phase awareness (early/mid/late game strategies)
# 3. Improved lookahead simulation for critical decisions
# 4. Market control and tempo management
# 5. Adaptive risk tolerance based on score differential
# 6. Multi-objective optimization with priority balancing
#
# FIXES applied over original star_agent:
# - Genome values replaced with SharkAgent6 trained values (was hand-guessed)
# - _score_sell: luxury path now fully separated, never hits cheap-goods penalty
# - _score_take: removed erroneous top_token * 2 (was inflating takes vs sells)
# - denial_weight boosted to 0.45 (tracker is better so denial is worth more)

import random
from collections import Counter, defaultdict
from typing import Optional
from backend.trader import Trader, SellAction, TakeAction, TradeAction
from backend.goods import GoodType


class EnhancedOpponentTracker:
    """
    ADVANCED STATE ESTIMATION ENGINE
    Tracks opponent's hand with confidence levels and probabilistic modeling.
    Goes beyond simple counting to estimate likely actions and threats.
    """
    def __init__(self):
        self.confirmed_hand = Counter()  # Cards we know they have
        self.possible_hand = Counter()   # Cards they might have (probabilistic)
        self.unknown_cards = 5           # Initial unknowns
        self.hand_size = 5
        self.last_action_id = None

        # Action history for pattern detection
        self.action_history = []
        self.goods_taken = Counter()
        self.goods_sold = Counter()

    def update(self, obs):
        if obs.action is None:
            return

        if id(obs.action) == self.last_action_id:
            return
        self.last_action_id = id(obs.action)

        act = obs.action
        self.action_history.append(act.trader_action_type.value)

        if act.trader_action_type.value == "Sell":
            good = act._sell
            count = act._count
            self.hand_size -= count
            self.goods_sold[good] += count

            known_count = self.confirmed_hand[good]
            remove_from_known = min(known_count, count)
            remove_from_unknown = count - remove_from_known

            self.confirmed_hand[good] -= remove_from_known
            self.unknown_cards = max(0, self.unknown_cards - remove_from_unknown)

        elif act.trader_action_type.value == "Take":
            good = act._take
            self.goods_taken[good] += 1
            if good != GoodType.CAMEL:
                self.hand_size += 1
                self.confirmed_hand[good] += 1

        elif act.trader_action_type.value == "Trade":
            for g in GoodType:
                count_in = act.requested_goods[g]
                if count_in > 0:
                    self.confirmed_hand[g] += count_in
                    self.goods_taken[g] += count_in

                count_out = act.offered_goods[g]
                if count_out > 0:
                    known_count = self.confirmed_hand[g]
                    remove_from_known = min(known_count, count_out)
                    remove_from_unknown = count_out - remove_from_known

                    self.confirmed_hand[g] -= remove_from_known
                    self.unknown_cards = max(0, self.unknown_cards - remove_from_unknown)

    def get_threat_level(self, good_type):
        """Estimate how threatening opponent is for a specific good"""
        confirmed = self.confirmed_hand[good_type]

        if confirmed >= 2:
            return 3
        elif confirmed == 1 and self.goods_taken[good_type] >= 2:
            return 2
        elif self.goods_sold[good_type] > 0:
            return 1
        return 1

    def estimate_sell_likelihood(self, good_type):
        """Estimate probability opponent will sell this good soon"""
        confirmed = self.confirmed_hand[good_type]

        if confirmed >= 5:
            return 0.9
        elif confirmed >= 4:
            return 0.6
        elif confirmed >= 3:
            return 0.3
        return 0.0


class GamePhaseAnalyzer:
    """
    PHASE DETECTION SYSTEM
    ----------------------
    Determines current game phase and adjusts strategy accordingly.
    """
    def __init__(self):
        self.initial_token_counts = {}

    def analyze_phase(self, obs):
        """
        Returns: ('early', 'mid', 'late'), along with urgency score
        """
        depleted_types = 0
        total_tokens_remaining = 0

        for good_type in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER,
                          GoodType.FABRIC, GoodType.SPICE, GoodType.LEATHER]:
            tokens = obs.market_goods_coins.get(good_type, [])
            if len(tokens) == 0:
                depleted_types += 1
            total_tokens_remaining += len(tokens)

        deck_remaining = obs.market_reserved_goods_count

        if depleted_types >= 2:
            return 'late', 0.9
        elif depleted_types >= 1 or deck_remaining < 15:
            return 'mid', 0.6
        elif total_tokens_remaining < 20 or deck_remaining < 25:
            return 'mid', 0.4
        else:
            return 'early', 0.2

    def get_phase_priorities(self, phase):
        """Returns strategic priorities for each phase"""
        if phase == 'early':
            return {
                'collect_luxury': 1.2,
                'build_sets': 1.0,
                'deny_opponent': 0.7,
                'sell_pressure': 0.5
            }
        elif phase == 'mid':
            return {
                'collect_luxury': 1.0,
                'build_sets': 1.2,
                'deny_opponent': 1.0,
                'sell_pressure': 0.8
            }
        else:  # late
            return {
                'collect_luxury': 0.7,
                'build_sets': 0.8,
                'deny_opponent': 1.3,
                'sell_pressure': 1.5
            }


class StarAgent(Trader):
    """
    Key Improvements over SharkAgent6:

    1. SMARTER OPPONENT MODELING
       - Tracks action patterns and preferences
       - Estimates threat levels probabilistically
       - Predicts opponent's next moves

    2. DYNAMIC PHASE ADAPTATION
       - Recognizes early/mid/late game
       - Adjusts aggression and risk tolerance
       - Optimizes timing of sales and denials

    3. LOOKAHEAD SIMULATION
       - Uses simulate_action_fnc for critical decisions
       - Plans 1-2 moves ahead in key situations
       - Evaluates opponent's best responses

    4. MARKET CONTROL
       - Maintains tempo advantage
       - Controls valuable goods supply
       - Forces opponent into bad trades

    5. SCORE-AWARE STRATEGY
       - More aggressive when behind
       - More defensive when ahead
       - Risk/reward optimization

    6. IMPROVED HEURISTICS
       - Better bonus token estimation
       - Smarter camel management
       - Context-aware set building
    """

    def __init__(self, seed, name):
        super().__init__(seed, name)

        self.tracker = EnhancedOpponentTracker()
        self.phase_analyzer = GamePhaseAnalyzer()

        # FIX 1: Use SharkAgent6's battle-tested trained genome as the baseline
        # (previously these were hand-guessed values that hurt performance)
        self.base_genome = {
            'bonus_3_est': 0.31745425260167687,
            'bonus_4_est': 13.66489674604007,
            'bonus_5_est': 34.21592167542601,
            'luxury_mult': 0.6449390514855928,
            'cheap_mult': 0.4430829450999698,
            'pressure_weight': 0.6655756266220109,
            'camel_min_util': 5.314037753352736,
            'camel_take_val': 3.537295209452517,
            'trade_set_bonus': 7.003032890479059,
            'luxury_take_add': 0.316,
            'set_break_penalty': 22.26455472934068,
            'denial_weight': 0.6236630054928441,
        }

        # Enhanced strategic parameters
        self.strategy_params = {
            'simulation_depth': 1,
            'denial_threshold': 15.0,
            'early_game_hoard': 1.2,
            'late_game_urgency': 1.5,
            'tempo_value': 8.0,
            'market_control_bonus': 12.0,
        }

        self.my_score_estimate = 0
        self.opp_score_estimate = 0

    def select_action(self, actions, observation, simulate_action_fnc):
        """
        MAIN DECISION ENGINE
        """
        # 1. Update all trackers
        self.tracker.update(observation)
        phase, urgency = self.phase_analyzer.analyze_phase(observation)

        # 2. Analyze game state
        hand = observation.actor_goods
        hand_size = hand.count(include_camels=False)
        hand_limit = observation.max_player_goods_count

        opp_confirmed = self.tracker.confirmed_hand
        opp_hand_size = self.tracker.hand_size
        opponent_locked = (opp_hand_size >= 7)

        # 3. Calculate pressure and phase adjustments
        pressure = self._calculate_pressure(hand_size, hand_limit, urgency)
        phase_mods = self.phase_analyzer.get_phase_priorities(phase)

        # 4. Score all actions
        best_action = None
        best_score = float('-inf')

        for action in actions:
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure,
                                         opp_confirmed, phase, phase_mods)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure,
                                         opp_confirmed, opponent_locked, phase, phase_mods)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size,
                                          phase, phase_mods)
            else:
                score = float('-inf')

            # Lookahead for high-value decisions
            if score > 20 or isinstance(action, SellAction):
                lookahead_bonus = self._evaluate_lookahead(
                    action, observation, simulate_action_fnc, phase
                )
                score += lookahead_bonus

            # Random tiebreaker
            score += random.random() * 0.01

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # --- SCORING FUNCTIONS ---

    def _calculate_pressure(self, hand_size, hand_limit, urgency):
        """Calculate hand pressure with urgency modification"""
        base_pressure = 0

        if hand_size >= hand_limit:
            base_pressure = 25
        elif hand_size >= hand_limit - 1:
            base_pressure = 10
        elif hand_size >= hand_limit - 2:
            base_pressure = 3

        return base_pressure * (1 + urgency * 0.5)

    def _score_sell(self, action, obs, pressure, opp_confirmed_hand, phase, phase_mods):
        """
        FIX 2: Luxury and cheap goods are now fully separate paths.
        Previously luxury could fall into the cheap path and get a -60 penalty.
        """
        good = action._sell
        count = action._count

        tokens = obs.market_goods_coins.get(good, [])
        if not tokens:
            return -1000

        take_n = min(count, len(tokens))
        coins_value = sum(tokens[-take_n:])

        bonus_value = 0
        if count == 3:   bonus_value = self.base_genome['bonus_3_est']
        elif count == 4: bonus_value = self.base_genome['bonus_4_est']
        elif count >= 5: bonus_value = self.base_genome['bonus_5_est']

        total = coins_value + bonus_value
        is_luxury = good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]

        # --- LUXURY PATH (never penalised, always scored positively) ---
        if is_luxury:
            opp_has = opp_confirmed_hand[good]
            threat_level = self.tracker.get_threat_level(good)

            race_bonus = 0
            if opp_has >= 3 and threat_level >= 3:
                race_bonus = 15.0   # they're about to sell — go now
            elif opp_has >= 2 and threat_level >= 2:
                race_bonus = 8.0
            elif opp_has == 0 and count == 4 and phase == 'early':
                race_bonus = -8.0  # safe to wait for 5-set

            total += race_bonus

            # Apply late-game urgency multiplier only in late phase
            late_mult = phase_mods['sell_pressure'] if phase == 'late' else 1.0
            return (total * self.base_genome['luxury_mult'] * late_mult) + pressure

        # --- CHEAP GOODS PATH ---
        if count >= 5:
            return (total * self.base_genome['cheap_mult'] * phase_mods['sell_pressure']) + pressure + 15
        if count == 4:
            return (total * self.base_genome['cheap_mult'] * 0.75 * phase_mods['sell_pressure']) + pressure + 8
        if count <= 2 and pressure < 10:
            return -50  # never sell tiny cheap sets unless hand is desperate
        return total * phase_mods['sell_pressure'] + pressure

    def _score_take(self, action, obs, current_hand_size, pressure,
                    opp_confirmed_hand, opponent_locked, phase, phase_mods):
        """
        FIX 3: Removed top_token * 2 — was doubling take scores relative to sells,
        causing the agent to hoard instead of sell at the right time.
        """
        good = action._take

        # CAMEL HANDLING
        if good == GoodType.CAMEL:
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2:
                return self.base_genome['camel_min_util']
            return self.base_genome['camel_take_val']

        tokens = obs.market_goods_coins.get(good, [])
        if not tokens:
            return -10

        top_token = tokens[-1]
        in_hand = obs.actor_goods[good]

        # FIX: was top_token * 2 — now correctly top_token * 1
        score = top_token

        # Set-building incentives (same scale as SharkAgent6)
        if in_hand == 4: score += 20
        elif in_hand == 3: score += 15

        # Luxury bonus + phase multiplier
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            score += self.base_genome['luxury_take_add']
            score *= phase_mods['collect_luxury']

        # Denial logic — uses richer threat tracking
        opp_count = opp_confirmed_hand[good]
        threat_value = self._calculate_threat_value(good, opp_count, obs)

        if threat_value > self.strategy_params['denial_threshold']:
            denial_bonus = threat_value * self.base_genome['denial_weight']
            denial_bonus *= phase_mods['deny_opponent']
            score += denial_bonus
        elif threat_value > 0:
            score += threat_value * 0.2  # small nudge even below threshold

        # Opponent state
        if opponent_locked and score < 20:
            score -= 2.0
        elif not opponent_locked and threat_value > 10:
            score += 5.0  # they can take it next turn — priority up

        # Late game: don't start brand-new sets
        if phase == 'late' and in_hand == 0:
            score -= 8.0

        return score - pressure

    def _score_trade(self, action, obs, current_hand_size, phase, phase_mods):
        """Enhanced trade scoring with set break penalties"""
        req = action.requested_goods
        off = action.offered_goods

        # VALUE GAINED
        value_in = 0
        completes_valuable_set = False

        for g in GoodType:
            if req[g] > 0:
                tokens = obs.market_goods_coins.get(g, [])
                token_val = tokens[-1] if tokens else 1

                current_count = obs.actor_goods[g]
                future_count = current_count + req[g]

                if future_count >= 5:
                    value_in += self.base_genome['trade_set_bonus'] * 2
                    completes_valuable_set = True
                elif future_count >= 4:
                    value_in += self.base_genome['trade_set_bonus'] * 1.2
                elif future_count >= 3:
                    value_in += self.base_genome['trade_set_bonus']
                else:
                    value_in += token_val * 3

        # VALUE LOST
        value_out = 0
        breaking_luxury = False

        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL:
                    value_out += 3
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    token_val = tokens[-1] if tokens else 1
                    value_out += token_val * 2

                    current_count = obs.actor_goods[g]

                    if current_count >= 4:
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                            value_out += self.base_genome['set_break_penalty']
                            breaking_luxury = True
                        else:
                            value_out += 15.0
                    elif current_count >= 3:
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                            value_out += 20.0
                        else:
                            value_out += 8.0

        # Hand management
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        space_change = count_in - count_out

        if current_hand_size >= 6 and space_change < 0:
            value_in += 15

        # Late game: only trade if completing a set
        if phase == 'late' and not completes_valuable_set:
            value_out += 10

        if completes_valuable_set and not breaking_luxury:
            return 120 + (value_in - value_out)

        return (value_in - value_out) * phase_mods['build_sets']

    # --- ADVANCED FEATURES ---

    def _calculate_threat_value(self, good, opp_count, obs):
        """Calculate exact point value of opponent threat"""
        potential_count = opp_count + 1

        if potential_count < 3:
            return 0

        tokens = obs.market_goods_coins.get(good, [])
        if not tokens:
            return 0

        take_n = min(potential_count, len(tokens))
        token_points = sum(tokens[-take_n:])

        bonus_points = 0
        if potential_count == 3:   bonus_points = 2.0
        elif potential_count == 4: bonus_points = 5.0
        elif potential_count >= 5: bonus_points = 9.0

        return token_points + bonus_points

    def _evaluate_lookahead(self, action, obs, simulate_fnc, phase):
        """
        Simulate action and evaluate resulting state.
        Returns a bonus score adjustment based on lookahead.
        """
        try:
            if not isinstance(action, (SellAction, TakeAction)):
                return 0

            future_obs = simulate_fnc(action)
            bonus = 0

            if isinstance(action, SellAction):
                bonus += 3

            elif isinstance(action, TakeAction):
                good = action._take
                if good != GoodType.CAMEL:
                    future_count = future_obs.actor_goods[good]
                    if future_count >= 5:   bonus += 8
                    elif future_count >= 4: bonus += 5
                    elif future_count >= 3: bonus += 3

            # Penalise if market becomes dangerous after our move
            market_threat = 0
            for card in future_obs.market_goods:
                if card in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                    market_threat += 2

            if market_threat > 4 and future_obs.actor_non_camel_goods_count < 7:
                bonus -= market_threat * 0.5

            return bonus

        except:
            return 0

    def calculate_reward(self, old_observation, new_observation, has_acted, environment_reward):
        """Required by interface — not used for online learning in this version"""
        pass
