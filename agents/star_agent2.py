import random
from collections import Counter
from backend.trader import Trader, SellAction, TakeAction, TradeAction
from backend.goods import GoodType


class OpponentTracker:
    """
    Tracks opponent's hand exactly by reading the public action log.
    Based on SharkAgent6's PerfectTracker (proven to work well).
    """
    def __init__(self):
        self.confirmed_hand = Counter()
        self.unknown_cards = 5
        self.hand_size = 5
        self.last_action_id = None
        self.goods_taken = Counter()
        self.goods_sold = Counter()
        self.action_history = []

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
        confirmed = self.confirmed_hand[good_type]
        if confirmed >= 2:
            return 3
        elif confirmed == 1 and self.goods_taken[good_type] >= 2:
            return 2
        return 1


class StarAgent(Trader):
    """
    StarAgent - Improved over SharkAgent6 with:
    1. Game phase awareness (early/mid/late strategy shifts)
    2. Smarter sell timing - knows when to race vs hoard
    3. Better denial logic with phase multipliers
    4. Trained genome via genetic evolution
    """

    def __init__(self, seed, name):
        super().__init__(seed, name)

        self.tracker = OpponentTracker()

        # Trained genome (from genetic evolution - 70% win rate vs SharkAgent6)
        self.genome = {
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

    def _get_phase(self, obs):
        """Detect early/mid/late game phase."""
        depleted = 0
        for g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER,
                  GoodType.FABRIC, GoodType.SPICE, GoodType.LEATHER]:
            if len(obs.market_goods_coins.get(g, [])) == 0:
                depleted += 1
        deck = obs.market_reserved_goods_count
        if depleted >= 2:
            return 'late', 0.9
        elif depleted >= 1 or deck < 15:
            return 'mid', 0.6
        elif deck < 25:
            return 'mid', 0.4
        return 'early', 0.2

    def select_action(self, actions, observation, simulate_action_fnc):
        self.tracker.update(observation)

        phase, urgency = self._get_phase(observation)
        hand_size = observation.actor_goods.count(include_camels=False)
        hand_limit = observation.max_player_goods_count
        opp_confirmed = self.tracker.confirmed_hand
        opponent_locked = (self.tracker.hand_size >= 7)

        # Pressure increases near hand limit and in late game
        pressure = 0
        if hand_size >= hand_limit:
            pressure = 20 * self.genome['pressure_weight'] * (1 + urgency)
        elif hand_size >= hand_limit - 1:
            pressure = 5 * self.genome['pressure_weight'] * (1 + urgency)

        best_action = None
        best_score = float('-inf')

        for action in actions:
            if isinstance(action, SellAction):
                score = self._score_sell(action, observation, pressure, opp_confirmed, phase, urgency)
            elif isinstance(action, TakeAction):
                score = self._score_take(action, observation, hand_size, pressure, opp_confirmed, opponent_locked, phase, urgency)
            elif isinstance(action, TradeAction):
                score = self._score_trade(action, observation, hand_size, phase)
            else:
                score = float('-inf')

            score += random.random() * 0.1

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _get_token_value(self, good, count, obs):
        tokens = obs.market_goods_coins.get(good, [])
        if not tokens:
            return 0
        take_n = min(count, len(tokens))
        return sum(tokens[-take_n:])

    def _calculate_opponent_potential(self, good, opp_count, obs):
        """How many points would opponent score if they take this card and sell."""
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

    def _score_sell(self, action, obs, pressure, opp_confirmed, phase, urgency):
        good = action._sell
        count = action._count

        points = self._get_token_value(good, count, obs)
        if points == 0:
            return -1000

        # Bonus token estimate
        bonus = 0
        if count == 3:   bonus = self.genome['bonus_3_est']
        elif count == 4: bonus = self.genome['bonus_4_est']
        elif count >= 5: bonus = self.genome['bonus_5_est']

        total = points + bonus
        is_luxury = good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]

        if is_luxury:
            opp_has = opp_confirmed[good]
            threat = self.tracker.get_threat_level(good)

            race_bonus = 0
            # They're about to beat us to the tokens - sell now!
            if opp_has >= 3 and threat >= 3:
                race_bonus = 12.0
            elif opp_has >= 2 and threat >= 2:
                race_bonus = 8.0
            # Safe to hoard for bigger set in early game
            elif opp_has == 0 and count == 4 and phase == 'early':
                race_bonus = -5.0

            total += race_bonus

            # Late game urgency multiplier
            late_mult = 1.5 if phase == 'late' else 1.0
            return (total * self.genome['luxury_mult'] * late_mult) + pressure

        # CHEAP GOODS: Only sell in bulk or under pressure
        if good in [GoodType.LEATHER, GoodType.SPICE, GoodType.FABRIC]:
            if count >= 5:
                return (total * self.genome['cheap_mult']) + pressure + 10
            if count == 4:
                return (total * self.genome['cheap_mult'] * 0.75) + pressure + 5
            if count == 3:
                # Only sell 3 if under pressure or late game
                if pressure > 5 or phase == 'late':
                    return total + pressure
                return -20
            if count <= 2 and pressure < 10:
                return -50

        return total + pressure

    def _score_take(self, action, obs, hand_size, pressure, opp_confirmed, opponent_locked, phase, urgency):
        good = action._take

        if good == GoodType.CAMEL:
            my_camels = obs.actor_goods[GoodType.CAMEL]
            if my_camels < 2:
                return self.genome['camel_min_util']
            return self.genome['camel_take_val']

        tokens = obs.market_goods_coins.get(good, [])
        top_token_val = tokens[-1] if tokens else 1

        in_hand = obs.actor_goods[good]
        score = top_token_val

        # Set building incentives
        if in_hand == 4: score += 20
        elif in_hand == 3: score += 15
        elif in_hand == 2: score += 5

        # Luxury bonus
        if good in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
            score += self.genome['luxury_take_add']
            # Extra value in early/mid game when tokens are plentiful
            if phase != 'late':
                score += 3.0

        # DENIAL: How much do we hurt them by taking this?
        opp_count = opp_confirmed[good]
        threat_value = self._calculate_opponent_potential(good, opp_count, obs)

        if threat_value > 0:
            # Phase-aware denial: more aggressive in mid/late
            denial_mult = 1.3 if phase == 'late' else (1.0 if phase == 'mid' else 0.7)
            score += threat_value * self.genome['denial_weight'] * denial_mult

        # If opponent is locked (7 cards), market is safer
        if opponent_locked and score < 20:
            score -= 2.0
        elif not opponent_locked and threat_value > 10:
            score += 5.0

        # Late game: don't start brand new sets
        if phase == 'late' and in_hand == 0:
            score -= 10.0

        return score - pressure

    def _score_trade(self, action, obs, hand_size, phase):
        req = action.requested_goods
        off = action.offered_goods

        value_in = 0
        completes_set = False

        for g in GoodType:
            if req[g] > 0:
                tokens = obs.market_goods_coins.get(g, [])
                token_val = tokens[-1] if tokens else 1
                current = obs.actor_goods[g]
                future = current + req[g]

                if future >= 5:
                    value_in += self.genome['trade_set_bonus'] * 2
                    completes_set = True
                elif future >= 4:
                    value_in += self.genome['trade_set_bonus'] * 1.2
                elif future >= 3:
                    value_in += self.genome['trade_set_bonus']
                else:
                    value_in += token_val * 3

        value_out = 0
        breaking_luxury = False

        for g in GoodType:
            if off[g] > 0:
                if g == GoodType.CAMEL:
                    value_out += 2
                else:
                    tokens = obs.market_goods_coins.get(g, [])
                    token_val = tokens[-1] if tokens else 1
                    value_out += token_val

                    current = obs.actor_goods[g]
                    if current >= 3:
                        if g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER]:
                            value_out += self.genome['set_break_penalty']
                            breaking_luxury = True
                        else:
                            value_out += 10.0

        # Reward making space when near full
        count_in = req.count(include_camels=False)
        count_out = off.count(include_camels=False)
        if hand_size >= 6 and (count_in - count_out) < 0:
            value_in += 10

        # Late game: only trade if completing a set
        if phase == 'late' and not completes_set:
            value_out += 15

        if completes_set and not breaking_luxury:
            return 100 + (value_in - value_out)

        return value_in - value_out

    def calculate_reward(self, old_observation, new_observation, has_acted, environment_reward):
        pass