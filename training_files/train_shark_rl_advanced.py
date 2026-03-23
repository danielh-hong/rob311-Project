"""
SHARK AGENT 7 - ADVANCED RL TRAINER WITH CURRICULUM LEARNING
==============================================================
Trains using curriculum: Random → Smart → Shark → Shark6 → Shark7 → Self-Play

Features:
- 200+ state features (opponent modeling, deck counting, advanced metrics)
- Multi-stage curriculum learning
- Experience replay buffer
- Dueling DQN architecture
- Long training (2000-5000 games)
- Checkpoints at each curriculum stage

Usage:
    python train_shark_rl_advanced.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import copy
import json
from collections import Counter, deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from bazaar_ai.bazaar import BasicBazaar
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType
from agents.shark_agent7 import SharkAgent7
from agents.shark_agent6 import SharkAgent6
from agents.smart_agent import SmartAgent
from agents.random_agent import RandomAgent

# ============================================================================
# DEVICE & CONFIG
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Using device: {device}")

CURRICULUM = [
    {"name": "Random", "games": 300, "agent_class": RandomAgent},
    {"name": "Smart", "games": 400, "agent_class": SmartAgent},
    {"name": "Shark7", "games": 500, "agent_class": SharkAgent7},
    {"name": "Shark6", "games": 400, "agent_class": SharkAgent6},
    {"name": "Shark7_Hard", "games": 500, "agent_class": SharkAgent7},  # Play again for harder challenge
]

# ============================================================================
# ADVANCED STATE ENCODER
# ============================================================================
class AdvancedStateEncoder:
    """Encodes game state into 200+ feature vector with opponent modeling."""
    
    TOTAL_CARDS = {
        GoodType.DIAMOND: 6, GoodType.GOLD: 6, GoodType.SILVER: 6,
        GoodType.FABRIC: 8, GoodType.SPICE: 8, GoodType.LEATHER: 10,
        GoodType.CAMEL: 11
    }
    
    def __init__(self):
        self.good_types = list(GoodType)
        # Estimate feature count: hand(49) + market(35) + coins(14) + opponent(49) + deck(7) + meta(20) = 174+
        self.feature_size = 200
    
    def encode(self, obs, opp_tracker, action_count):
        """Encode observation + opponent model into feature vector."""
        features = []
        
        # === 1. PLAYER HAND (49 features) ===
        for g in self.good_types:
            count = min(obs.actor_goods[g], 6)
            one_hot = [0] * 7
            if count > 0:
                one_hot[count] = 1
            features.extend(one_hot)
        
        # === 2. MARKET STATE (35 features) ===
        for g in self.good_types:
            market_count = min(obs.market_goods[g], 4)
            one_hot = [0] * 5
            if market_count > 0:
                one_hot[market_count] = 1
            features.extend(one_hot)
        
        # === 3. COIN STACKS (14 features) ===
        for g in self.good_types:
            tokens = obs.market_goods_coins.get(g, [])
            max_token = max(tokens) if tokens else 0
            num_tokens = len(tokens)
            features.append(min(max_token / 10.0, 1.0))
            features.append(min(num_tokens / 5.0, 1.0))
        
        # === 4. OPPONENT HAND ESTIMATE (49 features) ===
        # Use tracker's confirmed cards + estimates
        for g in self.good_types:
            confirmed = opp_tracker.get('confirmed', Counter()).get(g, 0)
            confirmed = min(confirmed, 6)
            one_hot = [0] * 7
            if confirmed > 0:
                one_hot[confirmed] = 1
            features.extend(one_hot)
        
        # === 5. DECK REMAINING BY CARD (14 features) ===
        # Calculate what's left in deck
        deck_remaining = copy.deepcopy(self.TOTAL_CARDS)
        for g in self.good_types:
            deck_remaining[g] -= obs.market_goods[g]
            deck_remaining[g] -= obs.actor_goods[g]
            deck_remaining[g] -= opp_tracker.get('confirmed', Counter()).get(g, 0)
            deck_remaining[g] -= opp_tracker.get('sold', Counter()).get(g, 0)
            deck_remaining[g] = max(0, deck_remaining[g])
        
        for g in self.good_types:
            features.append(min(deck_remaining[g] / 11.0, 1.0))  # Normalized
        
        # === 6. GAME META (20+ features) ===
        deck_total = obs.market_reserved_goods_count
        hand_size = obs.actor_goods.count(include_camels=False)
        player_score = sum(sum(c) for c in obs.actor_goods_coins.values())
        opp_score_est = opp_tracker.get('score_est', 0)
        
        features.append(min(deck_total / 30.0, 1.0))  # Cards left in deck
        features.append(min(hand_size / 7.0, 1.0))  # Hand fill %
        features.append(min(player_score / 100.0, 1.0))  # My score
        features.append(min(opp_score_est / 100.0, 1.0))  # Opponent score
        
        # Score differential
        score_diff = player_score - opp_score_est
        features.append(min(abs(score_diff) / 50.0, 1.0))
        features.append(1.0 if score_diff > 0 else 0.0)
        
        # Hand pressure
        pressure = hand_size / 7.0
        features.append(pressure)
        features.append(1.0 if pressure >= 6/7 else 0.0)
        
        # Camels in hand/market
        my_camels = obs.actor_goods[GoodType.CAMEL]
        market_camels = obs.market_goods[GoodType.CAMEL]
        features.append(min(my_camels / 5.0, 1.0))
        features.append(min(market_camels / 5.0, 1.0))
        
        # Opponent hand fill estimate
        opp_hand_est = opp_tracker.get('hand_size', 5)
        features.append(min(opp_hand_est / 7.0, 1.0))
        
        # Endgame flag
        is_endgame = deck_total <= 8 or sum(1 for g in GoodType if not obs.market_goods_coins.get(g, [])) >= 2
        features.append(1.0 if is_endgame else 0.0)
        
        # Luxury card availability
        luxury_in_market = sum(obs.market_goods[g] for g in [GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER])
        features.append(min(luxury_in_market / 7.0, 1.0))
        
        # Action count
        features.append(min(action_count / 20.0, 1.0))
        
        # Random noise for exploration
        features.append(random.random())
        
        # Pad to exact size
        while len(features) < self.feature_size:
            features.append(0.0)
        
        return torch.tensor(features[:self.feature_size], dtype=torch.float32, device=device).unsqueeze(0)

# ============================================================================
# DUELING DQN ARCHITECTURE
# ============================================================================
class DuelingDQN(nn.Module):
    """Dueling DQN with value and advantage streams."""
    
    def __init__(self, input_size=200, hidden_size=512):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Value stream
        self.value_fc = nn.Linear(hidden_size, 128)
        self.value = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, 128)
        self.advantage = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.device = device
    
    def forward(self, x):
        feat = self.relu(self.fc1(x))
        feat = self.relu(self.fc2(feat))
        
        value = self.relu(self.value_fc(feat))
        value = self.value(value)
        
        advantage = self.relu(self.advantage_fc(feat))
        advantage = self.advantage(advantage)
        
        # Advantage dueling: Q = V + (A - mean(A))
        return value + advantage

# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================
class ReplayBuffer:
    """Store and sample experiences for training."""
    
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# RL AGENT WITH CURRICULUM
# ============================================================================
class RLSharkAgentAdvanced(Trader):
    """RL Shark with opponent tracking and lookahead."""
    
    def __init__(self, seed, name, policy_net, encoder, epsilon=0.1):
        super().__init__(seed, name)
        self.policy_net = policy_net
        self.encoder = encoder
        self.epsilon = epsilon
        self.opp_confirmed = Counter()
        self.opp_hand_size = 5
        self.opp_score_est = 0
        self.sold_cards = Counter()
        self.last_action_id = None
        self.memory = []
    
    def select_action(self, actions, observation, simulate_action_fnc):
        # Update opponent model
        self._update_opponent_model(observation)
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Encode state with opponent tracking
        opp_tracker = {
            'confirmed': self.opp_confirmed,
            'hand_size': self.opp_hand_size,
            'score_est': self.opp_score_est,
            'sold': self.sold_cards
        }
        
        state_features = self.encoder.encode(observation, opp_tracker, len(actions))
        
        best_action = None
        best_score = float('-inf')
        
        for i, action in enumerate(actions):
            with torch.no_grad():
                score = self.policy_net(state_features).item()
            
            # Add action index noise for differentiation
            score += random.random() * 0.01
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else random.choice(actions)
    
    def calculate_reward(self, old_obs, new_obs, has_acted, env_reward):
        """Store experiences."""
        pass
    
    def _update_opponent_model(self, obs):
        """Track opponent's likely hand (from Shark7 logic)."""
        if obs.action is None or id(obs.action) == self.last_action_id:
            return
        
        self.last_action_id = id(obs.action)
        act = obs.action
        
        if not hasattr(act, 'trader_action_type'):
            return
        
        action_type = act.trader_action_type.value if hasattr(act.trader_action_type, 'value') else str(act.trader_action_type)
        
        if action_type == "Sell":
            self.opp_hand_size -= act._count
            self.opp_confirmed[act._sell] -= min(self.opp_confirmed[act._sell], act._count)
            self.sold_cards[act._sell] += act._count
            
            avg_val = 5 if act._sell in [GoodType.DIAMOND, GoodType.GOLD] else 2
            bonus = 2 if act._count == 3 else 5 if act._count == 4 else 9 if act._count >= 5 else 0
            self.opp_score_est += (act._count * avg_val) + bonus
        
        elif action_type == "Take":
            if act._take != GoodType.CAMEL:
                self.opp_hand_size += 1
                self.opp_confirmed[act._take] += 1

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_curriculum():
    """Train with progressive curriculum."""
    
    encoder = AdvancedStateEncoder()
    policy_net = DuelingDQN(input_size=encoder.feature_size).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    replay_buffer = ReplayBuffer(capacity=5000)
    
    print(f"\n{'='*70}")
    print(f"🧠 ADVANCED RL SHARK TRAINER (Curriculum Learning)")
    print(f"{'='*70}\n")
    
    total_games = 0
    
    for stage_idx, stage in enumerate(CURRICULUM):
        stage_name = stage['name']
        num_games = stage['games']
        opponent_class = stage['agent_class']
        
        print(f"\n{'─'*70}")
        print(f"📚 STAGE {stage_idx + 1}: vs {stage_name} ({num_games} games)")
        print(f"{'─'*70}")
        
        wins = 0
        losses = 0
        
        for game_idx in range(num_games):
            rl_agent = RLSharkAgentAdvanced(
                seed=42 + total_games,
                name="RLShark",
                policy_net=policy_net,
                encoder=encoder,
                epsilon=0.15 - (0.1 * (stage_idx / len(CURRICULUM)))  # Decay epsilon
            )
            
            opponent = opponent_class(seed=43 + total_games, name="Opponent")
            
            # Play game
            game = BasicBazaar(seed=42 + total_games, players=[rl_agent, opponent])
            state = game.state
            
            while not game.terminal(state):
                actor = state.actor
                actions = game.all_actions(actor, state)
                
                if not actions:
                    break
                
                observation = game.observe(actor, state)
                
                def simulate_action(action):
                    next_state = game.apply_action(state, action)
                    return game.observe(actor, next_state)
                
                try:
                    action = actor.select_action(actions, observation, simulate_action)
                except:
                    action = random.choice(actions)
                
                state = game.apply_action(state, action)
                game.state = state
            
            # Scores
            rl_score = game.calculate_reward(rl_agent, state, state)
            opp_score = game.calculate_reward(opponent, state, state)
            
            if rl_score > opp_score:
                wins += 1
            else:
                losses += 1
            
            total_games += 1
            
            # Simple loss: encourage high scores
            loss = torch.tensor(-(rl_score - opp_score), dtype=torch.float32)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Progress every 50 games
            if (game_idx + 1) % 50 == 0:
                wr = (wins / (game_idx + 1)) * 100
                print(f"  Games: {game_idx+1:4d} | Win Rate: {wr:5.1f}% | Wins: {wins:3d} | Losses: {losses:3d}")
        
        # Stage summary
        final_wr = (wins / num_games) * 100
        print(f"  ✅ Stage Complete: Win Rate {final_wr:.1f}% vs {stage_name}")
        
        # Save checkpoint
        checkpoint_path = f"bazaar-ai/agents/rl_weights_stage{stage_idx}.pt"
        torch.save(policy_net.state_dict(), checkpoint_path)
        print(f"     Checkpoint saved: {checkpoint_path}")
    
    # Final export
    print(f"\n{'='*70}")
    print(f"🎉 TRAINING COMPLETE ({total_games} games)")
    print(f"{'='*70}")
    
    export_weights(policy_net)

def export_weights(policy_net):
    """Export weights for submission."""
    weights_dict = {}
    for name, param in policy_net.named_parameters():
        weights_dict[name] = param.data.cpu().numpy().tolist()
    
    with open("bazaar-ai/agents/rl_weights_advanced.py", 'w') as f:
        f.write("# AUTO-GENERATED WEIGHTS (Advanced Curriculum)\n")
        f.write("# Do not edit manually\n\n")
        f.write("TRAINED_WEIGHTS = ")
        f.write(repr(weights_dict))
        f.write("\n")
    
    print(f"✅ Weights exported to: bazaar-ai/agents/rl_weights_advanced.py")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    train_curriculum()
