"""
SHARK AGENT 7 - REINFORCEMENT LEARNING TRAINER
================================================
Trains a neural network agent on GPU using PyTorch.
Embeds final weights into a submission-ready .py file.

Usage:
    python train_shark_rl.py

Requirements:
    pip install torch numpy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import copy
import json
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from bazaar_ai.bazaar import BasicBazaar
from bazaar_ai.trader import Trader, SellAction, TakeAction, TradeAction
from bazaar_ai.goods import GoodType
from agents.shark_agent7 import SharkAgent7
from agents.smart_agent import SmartAgent

# ============================================================================
# DEVICE SETUP
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# STATE ENCODER: Convert observation to fixed-size vector
# ============================================================================
class StateEncoder:
    """Encodes game state into a fixed-size feature vector."""
    
    TOTAL_CARDS = {
        GoodType.DIAMOND: 6, GoodType.GOLD: 6, GoodType.SILVER: 6,
        GoodType.FABRIC: 8, GoodType.SPICE: 8, GoodType.LEATHER: 10,
        GoodType.CAMEL: 11
    }
    
    def __init__(self):
        self.good_types = list(GoodType)
        # Feature size: hand(7*7) + market(7*5) + coins(estimates) + meta(5)
        self.feature_size = 7*7 + 7*5 + 7*2 + 5
    
    def encode(self, obs, action_count):
        """Convert observation to feature vector."""
        features = []
        
        # 1. Player Hand (one-hot: 7 goods x 7 cards each = 49)
        for g in self.good_types:
            count = obs.actor_goods[g]
            # Clamp to 0-6
            count = min(count, 6)
            one_hot = [0] * 7
            if count > 0:
                one_hot[count] = 1
            features.extend(one_hot)
        
        # 2. Market State (one-hot: 7 goods x 5 slots = 35)
        for g in self.good_types:
            market_count = obs.market_goods[g]
            market_count = min(market_count, 4)
            one_hot = [0] * 5
            if market_count > 0:
                one_hot[market_count] = 1
            features.extend(one_hot)
        
        # 3. Coin Stack Values (7 goods x 2: max_token, num_tokens)
        for g in self.good_types:
            tokens = obs.market_goods_coins.get(g, [])
            max_token = max(tokens) if tokens else 0
            num_tokens = len(tokens)
            # Normalize
            features.append(min(max_token / 10.0, 1.0))
            features.append(min(num_tokens / 5.0, 1.0))
        
        # 4. Game State Meta (5 features)
        deck_remaining = obs.market_reserved_goods_count
        hand_size = obs.actor_goods.count(include_camels=False)
        player_score = sum(sum(c) for c in obs.actor_goods_coins.values())
        
        features.append(min(deck_remaining / 30.0, 1.0))  # Normalized deck
        features.append(min(hand_size / 7.0, 1.0))  # Hand fill %
        features.append(min(player_score / 50.0, 1.0))  # Score estimate
        features.append(min(action_count / 20.0, 1.0))  # Legal actions available
        features.append(random.random())  # Noise for exploration
        
        return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

# ============================================================================
# NEURAL NETWORK POLICY
# ============================================================================
class PolicyNetwork(nn.Module):
    """Simple dense network to score actions."""
    
    def __init__(self, input_size=149, hidden_size=256, output_size=1):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.device = device
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============================================================================
# RL AGENT
# ============================================================================
class RLSharkAgent(Trader):
    """Shark Agent with RL policy network."""
    
    def __init__(self, seed, name, policy_net, encoder, epsilon=0.1):
        super().__init__(seed, name)
        self.policy_net = policy_net
        self.encoder = encoder
        self.epsilon = epsilon
        self.memory = []
        self.current_state = None
        
    def select_action(self, actions, observation, simulate_action_fnc):
        # Epsilon-greedy: explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Otherwise, use policy network to score actions
        state_features = self.encoder.encode(observation, len(actions))
        
        best_action = None
        best_score = float('-inf')
        
        for i, action in enumerate(actions):
            # Simple action encoding: append action type + index
            action_features = state_features.clone()
            action_type = [0, 0, 0]
            if isinstance(action, SellAction):
                action_type[0] = 1
            elif isinstance(action, TakeAction):
                action_type[1] = 1
            else:
                action_type[2] = 1
            
            # Combine state + action
            combined = torch.cat([
                action_features,
                torch.tensor([action_type + [i / len(actions)]], dtype=torch.float32, device=device)
            ], dim=1)
            
            with torch.no_grad():
                score = self.policy_net(combined).item()
            
            if score > best_score:
                best_score = score
                best_action = action
        
        self.current_state = state_features
        return best_action
    
    def calculate_reward(self, old_obs, new_obs, has_acted, env_reward):
        # Store experience for training
        reward = env_reward if env_reward else 0
        self.memory.append({
            'state': self.current_state,
            'reward': reward
        })

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_rl_agent(num_games=500, learning_rate=0.001):
    """Train RL agent against SmartAgent."""
    
    encoder = StateEncoder()
    policy_net = PolicyNetwork(input_size=149 + 4).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    rl_agent = RLSharkAgent(seed=42, name="RLShark", policy_net=policy_net, encoder=encoder, epsilon=0.2)
    opponent = SmartAgent(seed=43, name="SmartAgent")
    
    win_count = 0
    loss_count = 0
    wins_by_100 = []
    
    print(f"\n{'='*70}")
    print(f"TRAINING RL SHARK AGENT (GPU: {device})")
    print(f"{'='*70}\n")
    
    for game_idx in range(num_games):
        # Reset agents
        rl_agent.memory = []
        
        # Run game
        game = BasicBazaar(seed=42 + game_idx, players=[rl_agent, opponent])
        state = game.state
        episode_rewards = []
        
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
        
        # Calculate final scores
        rl_score = game.calculate_reward(rl_agent, state, state)
        opponent_score = game.calculate_reward(opponent, state, state)
        
        if rl_score > opponent_score:
            win_count += 1
        else:
            loss_count += 1
        
        wins_by_100.append(win_count)
        
        # Simple loss: encourage high scores
        if rl_agent.memory:
            total_reward = sum(m['reward'] for m in rl_agent.memory)
            loss = -torch.tensor(total_reward, dtype=torch.float32)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        # Progress
        if (game_idx + 1) % 50 == 0:
            wr = (win_count / (game_idx + 1)) * 100
            print(f"Games: {game_idx+1:4d} | Win Rate: {wr:5.1f}% | Wins: {win_count} | Losses: {loss_count}")
    
    final_wr = (win_count / num_games) * 100
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: Win Rate {final_wr:.1f}% ({win_count}/{num_games})")
    print(f"{'='*70}\n")
    
    return policy_net, encoder, win_count, num_games

# ============================================================================
# WEIGHT EXPORT: Convert trained weights to submission format
# ============================================================================
def export_weights_to_python(policy_net, output_file):
    """Export PyTorch weights to pure Python data structure."""
    
    weights_dict = {}
    
    for name, param in policy_net.named_parameters():
        # Convert to numpy, then to Python list for JSON serialization
        weights_dict[name] = param.data.cpu().numpy().tolist()
    
    # Write as Python dict literal
    with open(output_file, 'w') as f:
        f.write("# AUTO-GENERATED WEIGHTS\n")
        f.write("# Do not edit manually\n\n")
        f.write("TRAINED_WEIGHTS = ")
        f.write(repr(weights_dict))
        f.write("\n")
    
    print(f"Weights exported to: {output_file}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Train
    trained_net, encoder_obj, wins, total = train_rl_agent(num_games=500)
    
    # Export
    export_weights_to_python(
        trained_net,
        "bazaar-ai/agents/rl_weights.py"
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Weights saved to: bazaar-ai/agents/rl_weights.py")
    print(f"   Use custom_agent_rl.py as your submission file")
