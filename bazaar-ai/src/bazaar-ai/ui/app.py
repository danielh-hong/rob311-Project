from flask import Flask, send_file, jsonify, request
from flask_cors import CORS
import time
import sys
import importlib.util
from threading import Lock, Thread
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent.absolute()

# Add the parent folder to sys.path to enable importing backend as a package
sys.path.insert(0, str(BASE_DIR.parent))

# Now import from local backend package
from backend.bazaar import BasicBazaar
from backend.trader import Trader, SellAction, TakeAction, TradeAction
from backend.goods import GoodType, Goods
from backend.coins import BonusType

app = Flask(__name__)
CORS(app)


def discover_agents():
    """Discover all agent classes in the agents folder"""
    # Look for agents folder at the repository root level
    agents_dir = BASE_DIR.parent.parent.parent / 'agents'
    agents = {}
    
    if not agents_dir.exists():
        print(f"⚠️  Agents folder not found: {agents_dir}")
        return agents
    
    # Ensure backend is in sys.path for agent imports
    # (Already added earlier but ensure it's there for agent loading)
    backend_parent = BASE_DIR.parent
    if str(backend_parent) not in sys.path:
        sys.path.insert(0, str(backend_parent))
    
    print(f"\n🔍 Searching for agents in: {agents_dir}")
    
    for agent_file in agents_dir.glob('*.py'):
        if agent_file.name.startswith('_'):
            continue
        
        try:
            # Load the module
            module_name = agent_file.stem
            spec = importlib.util.spec_from_file_location(module_name, agent_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find all Trader subclasses in the module
            for item_name in dir(module):
                item = getattr(module, item_name)
                if (isinstance(item, type) and 
                    issubclass(item, Trader) and 
                    item is not Trader and
                    item.__module__ == module_name):
                    
                    agent_display_name = item_name.replace('Agent', '').replace('_', ' ').title()
                    agents[module_name] = {
                        'class': item,
                        'name': agent_display_name,
                        'module': module_name,
                        'file': agent_file.name
                    }
                    print(f"  ✓ Found agent: {agent_display_name} ({item_name})")
        
        except Exception as e:
            print(f"  ✗ Error loading {agent_file.name}: {e}")
    
    if agents:
        print(f"\n✅ Loaded {len(agents)} agent(s)")
    else:
        print("\n⚠️  No agents found")
    
    return agents


# Discover available agents
AVAILABLE_AGENTS = discover_agents()


class HumanPlayer(Trader):
    """A human player that waits for actions from the web interface"""
    
    def __init__(self, seed, name, player_id):
        super().__init__(seed, name)
        self.player_id = player_id  # "player1" or "player2"
        self.pending_action = None
    
    def select_action(self, actions, observation, simulate_action_fnc):
        """Wait for action to be received from web client"""
        return None


class GameState:
    def __init__(self):
        self.game = None
        self.player1_connected = False
        self.player2_connected = False
        self.player1 = None
        self.player2 = None
        self.game_started = False
        self.game_over = False
        self.waiting_for_player = None
        self.lock = Lock()
        self.last_player1_ping = 0
        self.last_player2_ping = 0
        self.game_mode = None  # 'human', 'human-vs-bot', or 'bot'
        self.pending_mode = None  # Mode selected before game starts
        self.bot_thread = None
        self.bot_running = False
        self.bot_paused = False  # New: pause control
        self.bot_step_requested = False  # New: step control
        self.bot_delay = 1.5  # Delay between bot moves in seconds
        self.bot_speed = 1.0  # Speed multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast)
        self.bot_timeout = 30.0  # Timeout in seconds for bot decision (0 = disabled)
        self.bot_timeout_player = None  # Player who timed out
        
    def check_players_ready(self):
        if self.game_mode == 'bot':
            return True  # Bots are always ready
        return self.player1_connected and self.player2_connected
    
    def create_game(self, mode='human', agent1_id=None, agent2_id=None):
        with self.lock:
            self.game_mode = mode
            
            if mode == 'human':
                if not self.check_players_ready():
                    return False
                    
                self.player1 = HumanPlayer(seed=356, name="Player 1", player_id="player1")
                self.player2 = HumanPlayer(seed=789, name="Player 2", player_id="player2")
                print("Game created with two human players")
            
            elif mode == 'human-vs-bot':
                # Human vs Bot mode
                if not self.player1_connected:
                    print("Player 1 not connected")
                    return False
                if not agent2_id or agent2_id not in AVAILABLE_AGENTS:
                    print(f"Invalid agent2_id: {agent2_id}")
                    return False
                
                agent2_info = AVAILABLE_AGENTS[agent2_id]
                
                self.player1 = HumanPlayer(seed=356, name="Player 1", player_id="player1")
                self.player2 = agent2_info['class'](seed=789, name=f"{agent2_info['name']}")
                
                print(f"Game created with human vs bot: Player 1 vs {self.player2.name}")
            
            elif mode == 'bot':
                # Create bot players
                if not agent1_id or agent1_id not in AVAILABLE_AGENTS:
                    print(f"Invalid agent1_id: {agent1_id}")
                    return False
                if not agent2_id or agent2_id not in AVAILABLE_AGENTS:
                    print(f"Invalid agent2_id: {agent2_id}")
                    return False
                
                agent1_info = AVAILABLE_AGENTS[agent1_id]
                agent2_info = AVAILABLE_AGENTS[agent2_id]
                
                self.player1 = agent1_info['class'](seed=356, name=f"{agent1_info['name']} 1")
                self.player2 = agent2_info['class'](seed=789, name=f"{agent2_info['name']} 2")
                
                print(f"Game created with bot players: {self.player1.name} vs {self.player2.name}")
            
            else:
                print(f"Unknown game mode: {mode}")
                return False
            
            players = [self.player1, self.player2]
            seed = int(time.time() * 1000)
            self.game = BasicBazaar(seed=seed, players=players)
            self.game_started = True
            self.game_over = False
            
            # Set waiting_for_player based on who starts
            if mode in ['human', 'human-vs-bot']:
                starting_actor = self.game.state.actor
                if isinstance(starting_actor, HumanPlayer):
                    self.waiting_for_player = starting_actor.player_id
                else:
                    # Bot goes first in human-vs-bot mode
                    self.waiting_for_player = None
                    if mode == 'human-vs-bot':
                        Thread(target=self._execute_bot_turn, daemon=True).start()
            else:
                self.waiting_for_player = None
            
            # Start bot thread if in bot mode
            if mode == 'bot':
                self.bot_running = True
                self.bot_paused = True  # Start paused by default
                self.bot_thread = Thread(target=self.run_bot_game, daemon=True)
                self.bot_thread.start()
                print("🤖 Bot game starting in PAUSED state")
            
            return True
    
    def run_bot_game(self):
        """Run the game automatically with bot players"""
        print("🤖 Bot game thread started")
        
        while self.bot_running and not self.game_over:
            try:
                # Check if paused (wait until unpaused or step requested)
                while self.bot_paused and not self.bot_step_requested:
                    if not self.bot_running:
                        break
                    time.sleep(0.1)  # Check every 100ms
                    continue
                
                # If we're exiting due to stop, break
                if not self.bot_running:
                    break
                
                # If step was requested, clear the flag
                if self.bot_step_requested:
                    self.bot_step_requested = False
                
                with self.lock:
                    if not self.game or self.game.terminal(self.game.state):
                        self.game_over = True
                        print("🏁 Bot game finished")
                        break
                    
                    # Get current player
                    current_player = self.game.state.actor
                    
                    # Get legal actions
                    actions = self.game.all_actions(current_player, self.game.state)
                    
                    if not actions:
                        print(f"⚠️  No legal actions for {current_player.name}")
                        break
                    
                    # Get observation
                    observation = self.game.observe(current_player, self.game.state)
                    
                    # Let the bot select an action
                    def simulate_action(obs, act):
                        temp_state = self.game.state.clone()
                        temp_state = self.game.apply_action(temp_state, act.clone())
                        return self.game.observe(current_player, temp_state)
                    
                    # Check for timeout if enabled
                    action = None
                    if self.bot_timeout > 0:
                        start_time = time.time()
                        try:
                            # Run bot decision with timeout tracking
                            action = current_player.select_action(actions, observation, simulate_action)
                            elapsed = time.time() - start_time
                            if elapsed > self.bot_timeout:
                                print(f"⏱️  Bot {current_player.name} exceeded timeout: {elapsed:.2f}s > {self.bot_timeout}s")
                                self.bot_timeout_player = current_player.name
                                self.game_over = True
                                break
                        except Exception as e:
                            print(f"❌ Bot {current_player.name} error: {e}")
                            self.bot_timeout_player = current_player.name
                            self.game_over = True
                            break
                    else:
                        action = current_player.select_action(actions, observation, simulate_action)
                    
                    if not action:
                        print(f"⚠️  Bot {current_player.name} returned no action")
                        break
                    
                    # Execute the action
                    self.game.old_state = self.game.state.clone()
                    self.game.state = self.game.apply_action(self.game.state.clone(), action.clone())
                    
                    # Update rewards
                    for player in self.game.players:
                        has_acted = player == self.game.old_state.actor
                        old_observation = self.game.observe(player, self.game.old_state)
                        current_observation = self.game.observe(player, self.game.state)
                        environment_reward = self.game.calculate_reward(
                            player.clone(), self.game.old_state.clone(), self.game.state.clone()
                        )
                        player.calculate_reward(
                            old_observation.clone(), current_observation.clone(),
                            has_acted, environment_reward
                        )
                    
                    self.game.round += 1
                    
                    print(f"🎮 Round {self.game.round}: {current_player.name} played {action.trader_action_type.value}")
                
                # Sleep between moves to make it watchable (only if not paused)
                if not self.bot_paused:
                    # Apply speed multiplier: lower speed = longer delay
                    actual_delay = self.bot_delay / self.bot_speed
                    time.sleep(actual_delay)
                
            except Exception as e:
                print(f"❌ Error in bot game: {e}")
                import traceback
                traceback.print_exc()
                break
        
        self.bot_running = False
        print("🤖 Bot game thread stopped")
    
    def get_public_state(self):
        """Get state visible to everyone"""
        if not self.game:
            return {
                'type': 'public',
                'gameStarted': False,
                'gameMode': self.game_mode,
                'pendingMode': self.pending_mode,
                'playersReady': self.check_players_ready(),
                'player1Connected': self.player1_connected,
                'player2Connected': self.player2_connected,
                'round': 0,
                'market': {},
                'marketCoins': {},
                'marketBonusTokens': {'THREE': 0, 'FOUR': 0, 'FIVE': 0},
                'deckSize': 0,
                'players': [
                    {'name': 'Player 1', 'score': 0, 'camelCount': 0, 'goods': {}, 'coins': {}, 'bonusCounts': {}},
                    {'name': 'Player 2', 'score': 0, 'camelCount': 0, 'goods': {}, 'coins': {}, 'bonusCounts': {}}
                ],
                'currentPlayer': None,
                'isTerminal': False,
                'lastAction': None,
                'waitingForPlayer': None,
                'botPaused': False,
                'botRunning': False,
                'botSpeed': 1.0,
                'botTimeout': 30.0,
                'botTimeoutPlayer': None
            }
        
        state = self.game.state
        is_terminal = self.game.terminal(state) or self.bot_timeout_player is not None
        
        # Market goods
        market_goods = {good_type.name: state.goods[good_type] 
                       for good_type in GoodType if state.goods[good_type] > 0}
        
        # Market coins
        market_coins = {good_type.name: state.coins.goods_coins[good_type]
                       for good_type in GoodType if good_type != GoodType.CAMEL 
                       and state.coins.goods_coins[good_type]}
        
        # Market bonus tokens remaining
        market_bonus_tokens = {bonus_type.name: len(state.coins.bonus_coins[bonus_type])
                              for bonus_type in BonusType}
        
        # Player data
        players_data = []
        for player in self.game.players:
            player_goods = {good_type.name: state.player_goods[player][good_type]
                          for good_type in GoodType if state.player_goods[player][good_type] > 0}
            
            player_coins = {good_type.name: state.player_coins[player].goods_coins[good_type]
                          for good_type in GoodType 
                          if state.player_coins[player].goods_coins[good_type]}
            
            bonus_counts = {bonus_type.name: len(state.player_coins[player].bonus_coins[bonus_type])
                          for bonus_type in BonusType 
                          if len(state.player_coins[player].bonus_coins[bonus_type]) > 0}
            
            # Calculate score components
            raw_score = sum(sum(coins) for coins in state.player_coins[player].goods_coins.values())
            score = raw_score
            camel_bonus = 0
            bonus_3x = 0
            bonus_4x = 0
            bonus_5x = 0
            
            if is_terminal:
                # Add bonus tokens
                bonus_3x = sum(state.player_coins[player].bonus_coins[BonusType.THREE])
                bonus_4x = sum(state.player_coins[player].bonus_coins[BonusType.FOUR])
                bonus_5x = sum(state.player_coins[player].bonus_coins[BonusType.FIVE])
                score += bonus_3x + bonus_4x + bonus_5x
                
                # Add camel bonus
                other_player = state.get_non_actor() if player == state.actor else state.actor
                if state.player_goods[player][GoodType.CAMEL] > state.player_goods[other_player][GoodType.CAMEL]:
                    camel_bonus = state.camel_bonus
                    score += camel_bonus
            
            players_data.append({
                'name': player.name,
                'score': score,
                'rawScore': raw_score,
                'camelBonus': camel_bonus,
                'bonus3x': bonus_3x,
                'bonus4x': bonus_4x,
                'bonus5x': bonus_5x,
                'camelCount': state.player_goods[player][GoodType.CAMEL],
                'goods': player_goods,
                'coins': player_coins,
                'bonusCounts': bonus_counts
            })
        
        # Last action
        last_action = None
        if state.action:
            action = state.action
            last_action = {
                'player': self.game.old_state.actor.name if hasattr(self.game, 'old_state') else 'Unknown',
                'type': action.trader_action_type.value,
                'offered': {gt.name: action.offered_goods[gt] for gt in GoodType if action.offered_goods[gt] > 0},
                'requested': {gt.name: action.requested_goods[gt] for gt in GoodType if action.requested_goods[gt] > 0}
            }
        
        return {
            'type': 'public',
            'gameStarted': self.game_started,
            'gameMode': self.game_mode,
            'playersReady': self.check_players_ready(),
            'player1Connected': self.player1_connected,
            'player2Connected': self.player2_connected,
            'round': self.game.round,
            'market': market_goods,
            'marketCoins': market_coins,
            'marketBonusTokens': market_bonus_tokens,
            'deckSize': len(state.reserved_goods),
            'players': players_data,
            'currentPlayer': state.actor.name if state.actor else None,
            'isTerminal': is_terminal,
            'lastAction': last_action,
            'waitingForPlayer': self.waiting_for_player,
            'botPaused': self.bot_paused if self.game_mode == 'bot' else None,
            'botRunning': self.bot_running if self.game_mode == 'bot' else None,
            'botSpeed': self.bot_speed if self.game_mode == 'bot' else None,
            'botTimeout': self.bot_timeout,
            'botTimeoutPlayer': self.bot_timeout_player
        }
    
    def get_player_state(self, player):
        """Get private state for a specific player"""
        if not self.game:
            return {'type': 'private', 'gameStarted': False, 'myTurn': False, 'goods': {}, 'legalActions': []}
        
        state = self.game.state
        is_my_turn = (state.actor == player)
        
        # Player's goods (including camels)
        player_goods = {good_type.name: state.player_goods[player][good_type]
                       for good_type in GoodType if state.player_goods[player][good_type] > 0}
        
        return {
            'type': 'private',
            'gameStarted': self.game_started,
            'myTurn': is_my_turn,
            'goods': player_goods
        }
    
    def is_action_valid(self, player, action_dict):
        """Validate action including 7-card hand limit"""
        if not self.game:
            return False
        
        state = self.game.state
        
        # Count current non-camel cards
        current_hand_size = sum(state.player_goods[player][gt] for gt in GoodType if gt != GoodType.CAMEL)
        
        # Count non-camel cards being offered/requested
        offered = action_dict.get('offered', {})
        requested = action_dict.get('requested', {})
        
        offered_non_camel = sum(count for good_name, count in offered.items() if good_name != 'CAMEL')
        requested_non_camel = sum(count for good_name, count in requested.items() if good_name != 'CAMEL')
        
        new_hand_size = current_hand_size - offered_non_camel + requested_non_camel
        
        if new_hand_size > 7:
            print(f"Action rejected: would result in {new_hand_size} cards (max 7)")
            return False
        
        return True
    
    def action_from_dict(self, player, action_dict):
        """Convert action dict to action object"""
        action_type = action_dict['type']
        offered = action_dict.get('offered', {})
        requested = action_dict.get('requested', {})
        
        if action_type == "Sell":
            for good_name, count in offered.items():
                return SellAction(player, GoodType[good_name], count)
        elif action_type == "Take":
            for good_name, count in requested.items():
                return TakeAction(player, GoodType[good_name], count)
        elif action_type == "Trade":
            net = Goods()
            for good_type in GoodType:
                offered_count = offered.get(good_type.name, 0)
                requested_count = requested.get(good_type.name, 0)
                net._goods[good_type] = requested_count - offered_count
            return TradeAction(player, net)
        
        return None
    
    def execute_turn(self, action):
        """Execute a turn"""
        with self.lock:
            if not self.game or self.game.terminal(self.game.state):
                return False
            
            actor = self.game.state.actor
            print(f"Processing action from {actor.name}: {action.trader_action_type.value}")
            
            self.game.old_state = self.game.state.clone()
            self.game.state = self.game.apply_action(self.game.state.clone(), action.clone())
            
            for player in self.game.players:
                has_acted = player == self.game.old_state.actor
                old_observation = self.game.observe(player, self.game.old_state)
                current_observation = self.game.observe(player, self.game.state)
                environment_reward = self.game.calculate_reward(
                    player.clone(), self.game.old_state.clone(), self.game.state.clone()
                )
                player.calculate_reward(
                    old_observation.clone(), current_observation.clone(),
                    has_acted, environment_reward
                )
            
            self.game.round += 1
            
            if self.game.terminal(self.game.state):
                self.game_over = True
                self.waiting_for_player = None
                print("Game Over!")
            else:
                # In human-vs-bot mode, if it's the bot's turn, play it automatically
                if self.game_mode == 'human-vs-bot':
                    next_actor = self.game.state.actor
                    if not isinstance(next_actor, HumanPlayer):
                        # It's the bot's turn - execute it automatically
                        Thread(target=self._execute_bot_turn, daemon=True).start()
                        self.waiting_for_player = None
                    else:
                        self.waiting_for_player = next_actor.player_id
                elif self.game_mode == 'human':
                    self.waiting_for_player = self.game.state.actor.player_id
                else:
                    self.waiting_for_player = None
            
            return True
    
    def _execute_bot_turn(self):
        """Execute a bot turn (used in human-vs-bot mode)"""
        time.sleep(0.5)  # Small delay for visual effect
        
        with self.lock:
            if not self.game or self.game.terminal(self.game.state):
                return
            
            current_player = self.game.state.actor
            if isinstance(current_player, HumanPlayer):
                return  # Don't execute if it's a human's turn
            
            # Get legal actions
            actions = self.game.all_actions(current_player, self.game.state)
            
            if not actions:
                print(f"⚠️  No legal actions for {current_player.name}")
                return
            
            # Get observation for the bot
            observation = self.game.observe(current_player, self.game.state)
            
            # Let bot select action with timeout checking
            chosen_action = None
            if self.bot_timeout > 0:
                start_time = time.time()
                try:
                    chosen_action = current_player.select_action(
                        actions, observation,
                        lambda action: self.game.apply_action(self.game.state.clone(), action)
                    )
                    elapsed = time.time() - start_time
                    if elapsed > self.bot_timeout:
                        print(f"⏱️  Bot {current_player.name} exceeded timeout: {elapsed:.2f}s > {self.bot_timeout}s")
                        self.bot_timeout_player = current_player.name
                        self.game_over = True
                        self.waiting_for_player = None
                        return
                except Exception as e:
                    print(f"❌ Bot {current_player.name} error: {e}")
                    self.bot_timeout_player = current_player.name
                    self.game_over = True
                    self.waiting_for_player = None
                    return
            else:
                chosen_action = current_player.select_action(
                    actions, observation,
                    lambda action: self.game.apply_action(self.game.state.clone(), action)
                )
            
            if chosen_action:
                print(f"🤖 Bot {current_player.name} chose: {chosen_action.trader_action_type.value}")
                
                # Execute the action
                self.game.old_state = self.game.state.clone()
                self.game.state = self.game.apply_action(self.game.state.clone(), chosen_action.clone())
                
                for player in self.game.players:
                    has_acted = player == self.game.old_state.actor
                    old_observation = self.game.observe(player, self.game.old_state)
                    current_observation = self.game.observe(player, self.game.state)
                    environment_reward = self.game.calculate_reward(
                        player.clone(), self.game.old_state.clone(), self.game.state.clone()
                    )
                    player.calculate_reward(
                        old_observation.clone(), current_observation.clone(),
                        has_acted, environment_reward
                    )
                
                self.game.round += 1
                
                if self.game.terminal(self.game.state):
                    self.game_over = True
                    self.waiting_for_player = None
                    print("🏁 Game Over!")
                else:
                    # Check if next turn is also bot (shouldn't happen in human-vs-bot)
                    next_actor = self.game.state.actor
                    if isinstance(next_actor, HumanPlayer):
                        self.waiting_for_player = next_actor.player_id
                    else:
                        # Chain bot turns if needed
                        self._execute_bot_turn()


# Global game state
game_state = GameState()


@app.route('/')
def index():
    # Try to find host.html
    host_path = BASE_DIR / 'host.html'
    host_orig_path = BASE_DIR / 'host.html.original'
    
    if host_path.exists():
        return send_file(host_path)
    elif host_orig_path.exists():
        return send_file(host_orig_path)
    else:
        return f"Error: host.html not found in {BASE_DIR}<br>Files present: {list(BASE_DIR.glob('*.html'))}", 404


@app.route('/host.html')
def host():
    host_path = BASE_DIR / 'host.html'
    host_orig_path = BASE_DIR / 'host.html.original'
    
    if host_path.exists():
        return send_file(host_path)
    elif host_orig_path.exists():
        return send_file(host_orig_path)
    else:
        return f"Error: host.html not found in {BASE_DIR}", 404


@app.route('/player.html')
def player():
    player_path = BASE_DIR / 'player.html'
    player_orig_path = BASE_DIR / 'player.html.original'
    
    if player_path.exists():
        return send_file(player_path)
    elif player_orig_path.exists():
        return send_file(player_orig_path)
    else:
        return f"Error: player.html not found in {BASE_DIR}", 404


@app.route('/shared.js')
def shared_js():
    shared_path = BASE_DIR / 'shared.js'
    
    if shared_path.exists():
        return send_file(shared_path, mimetype='application/javascript')
    else:
        return f"Error: shared.js not found in {BASE_DIR}", 404


@app.route('/api/agents')
def get_agents():
    """Get list of available agents"""
    agents_list = [
        {
            'id': agent_id,
            'name': info['name'],
            'module': info['module'],
            'file': info['file']
        }
        for agent_id, info in AVAILABLE_AGENTS.items()
    ]
    return jsonify({'agents': agents_list})


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """Set the pending game mode (called when user selects a mode on setup screen)"""
    data = request.json
    mode = data.get('mode')
    
    if mode in ['human-vs-human', 'human-vs-bot', 'bot-vs-bot']:
        game_state.pending_mode = mode
        print(f"Pending mode set to: {mode}")
        return jsonify({'success': True})
    
    return jsonify({'error': 'Invalid mode'}), 400


@app.route('/api/connect', methods=['POST'])
def connect():
    data = request.json
    client_type = data.get('clientType')
    
    print(f"Connection request: clientType={client_type}")
    print(f"Current state: player1={game_state.player1_connected}, player2={game_state.player2_connected}, pending_mode={game_state.pending_mode}")
    
    if client_type == 'player':
        # If bot-vs-bot mode is selected, reject all player connections
        if game_state.pending_mode == 'bot-vs-bot':
            print("Bot vs Bot mode selected - rejecting human player connection")
            return jsonify({'error': 'Bot vs Bot mode - human players cannot connect'}), 400
        
        # If human-vs-bot mode is selected, only allow player 1 to connect
        if game_state.pending_mode == 'human-vs-bot':
            if not game_state.player1_connected:
                game_state.player1_connected = True
                game_state.last_player1_ping = time.time()
                print("Assigned as Player 1 (Human vs Bot mode)")
                return jsonify({'playerId': 'player1', 'playerName': 'Player 1', 'sessionId': 'player1'})
            else:
                print("Human vs Bot mode - only one player allowed")
                return jsonify({'error': 'Human vs Bot mode - only one player can connect'}), 400
        
        # Human-vs-human mode (default) - allow both players
        if not game_state.player1_connected:
            game_state.player1_connected = True
            game_state.last_player1_ping = time.time()
            print("Assigned as Player 1")
            return jsonify({'playerId': 'player1', 'playerName': 'Player 1', 'sessionId': 'player1'})
        elif not game_state.player2_connected:
            game_state.player2_connected = True
            game_state.last_player2_ping = time.time()
            print("Assigned as Player 2")
            return jsonify({'playerId': 'player2', 'playerName': 'Player 2', 'sessionId': 'player2'})
        else:
            print("Game is full, rejecting connection")
            return jsonify({'error': 'Game is full'}), 400
    
    return jsonify({'success': True})


@app.route('/api/ping', methods=['POST'])
def ping():
    data = request.json
    player_id = data.get('playerId') or data.get('sessionId')
    
    if player_id == 'player1':
        game_state.last_player1_ping = time.time()
    elif player_id == 'player2':
        game_state.last_player2_ping = time.time()
    
    return jsonify({'success': True})


@app.route('/api/debug')
def debug():
    """Debug endpoint to check connection state"""
    return jsonify({
        'player1_connected': game_state.player1_connected,
        'player2_connected': game_state.player2_connected,
        'last_player1_ping': game_state.last_player1_ping,
        'last_player2_ping': game_state.last_player2_ping,
        'current_time': time.time()
    })


@app.route('/api/state')
def get_state():
    # Check for disconnected players (no ping for 10 seconds)
    current_time = time.time()
    
    # Only check if player was connected and has a last ping time
    if game_state.player1_connected:
        if game_state.last_player1_ping == 0:
            # Player 1 connected but never pinged - just connected
            pass
        elif current_time - game_state.last_player1_ping > 10:
            game_state.player1_connected = False
            print(f"Player 1 disconnected (timeout) - last ping was {current_time - game_state.last_player1_ping:.1f}s ago")
    
    if game_state.player2_connected:
        if game_state.last_player2_ping == 0:
            # Player 2 connected but never pinged - just connected
            pass
        elif current_time - game_state.last_player2_ping > 10:
            game_state.player2_connected = False
            print(f"Player 2 disconnected (timeout) - last ping was {current_time - game_state.last_player2_ping:.1f}s ago")
    
    return jsonify(game_state.get_public_state())


@app.route('/api/player_state')
def get_player_state():
    player_id = request.args.get('playerId')
    
    if player_id == 'player1' and game_state.player1:
        player_state = game_state.get_player_state(game_state.player1)
    elif player_id == 'player2' and game_state.player2:
        player_state = game_state.get_player_state(game_state.player2)
    else:
        player_state = {'type': 'private', 'gameStarted': False, 'myTurn': False, 'goods': {}}
    
    return jsonify(player_state)


@app.route('/api/start', methods=['POST'])
def start_game():
    data = request.json
    mode = data.get('mode', 'human')
    agent1_id = data.get('agent1')
    agent2_id = data.get('agent2')
    
    if game_state.create_game(mode=mode, agent1_id=agent1_id, agent2_id=agent2_id):
        return jsonify({'success': True})
    return jsonify({'error': 'Cannot start game'}), 400


@app.route('/api/action', methods=['POST'])
def submit_action():
    data = request.json
    player_id = data.get('playerId')
    action_dict = data.get('action')
    
    player = None
    if player_id == 'player1' and game_state.player1:
        player = game_state.player1
    elif player_id == 'player2' and game_state.player2:
        player = game_state.player2
    
    if not player:
        return jsonify({'error': 'Invalid player'}), 400
    
    if game_state.game.state.actor != player:
        return jsonify({'error': 'Not your turn'}), 400
    
    if not game_state.is_action_valid(player, action_dict):
        return jsonify({'error': 'Invalid action: would exceed 7 card hand limit'}), 400
    
    action = game_state.action_from_dict(player, action_dict)
    if not action:
        return jsonify({'error': 'Failed to process action'}), 400
    
    if game_state.execute_turn(action):
        return jsonify({'success': True})
    
    return jsonify({'error': 'Failed to execute action'}), 400


@app.route('/api/reset', methods=['POST'])
def reset_game():
    # Stop bot thread if running
    if game_state.bot_running:
        game_state.bot_running = False
        if game_state.bot_thread:
            game_state.bot_thread.join(timeout=2)
    
    game_state.game = None
    game_state.game_started = False
    game_state.game_over = False
    game_state.waiting_for_player = None
    game_state.game_mode = None
    game_state.pending_mode = None
    game_state.bot_paused = False
    game_state.bot_step_requested = False
    game_state.bot_timeout_player = None  # Reset timeout info
    return jsonify({'success': True})


@app.route('/api/bot/pause', methods=['POST'])
def pause_bot():
    """Pause the bot game"""
    if game_state.game_mode == 'bot':
        game_state.bot_paused = True
        print("⏸️  Bot game paused")
        return jsonify({'success': True, 'paused': True})
    return jsonify({'error': 'Not in bot mode'}), 400


@app.route('/api/bot/resume', methods=['POST'])
def resume_bot():
    """Resume the bot game"""
    if game_state.game_mode == 'bot':
        game_state.bot_paused = False
        print("▶️  Bot game resumed")
        return jsonify({'success': True, 'paused': False})
    return jsonify({'error': 'Not in bot mode'}), 400


@app.route('/api/bot/step', methods=['POST'])
def step_bot():
    """Execute one step in the bot game"""
    if game_state.game_mode == 'bot':
        if not game_state.bot_paused:
            game_state.bot_paused = True
            print("⏸️  Bot game paused (for step)")
        game_state.bot_step_requested = True
        print("👣 Bot step requested")
        return jsonify({'success': True})
    return jsonify({'error': 'Not in bot mode'}), 400


@app.route('/api/bot/speed', methods=['POST'])
def set_bot_speed():
    """Set the bot game speed"""
    if game_state.game_mode == 'bot':
        data = request.json
        speed = data.get('speed', 1.0)
        # Clamp speed between 0.25x and 4x
        game_state.bot_speed = max(0.25, min(4.0, float(speed)))
        print(f"⚡ Bot speed set to {game_state.bot_speed}x")
        return jsonify({'success': True, 'speed': game_state.bot_speed})
    return jsonify({'error': 'Not in bot mode'}), 400


@app.route('/api/bot/timeout', methods=['POST'])
def set_bot_timeout():
    """Set the bot timeout in seconds (0 = disabled)"""
    data = request.json
    timeout = data.get('timeout', 30.0)
    # Clamp timeout between 0 (disabled) and 300 seconds (5 minutes)
    game_state.bot_timeout = max(0, min(300.0, float(timeout)))
    print(f"⏱️  Bot timeout set to {game_state.bot_timeout}s {'(disabled)' if game_state.bot_timeout == 0 else ''}")
    return jsonify({'success': True, 'timeout': game_state.bot_timeout})


@app.route('/api/get_player_url', methods=['GET'])
def get_player_url():
    """Get the player URL for QR code generation"""
    # Get the host from the request
    host = request.host.split(':')[0]
    port = request.host.split(':')[1] if ':' in request.host else '5000'
    player_url = f"http://{host}:{port}/player.html"
    return jsonify({'playerUrl': player_url})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
