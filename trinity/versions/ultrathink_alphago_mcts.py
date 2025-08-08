#!/usr/bin/env python3
"""
ULTRATHINK AlphaGo MCTS - Monte Carlo Tree Search for Trading
Adapts AlphaGo's decision-making to financial markets
Real tree search, real simulations, real decisions
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import random
from collections import defaultdict
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ALPHAGO_MCTS')

class TradingState:
    """Represents a state in the trading decision tree"""
    
    def __init__(self, price: float, position: str = 'none', 
                 holding_period: int = 0, pnl: float = 0.0):
        self.price = price
        self.position = position  # 'long', 'short', 'none'
        self.holding_period = holding_period
        self.pnl = pnl
        self.entry_price = price if position != 'none' else None
    
    def __hash__(self):
        return hash((round(self.price, 2), self.position, self.holding_period))
    
    def __eq__(self, other):
        return (round(self.price, 2) == round(other.price, 2) and 
                self.position == other.position and
                self.holding_period == other.holding_period)
    
    def __repr__(self):
        return f"State(price={self.price:.2f}, pos={self.position}, hold={self.holding_period}, pnl={self.pnl:.2f})"

class MCTSNode:
    """Node in the Monte Carlo Tree"""
    
    def __init__(self, state: TradingState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = ['buy', 'sell', 'hold']
        
        # Remove invalid actions based on current position
        if state.position == 'long':
            self.untried_actions = ['sell', 'hold']
        elif state.position == 'short':
            self.untried_actions = ['buy', 'hold']
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1.4):
        """UCB1 formula for selection"""
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]
    
    def expand(self, action: str, next_state: TradingState):
        """Expand tree with new child"""
        child = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child
        self.untried_actions.remove(action)
        return child

class AlphaGoMCTS:
    """Monte Carlo Tree Search adapted for trading"""
    
    def __init__(self, simulation_depth: int = 20, num_simulations: int = 100):
        self.simulation_depth = simulation_depth
        self.num_simulations = num_simulations
        self.c_param = 1.4  # Exploration parameter
        
        # Market model parameters
        self.volatility = 0.02  # 2% daily volatility
        self.drift = 0.0001  # Slight upward drift
        self.transaction_cost = 0.001  # 0.1% per trade
        
        # Risk parameters
        self.stop_loss = 0.03  # 3% stop loss
        self.take_profit = 0.06  # 6% take profit
        self.max_holding = 20  # Maximum holding period
        
        # Performance tracking
        self.total_simulations = 0
        self.decision_history = []
        
        logger.info(f"🎯 AlphaGo MCTS initialized")
        logger.info(f"   Simulations per move: {self.num_simulations}")
        logger.info(f"   Simulation depth: {self.simulation_depth}")
    
    def simulate_price(self, current_price: float, steps: int = 1) -> List[float]:
        """Simulate future price path using geometric Brownian motion"""
        prices = [current_price]
        
        for _ in range(steps):
            # Geometric Brownian Motion
            dt = 1  # Daily steps
            random_shock = np.random.randn()
            price_change = self.drift * dt + self.volatility * math.sqrt(dt) * random_shock
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        return prices[1:]  # Return future prices only
    
    def get_next_state(self, state: TradingState, action: str, next_price: float) -> TradingState:
        """Calculate next state based on action and price"""
        new_position = state.position
        new_holding = state.holding_period
        new_pnl = state.pnl
        entry_price = state.entry_price
        
        if action == 'buy':
            if state.position == 'none':
                # Open long position
                new_position = 'long'
                new_holding = 1
                entry_price = next_price * (1 + self.transaction_cost)
            elif state.position == 'short':
                # Close short and open long
                short_pnl = (state.entry_price - next_price) / state.entry_price
                new_pnl = state.pnl + short_pnl - self.transaction_cost
                new_position = 'long'
                new_holding = 1
                entry_price = next_price * (1 + self.transaction_cost)
        
        elif action == 'sell':
            if state.position == 'none':
                # Open short position
                new_position = 'short'
                new_holding = 1
                entry_price = next_price * (1 - self.transaction_cost)
            elif state.position == 'long':
                # Close long and open short
                long_pnl = (next_price - state.entry_price) / state.entry_price
                new_pnl = state.pnl + long_pnl - self.transaction_cost
                new_position = 'short'
                new_holding = 1
                entry_price = next_price * (1 - self.transaction_cost)
        
        else:  # hold
            new_holding = state.holding_period + 1 if state.position != 'none' else 0
            
            # Check stop loss and take profit
            if state.position == 'long' and entry_price:
                current_pnl = (next_price - entry_price) / entry_price
                if current_pnl <= -self.stop_loss or current_pnl >= self.take_profit:
                    new_pnl = state.pnl + current_pnl - self.transaction_cost
                    new_position = 'none'
                    new_holding = 0
                    entry_price = None
            
            elif state.position == 'short' and entry_price:
                current_pnl = (entry_price - next_price) / entry_price
                if current_pnl <= -self.stop_loss or current_pnl >= self.take_profit:
                    new_pnl = state.pnl + current_pnl - self.transaction_cost
                    new_position = 'none'
                    new_holding = 0
                    entry_price = None
            
            # Force close if holding too long
            if new_holding >= self.max_holding and state.position != 'none':
                if state.position == 'long':
                    current_pnl = (next_price - entry_price) / entry_price
                else:
                    current_pnl = (entry_price - next_price) / entry_price
                new_pnl = state.pnl + current_pnl - self.transaction_cost
                new_position = 'none'
                new_holding = 0
                entry_price = None
        
        next_state = TradingState(next_price, new_position, new_holding, new_pnl)
        next_state.entry_price = entry_price
        return next_state
    
    def rollout(self, state: TradingState) -> float:
        """Simulate random play from current state to get reward"""
        current_state = TradingState(state.price, state.position, 
                                    state.holding_period, state.pnl)
        current_state.entry_price = state.entry_price
        
        # Simulate future prices
        future_prices = self.simulate_price(state.price, self.simulation_depth)
        
        for price in future_prices:
            # Choose random valid action
            if current_state.position == 'long':
                action = random.choice(['sell', 'hold', 'hold'])  # Bias toward holding
            elif current_state.position == 'short':
                action = random.choice(['buy', 'hold', 'hold'])
            else:
                action = random.choice(['buy', 'sell', 'hold'])
            
            current_state = self.get_next_state(current_state, action, price)
        
        # Return final PnL as reward
        final_reward = current_state.pnl
        
        # Add penalty for open positions at end
        if current_state.position != 'none' and current_state.entry_price:
            if current_state.position == 'long':
                unrealized = (current_state.price - current_state.entry_price) / current_state.entry_price
            else:
                unrealized = (current_state.entry_price - current_state.price) / current_state.entry_price
            final_reward += unrealized * 0.5  # Discount unrealized PnL
        
        return final_reward
    
    def search(self, initial_state: TradingState) -> str:
        """Run MCTS to find best action"""
        root = MCTSNode(initial_state)
        
        for sim in range(self.num_simulations):
            node = root
            
            # Selection - traverse tree using UCB1
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.best_child(self.c_param)
            
            # Expansion - add new child if not fully expanded
            if not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                next_price = self.simulate_price(node.state.price, 1)[0]
                next_state = self.get_next_state(node.state, action, next_price)
                node = node.expand(action, next_state)
            
            # Simulation - rollout from new node
            reward = self.rollout(node.state)
            
            # Backpropagation - update all nodes in path
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent
        
        self.total_simulations += self.num_simulations
        
        # Choose best action based on visit count
        if len(root.children) == 0:
            return 'hold'
        
        best_action = max(root.children.items(), 
                         key=lambda x: x[1].visits)[0]
        
        # Log decision statistics
        logger.debug(f"MCTS Decision Statistics:")
        for action, child in root.children.items():
            avg_value = child.value / child.visits if child.visits > 0 else 0
            logger.debug(f"   {action}: visits={child.visits}, avg_value={avg_value:.4f}")
        
        return best_action
    
    def analyze(self, market_data: Dict) -> Dict:
        """Analyze market using MCTS and return trading decision"""
        current_price = market_data.get('current_price', 100)
        current_position = market_data.get('position', 'none')
        holding_period = market_data.get('holding_period', 0)
        current_pnl = market_data.get('pnl', 0.0)
        
        # Create current state
        state = TradingState(current_price, current_position, holding_period, current_pnl)
        if current_position != 'none':
            state.entry_price = market_data.get('entry_price', current_price)
        
        # Run MCTS to find best action
        start_time = time.time()
        best_action = self.search(state)
        search_time = time.time() - start_time
        
        # Calculate confidence based on action distribution
        root = MCTSNode(state)
        temp_simulations = min(20, self.num_simulations)  # Quick confidence check
        
        action_visits = defaultdict(int)
        for _ in range(temp_simulations):
            sim_action = self.search(state)
            action_visits[sim_action] += 1
        
        confidence = action_visits[best_action] / temp_simulations
        
        # Store decision
        self.decision_history.append({
            'timestamp': time.time(),
            'action': best_action,
            'confidence': confidence,
            'state': str(state)
        })
        
        return {
            'signal': best_action,
            'confidence': confidence,
            'search_time_ms': search_time * 1000,
            'simulations_run': self.num_simulations,
            'total_simulations': self.total_simulations,
            'expected_value': self._calculate_expected_value(state, best_action),
            'reason': f'AlphaGo MCTS ({self.num_simulations} simulations, {search_time:.2f}s)'
        }
    
    def _calculate_expected_value(self, state: TradingState, action: str) -> float:
        """Calculate expected value of taking action"""
        rewards = []
        for _ in range(20):
            next_price = self.simulate_price(state.price, 1)[0]
            next_state = self.get_next_state(state, action, next_price)
            reward = self.rollout(next_state)
            rewards.append(reward)
        
        return np.mean(rewards)

# Test the AlphaGo MCTS system
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 ULTRATHINK ALPHAGO - MONTE CARLO TREE SEARCH")
    print("✨ Deep Decision Trees for Optimal Trading")
    print("=" * 60)
    
    # Initialize MCTS
    mcts = AlphaGoMCTS(simulation_depth=10, num_simulations=50)
    
    # Test different market scenarios
    scenarios = [
        {
            'name': 'Neutral Market',
            'current_price': 100,
            'position': 'none',
            'holding_period': 0,
            'pnl': 0
        },
        {
            'name': 'Profitable Long',
            'current_price': 105,
            'position': 'long',
            'entry_price': 100,
            'holding_period': 5,
            'pnl': 0
        },
        {
            'name': 'Losing Short',
            'current_price': 102,
            'position': 'short',
            'entry_price': 100,
            'holding_period': 3,
            'pnl': -0.01
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Current price: ${scenario['current_price']}")
        print(f"   Position: {scenario['position']}")
        
        result = mcts.analyze(scenario)
        
        print(f"\n   🎯 Decision: {result['signal'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Expected value: {result['expected_value']:.4f}")
        print(f"   Search time: {result['search_time_ms']:.1f}ms")
    
    print(f"\n📈 Total simulations run: {mcts.total_simulations}")
    print("\n✅ ALPHAGO MCTS READY FOR TRADING!")