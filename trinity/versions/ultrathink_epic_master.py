#!/usr/bin/env python3
"""
ULTRATHINK EPIC MASTER - The Ultimate AI Trading System
Combines HRM + ASI + AlphaGo for REAL intelligent trading
This is the REAL DEAL - All 3 AIs working together!
"""

import sys
import os
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import requests
import redis
from collections import deque
import threading

# Import our AI modules
sys.path.append('/tmp')
from ultrathink_hrm_trading import HRMTradingSystem
from ultrathink_asi_genetic import ASIGeneticEvolution
from ultrathink_alphago_mcts import AlphaGoMCTS, TradingState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ULTRATHINK_MASTER')

class UltrathinkEpicMaster:
    """The master orchestrator combining all 3 AI systems"""
    
    def __init__(self):
        logger.info("=" * 70)
        logger.info("🚀 ULTRATHINK EPIC MASTER SYSTEM INITIALIZING")
        logger.info("=" * 70)
        
        # Initialize all AI systems
        logger.info("\n⚡ Loading AI Components...")
        
        # 1. HRM - Hierarchical Reasoning Model
        logger.info("1️⃣ Initializing HRM (27M parameters)...")
        self.hrm = HRMTradingSystem()
        
        # 2. ASI - Genetic Evolution
        logger.info("2️⃣ Initializing ASI Genetic Evolution...")
        self.asi = ASIGeneticEvolution(population_size=50)
        
        # 3. AlphaGo MCTS
        logger.info("3️⃣ Initializing AlphaGo MCTS...")
        self.mcts = AlphaGoMCTS(simulation_depth=10, num_simulations=50)
        
        # Weights for combining signals
        self.hrm_weight = 0.40  # 40% weight for neural network
        self.asi_weight = 0.30  # 30% weight for genetic algorithm
        self.mcts_weight = 0.30  # 30% weight for tree search
        
        # Trading state
        self.current_position = 'none'
        self.entry_price = None
        self.holding_period = 0
        self.total_pnl = 0.0
        
        # Price history for analysis
        self.price_history = {}
        self.max_history = 100
        
        # Performance tracking
        self.decisions = []
        self.trades = []
        
        # Alpaca API credentials
        self.api_key = 'PKGXVRHYGL3DT8QQ795W'
        self.api_secret = 'uaqqu1ifB2eCh0IE0M1nMRTPjVceeJ3efOvuB1wm'
        self.base_url = 'https://paper-api.alpaca.markets'
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        # Redis connection for coordination
        try:
            self.redis_client = redis.Redis(host='10.100.2.200', port=6379, decode_responses=True)
            logger.info("✅ Connected to Redis message bus")
        except:
            self.redis_client = None
            logger.warning("⚠️ Redis not available, running standalone")
        
        logger.info("\n✅ ALL AI SYSTEMS LOADED AND READY!")
        logger.info("=" * 70)
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data from Alpaca"""
        try:
            # Get latest bar
            url = f"{self.base_url}/v2/stocks/{symbol}/bars/latest"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                bar = data.get('bar', {})
                return {
                    'symbol': symbol,
                    'price': bar.get('c', 0),
                    'volume': bar.get('v', 0),
                    'timestamp': bar.get('t', ''),
                    'high': bar.get('h', 0),
                    'low': bar.get('l', 0),
                    'open': bar.get('o', 0)
                }
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
        
        return {'symbol': symbol, 'price': 100, 'volume': 10000}
    
    def update_price_history(self, symbol: str, price: float):
        """Update price history for analysis"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.max_history)
        self.price_history[symbol].append(price)
    
    def analyze_with_all_ais(self, symbol: str) -> Dict:
        """Run analysis with all 3 AI systems and combine results"""
        logger.info(f"\n🧠 RUNNING FULL AI ANALYSIS FOR {symbol}")
        logger.info("-" * 50)
        
        # Get current market data
        market_data = self.get_market_data(symbol)
        current_price = market_data['price']
        
        # Update price history
        self.update_price_history(symbol, current_price)
        
        # Prepare data for each AI
        price_data = {
            'price': current_price,
            'volume': market_data['volume']
        }
        
        # 1. HRM Analysis
        logger.info("🔮 HRM Neural Network analyzing...")
        hrm_result = self.hrm.analyze(symbol, price_data)
        logger.info(f"   HRM Signal: {hrm_result['signal']} (conf: {hrm_result['confidence']:.2%})")
        
        # 2. ASI Genetic Analysis
        logger.info("🧬 ASI Genetic Evolution analyzing...")
        if symbol in self.price_history:
            asi_data = {'prices': list(self.price_history[symbol])}
            asi_result = self.asi.analyze(asi_data)
        else:
            asi_result = {'signal': 'hold', 'confidence': 0.0}
        logger.info(f"   ASI Signal: {asi_result['signal']} (conf: {asi_result['confidence']:.2%})")
        
        # 3. AlphaGo MCTS Analysis
        logger.info("🎯 AlphaGo MCTS analyzing...")
        mcts_data = {
            'current_price': current_price,
            'position': self.current_position,
            'holding_period': self.holding_period,
            'entry_price': self.entry_price,
            'pnl': self.total_pnl
        }
        mcts_result = self.mcts.analyze(mcts_data)
        logger.info(f"   MCTS Signal: {mcts_result['signal']} (conf: {mcts_result['confidence']:.2%})")
        
        # Combine signals with weighted voting
        signal_scores = {
            'buy': 0.0,
            'sell': 0.0,
            'hold': 0.0
        }
        
        # HRM contribution
        hrm_signal = hrm_result['signal']
        hrm_conf = hrm_result['confidence']
        signal_scores[hrm_signal] += self.hrm_weight * hrm_conf
        
        # ASI contribution
        asi_signal = asi_result['signal']
        asi_conf = asi_result['confidence']
        signal_scores[asi_signal] += self.asi_weight * asi_conf
        
        # MCTS contribution
        mcts_signal = mcts_result['signal']
        mcts_conf = mcts_result['confidence']
        signal_scores[mcts_signal] += self.mcts_weight * mcts_conf
        
        # Determine final signal
        final_signal = max(signal_scores.items(), key=lambda x: x[1])[0]
        final_confidence = signal_scores[final_signal]
        
        # Calculate consensus level
        signals = [hrm_signal, asi_signal, mcts_signal]
        consensus = signals.count(final_signal) / 3.0
        
        logger.info("\n🎯 FINAL AI CONSENSUS:")
        logger.info(f"   Signal: {final_signal.upper()}")
        logger.info(f"   Confidence: {final_confidence:.2%}")
        logger.info(f"   Consensus: {consensus:.1%} ({signals.count(final_signal)}/3 AIs agree)")
        logger.info("-" * 50)
        
        return {
            'symbol': symbol,
            'final_signal': final_signal,
            'confidence': final_confidence,
            'consensus': consensus,
            'price': current_price,
            'hrm': hrm_result,
            'asi': asi_result,
            'mcts': mcts_result,
            'signal_scores': signal_scores,
            'timestamp': datetime.now().isoformat()
        }
    
    def execute_trade(self, signal: str, symbol: str, confidence: float) -> Dict:
        """Execute trade on Alpaca"""
        logger.info(f"\n💰 EXECUTING TRADE: {signal.upper()} {symbol}")
        
        # Only trade if confidence is high enough
        if confidence < 0.5:
            logger.info(f"   ⚠️ Confidence too low ({confidence:.2%}), skipping trade")
            return {'status': 'skipped', 'reason': 'low_confidence'}
        
        try:
            # Check if market is open
            clock_url = f"{self.base_url}/v2/clock"
            clock_resp = requests.get(clock_url, headers=self.headers)
            clock = clock_resp.json()
            
            if not clock.get('is_open', False):
                logger.info("   ⏰ Market closed, queuing order for next open")
                time_in_force = 'opg'  # At market open
            else:
                time_in_force = 'day'
            
            # Determine order parameters
            qty = max(1, int(confidence * 5))  # Scale quantity with confidence
            
            if signal == 'buy':
                if self.current_position == 'short':
                    # Close short first
                    close_order = {
                        'symbol': symbol,
                        'qty': qty,
                        'side': 'buy',
                        'type': 'market',
                        'time_in_force': time_in_force
                    }
                    requests.post(f"{self.base_url}/v2/orders", 
                                headers=self.headers, json=close_order)
                
                # Open long
                order = {
                    'symbol': symbol,
                    'qty': qty,
                    'side': 'buy',
                    'type': 'market',
                    'time_in_force': time_in_force
                }
                
            elif signal == 'sell':
                if self.current_position == 'long':
                    # Close long first
                    close_order = {
                        'symbol': symbol,
                        'qty': qty,
                        'side': 'sell',
                        'type': 'market',
                        'time_in_force': time_in_force
                    }
                    requests.post(f"{self.base_url}/v2/orders",
                                headers=self.headers, json=close_order)
                
                # Open short
                order = {
                    'symbol': symbol,
                    'qty': qty,
                    'side': 'sell',
                    'type': 'market',
                    'time_in_force': time_in_force
                }
            
            else:  # hold
                logger.info("   📊 Holding position")
                return {'status': 'hold', 'position': self.current_position}
            
            # Submit order
            response = requests.post(f"{self.base_url}/v2/orders",
                                    headers=self.headers, json=order)
            
            if response.status_code in [200, 201]:
                order_data = response.json()
                logger.info(f"   ✅ Order placed! ID: {order_data.get('id', 'N/A')}")
                logger.info(f"   Qty: {qty}, Status: {order_data.get('status', 'N/A')}")
                
                # Update position
                if signal == 'buy':
                    self.current_position = 'long'
                elif signal == 'sell':
                    self.current_position = 'short'
                
                # Store trade
                self.trades.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'signal': signal,
                    'qty': qty,
                    'confidence': confidence,
                    'order_id': order_data.get('id')
                })
                
                # Publish to Redis if available
                if self.redis_client:
                    self.redis_client.lpush('ultrathink:epic:trades', 
                                          json.dumps(self.trades[-1]))
                
                return {'status': 'success', 'order': order_data}
            
            else:
                logger.error(f"   ❌ Order failed: {response.text}")
                return {'status': 'failed', 'error': response.text}
                
        except Exception as e:
            logger.error(f"   ❌ Trade execution error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def trading_loop(self, symbols: List[str] = None):
        """Main trading loop"""
        if symbols is None:
            symbols = ['SPY', 'AAPL', 'TSLA', 'GOOGL']
        
        logger.info("\n🚀 STARTING ULTRATHINK EPIC TRADING LOOP")
        logger.info(f"   Trading symbols: {', '.join(symbols)}")
        
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"📈 TRADING ITERATION #{iteration}")
            logger.info(f"{'='*70}")
            
            for symbol in symbols:
                # Run full AI analysis
                analysis = self.analyze_with_all_ais(symbol)
                
                # Store decision
                self.decisions.append(analysis)
                
                # Execute trade if consensus is strong
                if analysis['consensus'] >= 0.66:  # At least 2/3 AIs agree
                    result = self.execute_trade(
                        analysis['final_signal'],
                        symbol,
                        analysis['confidence']
                    )
                    logger.info(f"   Trade result: {result['status']}")
                else:
                    logger.info(f"   ⚠️ Low consensus ({analysis['consensus']:.1%}), no trade")
            
            # Wait before next iteration
            logger.info(f"\n⏳ Waiting 60 seconds for next analysis...")
            await asyncio.sleep(60)
    
    def get_account_status(self) -> Dict:
        """Get current account status from Alpaca"""
        try:
            response = requests.get(f"{self.base_url}/v2/account", 
                                  headers=self.headers)
            if response.status_code == 200:
                account = response.json()
                return {
                    'cash': float(account.get('cash', 0)),
                    'buying_power': float(account.get('buying_power', 0)),
                    'equity': float(account.get('equity', 0)),
                    'positions': account.get('position_market_value', 0)
                }
        except:
            pass
        return {}

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ULTRATHINK EPIC MASTER SYSTEM")
    print("🧠 HRM + 🧬 ASI + 🎯 AlphaGo = ULTIMATE AI TRADING")
    print("=" * 70)
    
    # Initialize the master system
    master = UltrathinkEpicMaster()
    
    # Get account status
    account = master.get_account_status()
    if account:
        print(f"\n💰 Account Status:")
        print(f"   Cash: ${account.get('cash', 0):,.2f}")
        print(f"   Equity: ${account.get('equity', 0):,.2f}")
    
    # Run a test analysis
    print("\n🔬 Running test analysis on SPY...")
    test_result = master.analyze_with_all_ais('SPY')
    
    print(f"\n✅ SYSTEM READY!")
    print(f"   Final Signal: {test_result['final_signal'].upper()}")
    print(f"   Confidence: {test_result['confidence']:.2%}")
    print(f"   Consensus: {test_result['consensus']:.1%}")
    
    # Start trading loop
    print("\n🎯 Starting automated trading loop...")
    print("   Press Ctrl+C to stop")
    
    try:
        asyncio.run(master.trading_loop(['SPY', 'AAPL', 'TSLA']))
    except KeyboardInterrupt:
        print("\n\n⛔ Trading stopped by user")
        print(f"   Total decisions: {len(master.decisions)}")
        print(f"   Total trades: {len(master.trades)}")
        print("\n✅ ULTRATHINK EPIC MASTER SHUTDOWN COMPLETE")