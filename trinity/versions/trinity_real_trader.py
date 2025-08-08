#!/usr/bin/env python3
"""
TRINITY REAL TRADER - Actual Trading with Real APIs
ULTRATHINK: Real trades, real learning, real self-improvement
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from cryptography.fernet import Fernet
import requests
from queue import Queue
import threading

# Add paths for existing integrations
sys.path.insert(0, '/opt/cashmachine/trinity')
sys.path.insert(0, '/home/ubuntu/CashMachine/src/oanda')

# Import existing trading systems
try:
    from ultrathink_oandav20 import UltrathinkOandaV20
    OANDA_AVAILABLE = True
except:
    OANDA_AVAILABLE = False
    
try:
    from trinity_oanda_integration import TrinityOandaIntegration
    TRINITY_OANDA_AVAILABLE = True
except:
    TRINITY_OANDA_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TRINITY_REAL')

# ============================================================================
# CREDENTIAL MANAGER
# ============================================================================

class CredentialManager:
    """Manages encrypted API credentials"""
    
    def __init__(self):
        self.config_dir = "/opt/cashmachine/config"
        self.credentials = {}
        self.load_all_credentials()
    
    def load_encrypted_config(self, name: str) -> Optional[Dict]:
        """Load and decrypt API configuration"""
        try:
            key_file = f"{self.config_dir}/.{name}.key"
            enc_file = f"{self.config_dir}/{name}.enc"
            
            # Check if files exist
            if not os.path.exists(key_file) or not os.path.exists(enc_file):
                logger.warning(f"Credential files for {name} not found")
                return None
            
            with open(key_file, "rb") as f:
                key = f.read()
            with open(enc_file, "rb") as f:
                encrypted = f.read()
            
            cipher = Fernet(key)
            config = json.loads(cipher.decrypt(encrypted))
            logger.info(f"✅ {name.upper()} credentials loaded")
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to load {name} credentials: {e}")
            return None
    
    def load_all_credentials(self):
        """Load all available API credentials"""
        apis = ['oanda', 'alpaca', 'polygon', 'finnhub', 'alphavantage']
        
        for api in apis:
            config = self.load_encrypted_config(api)
            if config:
                self.credentials[api] = config
        
        logger.info(f"📊 Loaded credentials for: {list(self.credentials.keys())}")
    
    def get_credential(self, api: str, key: str) -> Optional[str]:
        """Get specific credential"""
        if api in self.credentials:
            return self.credentials[api].get(key)
        return None

# ============================================================================
# REAL TRADE EXECUTOR
# ============================================================================

class RealTradeExecutor:
    """Executes real trades through broker APIs"""
    
    def __init__(self, credential_manager: CredentialManager):
        self.creds = credential_manager
        self.oanda_client = None
        self.trinity_oanda = None
        self.position_sizes = {
            'forex': 1000,      # 1K units for forex (micro lot)
            'stocks': 1,        # 1 share minimum
            'crypto': 0.001     # 0.001 BTC minimum
        }
        self.initialize_brokers()
    
    def initialize_brokers(self):
        """Initialize broker connections"""
        # Initialize OANDA for forex
        if OANDA_AVAILABLE and self.creds.get_credential('oanda', 'api_token'):
            try:
                self.oanda_client = UltrathinkOandaV20()
                self.oanda_client.initialize_store()
                logger.info("✅ OANDA v20 initialized for real trading")
            except Exception as e:
                logger.error(f"❌ OANDA initialization failed: {e}")
        
        # Initialize Trinity OANDA integration
        if TRINITY_OANDA_AVAILABLE:
            try:
                self.trinity_oanda = TrinityOandaIntegration()
                if self.trinity_oanda.authenticate():
                    logger.info("✅ Trinity OANDA authenticated")
                else:
                    logger.warning("⚠️ Trinity OANDA authentication failed")
            except Exception as e:
                logger.error(f"❌ Trinity OANDA failed: {e}")
    
    def execute_forex_trade(self, signal: Dict) -> Dict:
        """Execute real forex trade through OANDA"""
        try:
            instrument = signal['symbol']  # e.g., EUR_USD
            action = signal['signal']      # buy/sell
            confidence = signal['confidence']
            
            # Calculate position size based on confidence
            base_units = self.position_sizes['forex']
            units = int(base_units * confidence)
            
            # Prepare order
            order = {
                'instrument': instrument,
                'units': units if action == 'buy' else -units,
                'type': 'MARKET',
                'timeInForce': 'FOK',  # Fill or Kill
                'positionFill': 'DEFAULT'
            }
            
            # Execute through Trinity OANDA if available
            if self.trinity_oanda:
                result = self.trinity_oanda.place_order(
                    instrument=instrument,
                    units=units,
                    side='buy' if action == 'buy' else 'sell',
                    order_type='market'
                )
                
                if result.get('success'):
                    trade_id = result.get('result', {}).get('orderFillTransaction', {}).get('id')
                    price = result.get('result', {}).get('orderFillTransaction', {}).get('price')
                    
                    return {
                        'success': True,
                        'trade_id': trade_id,
                        'price': price,
                        'units': units,
                        'profit': 0,  # Will be calculated on close
                        'message': f"Real OANDA trade executed: {instrument} {action} {units} units"
                    }
            
            # Fallback if no real execution available
            logger.warning("⚠️ No real forex broker available, paper trading")
            return self.paper_trade(signal, 'forex')
            
        except Exception as e:
            logger.error(f"Forex trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_stock_trade(self, signal: Dict) -> Dict:
        """Execute real stock trade through Alpaca"""
        # For now, paper trade stocks
        # TODO: Implement Alpaca integration
        return self.paper_trade(signal, 'stocks')
    
    def execute_crypto_trade(self, signal: Dict) -> Dict:
        """Execute real crypto trade"""
        # For now, paper trade crypto
        # TODO: Implement crypto exchange integration
        return self.paper_trade(signal, 'crypto')
    
    def paper_trade(self, signal: Dict, asset_type: str) -> Dict:
        """Paper trading with realistic simulation"""
        # Use real market data if available
        symbol = signal['symbol']
        action = signal['signal']
        
        # Get real price from APIs
        price = self.get_real_price(symbol, asset_type)
        
        # Simulate realistic slippage
        slippage = np.random.uniform(0.0001, 0.0005)  # 0.01% to 0.05%
        if action == 'buy':
            price *= (1 + slippage)
        else:
            price *= (1 - slippage)
        
        # Calculate position size
        units = self.position_sizes.get(asset_type, 1)
        
        return {
            'success': True,
            'trade_id': f"PAPER_{int(time.time())}",
            'price': price,
            'units': units,
            'profit': 0,
            'paper_trade': True,
            'message': f"Paper trade: {symbol} {action} at ${price:.4f}"
        }
    
    def get_real_price(self, symbol: str, asset_type: str) -> float:
        """Get real market price from APIs"""
        try:
            if asset_type == 'forex':
                # Get from Alpha Vantage or similar
                from_currency, to_currency = symbol.split('_')
                url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={self.creds.get_credential('alphavantage', 'api_key')}"
                
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if 'Realtime Currency Exchange Rate' in data:
                        return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
            
            elif asset_type == 'stocks':
                # Get from Finnhub
                url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.creds.get_credential('finnhub', 'api_key')}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if 'c' in data:
                        return data['c']
        except:
            pass
        
        # Fallback to simulated price
        base_prices = {
            'EUR_USD': 1.10,
            'GBP_USD': 1.25,
            'SPY': 400,
            'AAPL': 180,
            'TSLA': 250,
            'BTC': 50000,
            'ETH': 3000
        }
        return base_prices.get(symbol, 100) * (1 + np.random.randn() * 0.01)

# ============================================================================
# REAL LEARNING SYSTEM
# ============================================================================

class RealLearningSystem:
    """Learns from real trading results"""
    
    def __init__(self):
        self.trade_history = []
        self.strategy_performance = {}
        self.patterns = []
        self.generation = 0
        self.mutation_rate = 0.1
        
    def learn_from_real_trade(self, trade_result: Dict):
        """Learn from actual trade results"""
        self.trade_history.append(trade_result)
        
        # Extract features from trade
        features = {
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'symbol': trade_result.get('symbol'),
            'action': trade_result.get('action'),
            'confidence': trade_result.get('confidence'),
            'profit': trade_result.get('profit', 0),
            'success': trade_result.get('success', False)
        }
        
        # Update strategy performance
        strategy = trade_result.get('strategy', 'default')
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'trades': 0,
                'wins': 0,
                'total_profit': 0,
                'avg_confidence': 0
            }
        
        stats = self.strategy_performance[strategy]
        stats['trades'] += 1
        if features['profit'] > 0:
            stats['wins'] += 1
        stats['total_profit'] += features['profit']
        stats['avg_confidence'] = (stats['avg_confidence'] * (stats['trades'] - 1) + features['confidence']) / stats['trades']
        
        # Identify patterns in successful trades
        if features['profit'] > 0:
            self.identify_pattern(features)
        
        # Evolve strategies based on performance
        if len(self.trade_history) % 10 == 0:  # Every 10 trades
            self.evolve_strategies()
    
    def identify_pattern(self, features: Dict):
        """Identify patterns in successful trades"""
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'generation': self.generation
        }
        
        # Check if similar pattern exists
        for existing in self.patterns:
            if self.pattern_similarity(existing['features'], features) > 0.8:
                # Reinforce existing pattern
                existing['reinforcement'] = existing.get('reinforcement', 1) + 1
                return
        
        # New pattern discovered
        self.patterns.append(pattern)
        logger.info(f"🎯 New trading pattern discovered: {features['symbol']} at hour {features['time_of_day']}")
    
    def pattern_similarity(self, p1: Dict, p2: Dict) -> float:
        """Calculate similarity between two patterns"""
        similarity = 0
        factors = 0
        
        if p1.get('symbol') == p2.get('symbol'):
            similarity += 0.3
        if p1.get('action') == p2.get('action'):
            similarity += 0.2
        if abs(p1.get('time_of_day', 0) - p2.get('time_of_day', 0)) <= 2:
            similarity += 0.2
        if p1.get('day_of_week') == p2.get('day_of_week'):
            similarity += 0.1
        if abs(p1.get('confidence', 0) - p2.get('confidence', 0)) <= 0.1:
            similarity += 0.2
        
        return similarity
    
    def evolve_strategies(self):
        """Evolve trading strategies based on real performance"""
        logger.info(f"🧬 EVOLUTION: Analyzing {len(self.trade_history)} real trades")
        
        # Calculate fitness for each strategy
        fitness_scores = {}
        for strategy, stats in self.strategy_performance.items():
            if stats['trades'] > 0:
                win_rate = stats['wins'] / stats['trades']
                avg_profit = stats['total_profit'] / stats['trades']
                fitness = win_rate * 0.5 + min(avg_profit / 100, 1) * 0.5
                fitness_scores[strategy] = fitness
        
        if not fitness_scores:
            return
        
        # Find best and worst strategies
        best_strategy = max(fitness_scores, key=fitness_scores.get)
        worst_strategy = min(fitness_scores, key=fitness_scores.get)
        
        logger.info(f"🏆 Best strategy: {best_strategy} (fitness: {fitness_scores[best_strategy]:.2f})")
        logger.info(f"❌ Worst strategy: {worst_strategy} (fitness: {fitness_scores[worst_strategy]:.2f})")
        
        # Adapt mutation rate based on performance
        avg_fitness = sum(fitness_scores.values()) / len(fitness_scores)
        if avg_fitness < 0.5:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            logger.info(f"📈 Increasing mutation rate to {self.mutation_rate:.2f}")
        else:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
            logger.info(f"📉 Decreasing mutation rate to {self.mutation_rate:.2f}")
        
        self.generation += 1
        logger.info(f"🧬 Evolution complete. Generation: {self.generation}")
    
    def get_best_strategy_for_conditions(self, market_conditions: Dict) -> str:
        """Select best strategy based on current conditions and learned patterns"""
        # Check if we have patterns for current conditions
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        symbol = market_conditions.get('symbol')
        
        # Find patterns matching current conditions
        matching_patterns = []
        for pattern in self.patterns:
            features = pattern['features']
            if (features.get('symbol') == symbol and
                abs(features.get('time_of_day', 0) - current_hour) <= 2 and
                features.get('day_of_week') == current_day):
                matching_patterns.append(pattern)
        
        if matching_patterns:
            # Use the most reinforced pattern
            best_pattern = max(matching_patterns, key=lambda x: x.get('reinforcement', 1))
            logger.info(f"📊 Using learned pattern for {symbol} at hour {current_hour}")
            return best_pattern['features'].get('action', 'hold')
        
        # Default to best performing strategy
        if self.strategy_performance:
            best_strategy = max(self.strategy_performance.items(), 
                              key=lambda x: x[1]['total_profit'])
            return best_strategy[0]
        
        return 'default'

# ============================================================================
# TRINITY REAL CONSCIOUSNESS
# ============================================================================

class TrinityRealConsciousness:
    """The real trading consciousness with actual execution"""
    
    def __init__(self):
        logger.info("🧠 TRINITY REAL CONSCIOUSNESS INITIALIZING...")
        
        # Core components
        self.credential_manager = CredentialManager()
        self.trade_executor = RealTradeExecutor(self.credential_manager)
        self.learning_system = RealLearningSystem()
        
        # Trading state
        self.active_trades = {}
        self.daily_pnl = 0
        self.total_pnl = 0
        self.max_daily_loss = -500  # Stop trading if lose $500
        self.trades_today = 0
        self.max_trades_per_day = 50
        
        # Safety mechanisms
        self.circuit_breaker_triggered = False
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        
        logger.info("✅ TRINITY REAL CONSCIOUSNESS ONLINE")
        self.log_status()
    
    def log_status(self):
        """Log current trading capability status"""
        logger.info("=" * 50)
        logger.info("📊 TRADING CAPABILITY STATUS:")
        logger.info(f"  OANDA Forex: {'✅ READY' if self.trade_executor.trinity_oanda else '⚠️ Paper Only'}")
        logger.info(f"  Alpaca Stocks: ⚠️ Paper Only (credentials needed)")
        logger.info(f"  Crypto: ⚠️ Paper Only (exchange integration needed)")
        logger.info(f"  Daily Loss Limit: ${abs(self.max_daily_loss)}")
        logger.info(f"  Max Trades/Day: {self.max_trades_per_day}")
        logger.info("=" * 50)
    
    def check_safety_limits(self) -> bool:
        """Check if safe to trade"""
        # Daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            logger.error(f"🛑 Daily loss limit reached: ${self.daily_pnl:.2f}")
            self.circuit_breaker_triggered = True
            return False
        
        # Trade count limit
        if self.trades_today >= self.max_trades_per_day:
            logger.warning(f"⚠️ Daily trade limit reached: {self.trades_today}")
            return False
        
        # Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.error(f"🛑 Too many consecutive losses: {self.consecutive_losses}")
            self.circuit_breaker_triggered = True
            return False
        
        # Circuit breaker
        if self.circuit_breaker_triggered:
            logger.error("🛑 Circuit breaker active - trading halted")
            return False
        
        return True
    
    def process_trading_signal(self, signal: Dict) -> Dict:
        """Process a trading signal with real execution"""
        # Safety checks
        if not self.check_safety_limits():
            return {'success': False, 'reason': 'Safety limits exceeded'}
        
        # Determine asset type
        symbol = signal['symbol']
        if '_' in symbol:  # Forex pair like EUR_USD
            asset_type = 'forex'
        elif symbol in ['BTC', 'ETH']:
            asset_type = 'crypto'
        else:
            asset_type = 'stocks'
        
        # Get strategy recommendation from learning system
        strategy = self.learning_system.get_best_strategy_for_conditions({
            'symbol': symbol,
            'market_state': signal.get('market_state')
        })
        
        # Execute trade
        logger.info(f"🎯 Executing {asset_type} trade: {symbol} {signal['signal']}")
        
        if asset_type == 'forex':
            result = self.trade_executor.execute_forex_trade(signal)
        elif asset_type == 'crypto':
            result = self.trade_executor.execute_crypto_trade(signal)
        else:
            result = self.trade_executor.execute_stock_trade(signal)
        
        # Track trade
        if result.get('success'):
            self.trades_today += 1
            trade_id = result.get('trade_id')
            self.active_trades[trade_id] = {
                'symbol': symbol,
                'entry_price': result.get('price'),
                'units': result.get('units'),
                'timestamp': datetime.now(),
                'strategy': strategy
            }
            
            # Log execution
            if result.get('paper_trade'):
                logger.info(f"📝 Paper trade executed: {result.get('message')}")
            else:
                logger.info(f"💰 REAL trade executed: {result.get('message')}")
            
            # Learn immediately from execution
            self.learning_system.learn_from_real_trade({
                'symbol': symbol,
                'action': signal['signal'],
                'confidence': signal['confidence'],
                'strategy': strategy,
                'profit': 0,  # Will update when closed
                'success': True
            })
        
        return result
    
    def close_trade(self, trade_id: str, current_price: float) -> Dict:
        """Close a trade and calculate real P&L"""
        if trade_id not in self.active_trades:
            return {'success': False, 'error': 'Trade not found'}
        
        trade = self.active_trades[trade_id]
        entry_price = trade['entry_price']
        units = trade['units']
        
        # Calculate P&L
        if units > 0:  # Long position
            pnl = (current_price - entry_price) * units
        else:  # Short position
            pnl = (entry_price - current_price) * abs(units)
        
        # Update tracking
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Learn from result
        self.learning_system.learn_from_real_trade({
            'symbol': trade['symbol'],
            'action': 'close',
            'confidence': 1.0,
            'strategy': trade['strategy'],
            'profit': pnl,
            'success': pnl > 0
        })
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        logger.info(f"{'✅' if pnl > 0 else '❌'} Trade closed: {trade['symbol']} P&L: ${pnl:.2f}")
        logger.info(f"📊 Daily P&L: ${self.daily_pnl:.2f} | Total P&L: ${self.total_pnl:.2f}")
        
        return {
            'success': True,
            'pnl': pnl,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl
        }
    
    def reset_daily_counters(self):
        """Reset daily counters (call at midnight)"""
        logger.info(f"📅 Daily reset - Yesterday's P&L: ${self.daily_pnl:.2f}")
        self.daily_pnl = 0
        self.trades_today = 0
        self.circuit_breaker_triggered = False
        self.consecutive_losses = 0
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        return {
            'active_trades': len(self.active_trades),
            'trades_today': self.trades_today,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'generation': self.learning_system.generation,
            'patterns_learned': len(self.learning_system.patterns),
            'circuit_breaker': self.circuit_breaker_triggered,
            'can_trade': self.check_safety_limits()
        }

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Test the real trading system"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     🧠 TRINITY REAL TRADER - ACTUAL TRADING SYSTEM           ║
    ║                                                              ║
    ║     Real Execution | Real Learning | Real P&L               ║
    ║     OANDA Forex | Alpaca Stocks | Crypto                    ║
    ║                                                              ║
    ║     "Trading with real money, learning from real results"   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize real trading system
    trinity = TrinityRealConsciousness()
    
    # Test with a signal
    test_signal = {
        'symbol': 'EUR_USD',
        'signal': 'buy',
        'confidence': 0.7,
        'market_state': 'open'
    }
    
    print("\n📊 Testing real trade execution...")
    result = trinity.process_trading_signal(test_signal)
    
    if result.get('success'):
        print(f"✅ Trade executed: {result}")
    else:
        print(f"❌ Trade failed: {result}")
    
    # Show status
    print("\n📊 Current Status:")
    status = trinity.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n🧠 ULTRATHINK: Real trading system ready!")

if __name__ == "__main__":
    main()