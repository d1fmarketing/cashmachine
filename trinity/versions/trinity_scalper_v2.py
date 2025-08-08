#\!/usr/bin/env python3
"""
TRINITY SCALPER v2.0 - ULTRATHINK PREMIUM EDITION
Real-time scalping with premium APIs and paper trading execution
Zero humans, infinite intelligence, maximum profits
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
from collections import deque
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/trinity-scalper-v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TRINITY_V2')

# ============================================================================
# CREDENTIAL MANAGER WITH PREMIUM API
# ============================================================================

class PremiumCredentialManager:
    """Manages API credentials with premium Alpha Vantage"""
    
    def __init__(self):
        self.config_dir = "/opt/cashmachine/config"
        self.credentials = {}
        self.load_all_credentials()
        logger.info(f"✅ Loaded APIs: {list(self.credentials.keys())}")
    
    def load_all_credentials(self):
        """Load all API credentials"""
        try:
            from cryptography.fernet import Fernet
            
            apis = ['alphavantage', 'finnhub', 'polygon', 'oanda', 'alpaca']
            for api in apis:
                try:
                    key_file = f"{self.config_dir}/.{api}.key"
                    enc_file = f"{self.config_dir}/{api}.enc"
                    
                    if os.path.exists(key_file) and os.path.exists(enc_file):
                        with open(key_file, "rb") as f:
                            key = f.read()
                        with open(enc_file, "rb") as f:
                            encrypted = f.read()
                        
                        cipher = Fernet(key)
                        config = json.loads(cipher.decrypt(encrypted))
                        self.credentials[api] = config
                        
                        if api == 'alphavantage':
                            logger.info(f"💎 PREMIUM Alpha Vantage: {config.get('api_key')}")
                except Exception as e:
                    logger.warning(f"Could not load {api}: {e}")
        except ImportError:
            logger.error("Cryptography module required\!")
    
    def get(self, api: str, key: str, default=None):
        """Get credential value"""
        if api in self.credentials:
            return self.credentials[api].get(key, default)
        return default

# ============================================================================
# REAL BROKER CONNECTIONS
# ============================================================================

class BrokerConnection:
    """Manages real broker connections for paper trading"""
    
    def __init__(self, creds: PremiumCredentialManager):
        self.creds = creds
        self.oanda_connected = False
        self.alpaca_connected = False
        self.positions = {}
        self.connect_brokers()
    
    def connect_brokers(self):
        """Connect to OANDA and Alpaca"""
        # OANDA Connection
        oanda_token = self.creds.get('oanda', 'api_token')
        oanda_account = self.creds.get('oanda', 'account_id')
        if oanda_token and oanda_account:
            self.oanda_url = "https://api-fxpractice.oanda.com/v3"
            self.oanda_headers = {"Authorization": f"Bearer {oanda_token}"}
            self.oanda_account = oanda_account
            self.oanda_connected = True
            logger.info(f"✅ OANDA connected: {oanda_account}")
        
        # Alpaca Connection
        alpaca_key = self.creds.get('alpaca', 'api_key')
        alpaca_secret = self.creds.get('alpaca', 'api_secret')
        if alpaca_key and alpaca_secret:
            self.alpaca_url = "https://paper-api.alpaca.markets/v2"
            self.alpaca_headers = {
                "APCA-API-KEY-ID": alpaca_key,
                "APCA-API-SECRET-KEY": alpaca_secret
            }
            self.alpaca_connected = True
            logger.info(f"✅ Alpaca connected")
    
    async def execute_forex_trade(self, symbol: str, units: int, side: str) -> Dict:
        """Execute forex trade on OANDA"""
        if not self.oanda_connected:
            return {"success": False, "error": "OANDA not connected"}
        
        try:
            import aiohttp
            
            order = {
                "order": {
                    "instrument": symbol.replace('_', '_'),
                    "units": str(units if side == 'buy' else -units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
                }
            }
            
            url = f"{self.oanda_url}/accounts/{self.oanda_account}/orders"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=order, headers=self.oanda_headers) as resp:
                    if resp.status == 201:
                        data = await resp.json()
                        return {
                            "success": True,
                            "order_id": data.get("orderFillTransaction", {}).get("id"),
                            "price": float(data.get("orderFillTransaction", {}).get("price", 0))
                        }
                    else:
                        return {"success": False, "error": f"Status {resp.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_stock_trade(self, symbol: str, qty: int, side: str) -> Dict:
        """Execute stock trade on Alpaca"""
        if not self.alpaca_connected:
            return {"success": False, "error": "Alpaca not connected"}
        
        try:
            import aiohttp
            
            order = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": "market",
                "time_in_force": "ioc"  # Immediate or cancel for scalping
            }
            
            url = f"{self.alpaca_url}/orders"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=order, headers=self.alpaca_headers) as resp:
                    if resp.status in [200, 201]:
                        data = await resp.json()
                        return {
                            "success": True,
                            "order_id": data.get("id"),
                            "price": float(data.get("filled_avg_price", 0))
                        }
                    else:
                        return {"success": False, "error": f"Status {resp.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# ============================================================================
# PREMIUM API DATA MANAGER
# ============================================================================

class PremiumDataManager:
    """Manages premium API calls with intelligent rate limiting"""
    
    def __init__(self, creds: PremiumCredentialManager):
        self.creds = creds
        self.alpha_vantage_key = creds.get('alphavantage', 'api_key')
        self.finnhub_key = creds.get('finnhub', 'api_key')
        
        # Rate limiting: 75 calls/minute for Alpha Vantage
        self.av_calls = deque(maxlen=75)  # Track last 75 calls
        self.av_minute_start = time.time()
        
        # Price cache
        self.price_cache = {}
        self.cache_ttl = 1  # 1 second cache
        
        logger.info(f"💎 Premium Data Manager initialized")
        logger.info(f"   Alpha Vantage: 75 calls/minute available")
    
    async def get_forex_price(self, symbol: str) -> Optional[Dict]:
        """Get real-time forex price from premium Alpha Vantage"""
        # Check cache first
        cache_key = f"forex_{symbol}"
        if cache_key in self.price_cache:
            cached = self.price_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['data']
        
        # Rate limit check
        now = time.time()
        if now - self.av_minute_start > 60:
            self.av_calls.clear()
            self.av_minute_start = now
        
        if len(self.av_calls) >= 75:
            # Wait until next minute
            wait_time = 60 - (now - self.av_minute_start)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.av_calls.clear()
                self.av_minute_start = time.time()
        
        # Make API call
        try:
            import aiohttp
            from_curr, to_curr = symbol.split('_')
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_curr,
                "to_currency": to_curr,
                "apikey": self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "Realtime Currency Exchange Rate" in data:
                            rate_data = data["Realtime Currency Exchange Rate"]
                            result = {
                                "price": float(rate_data.get("5. Exchange Rate", 0)),
                                "bid": float(rate_data.get("8. Bid Price", 0)),
                                "ask": float(rate_data.get("9. Ask Price", 0)),
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # Cache result
                            self.price_cache[cache_key] = {
                                'data': result,
                                'timestamp': time.time()
                            }
                            
                            # Track API call
                            self.av_calls.append(time.time())
                            
                            return result
        except Exception as e:
            logger.error(f"Forex API error: {e}")
        
        return None
    
    async def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """Get real-time stock price from Finnhub"""
        try:
            import aiohttp
            url = f"https://finnhub.io/api/v1/quote"
            params = {"symbol": symbol, "token": self.finnhub_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "c" in data and data["c"] > 0:
                            return {
                                "price": data["c"],
                                "bid": data["c"] - 0.01,
                                "ask": data["c"] + 0.01,
                                "volume": data.get("v", 0),
                                "timestamp": datetime.now().isoformat()
                            }
        except Exception as e:
            logger.error(f"Stock API error: {e}")
        
        return None

# ============================================================================
# ENHANCED SCALPING STRATEGY
# ============================================================================

class UltraScalpStrategy:
    """Advanced scalping strategy with multiple signals"""
    
    def __init__(self):
        self.price_history = {}
        self.rsi_period = 14
        self.profit_target_pips = 5
        self.stop_loss_pips = 3
        self.max_spread_pips = 2
        
    def calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [c for c in changes[-self.rsi_period:] if c > 0]
        losses = [-c for c in changes[-self.rsi_period:] if c < 0]
        
        avg_gain = sum(gains) / self.rsi_period if gains else 0
        avg_loss = sum(losses) / self.rsi_period if losses else 0
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def analyze(self, symbol: str, price_data: Dict) -> Optional[Dict]:
        """Analyze price for scalp opportunity"""
        if not price_data:
            return None
        
        price = price_data.get('price', 0)
        bid = price_data.get('bid', price)
        ask = price_data.get('ask', price)
        
        if price <= 0:
            return None
        
        # Store price history
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=50)
        self.price_history[symbol].append(price)
        
        if len(self.price_history[symbol]) < 20:
            return None  # Need more data
        
        prices = list(self.price_history[symbol])
        
        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        
        # Momentum (5-tick change)
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        # Spread check
        spread = (ask - bid) / bid if bid > 0 else 1
        spread_pips = spread * 10000 if '_' in symbol else spread * 100
        
        if spread_pips > self.max_spread_pips:
            return None  # Spread too wide
        
        # Generate signals
        signal = None
        confidence = 0
        reason = ""
        
        # Strong RSI reversal
        if rsi < 20:
            signal = 'buy'
            confidence = 0.8
            reason = f"RSI oversold ({rsi:.0f})"
        elif rsi > 80:
            signal = 'sell'
            confidence = 0.8
            reason = f"RSI overbought ({rsi:.0f})"
        
        # Momentum spike
        elif abs(momentum) > 0.0005:  # 5 pips movement
            signal = 'buy' if momentum > 0 else 'sell'
            confidence = 0.7
            reason = f"Momentum spike ({momentum*10000:.1f} pips)"
        
        # Weak RSI signal
        elif rsi < 30:
            signal = 'buy'
            confidence = 0.6
            reason = f"RSI low ({rsi:.0f})"
        elif rsi > 70:
            signal = 'sell'
            confidence = 0.6
            reason = f"RSI high ({rsi:.0f})"
        
        if signal:
            # Calculate targets
            pip_value = 0.0001 if '_' in symbol else 0.01
            
            if signal == 'buy':
                target = price + (self.profit_target_pips * pip_value)
                stop = price - (self.stop_loss_pips * pip_value)
            else:
                target = price - (self.profit_target_pips * pip_value)
                stop = price + (self.stop_loss_pips * pip_value)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'entry': price,
                'target': target,
                'stop': stop,
                'rsi': rsi,
                'momentum': momentum,
                'spread_pips': spread_pips
            }
        
        return None

# ============================================================================
# TRINITY SCALPER V2.0 MAIN ENGINE
# ============================================================================

class TrinityScalperV2:
    """Ultra-fast scalping with premium APIs and real execution"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("⚡ TRINITY SCALPER V2.0 - PREMIUM EDITION")
        logger.info("=" * 60)
        
        # Initialize components
        self.creds = PremiumCredentialManager()
        self.broker = BrokerConnection(self.creds)
        self.data_mgr = PremiumDataManager(self.creds)
        self.strategy = UltraScalpStrategy()
        
        # Trading state
        self.active_trades = {}
        self.performance = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pips': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'daily_pnl': 0
        }
        
        # Symbols to trade
        self.forex_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        self.stocks = ['SPY', 'QQQ']
        
        # Risk management
        self.max_concurrent_trades = 3
        self.max_daily_loss = -500
        self.circuit_breaker = False
        
        # Control
        self.running = True
        self.last_heartbeat = time.time()
        
        logger.info("✅ All systems initialized")
        self.log_config()
    
    def log_config(self):
        """Log configuration"""
        logger.info("Configuration:")
        logger.info(f"  Forex pairs: {', '.join(self.forex_pairs)}")
        logger.info(f"  Stocks: {', '.join(self.stocks)}")
        logger.info(f"  Max concurrent: {self.max_concurrent_trades}")
        logger.info(f"  OANDA: {'✅' if self.broker.oanda_connected else '❌'}")
        logger.info(f"  Alpaca: {'✅' if self.broker.alpaca_connected else '❌'}")
        logger.info("=" * 60)
    
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time market data"""
        if '_' in symbol:  # Forex
            return await self.data_mgr.get_forex_price(symbol)
        else:  # Stock
            return await self.data_mgr.get_stock_price(symbol)
    
    async def execute_trade(self, symbol: str, signal: Dict) -> Dict:
        """Execute real trade on broker"""
        try:
            trade_id = f"T{int(time.time() * 1000)}"
            
            # Determine broker and size
            if '_' in symbol:  # Forex
                units = 5000  # 0.05 lots
                result = await self.broker.execute_forex_trade(
                    symbol, units, signal['signal']
                )
            else:  # Stock
                qty = 2  # 2 shares for scalping
                result = await self.broker.execute_stock_trade(
                    symbol, qty, signal['signal']
                )
            
            if result.get('success'):
                # Track trade
                self.active_trades[trade_id] = {
                    'symbol': symbol,
                    'side': signal['signal'],
                    'entry': result.get('price', signal['entry']),
                    'target': signal['target'],
                    'stop': signal['stop'],
                    'reason': signal['reason'],
                    'timestamp': datetime.now(),
                    'order_id': result.get('order_id')
                }
                
                logger.info(f"✅ TRADE EXECUTED: {symbol} {signal['signal'].upper()} "
                          f"@ {result.get('price', 0):.5f} - {signal['reason']}")
                
                return {'success': True, 'trade_id': trade_id}
            else:
                logger.warning(f"Trade failed: {result.get('error')}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def manage_trade(self, trade_id: str, current_price: float):
        """Manage open trade with stops and targets"""
        trade = self.active_trades[trade_id]
        entry = trade['entry']
        
        # Calculate P&L
        pip_value = 0.0001 if '_' in trade['symbol'] else 0.01
        if trade['side'] == 'buy':
            pnl_pips = (current_price - entry) / pip_value
            hit_target = current_price >= trade['target']
            hit_stop = current_price <= trade['stop']
        else:
            pnl_pips = (entry - current_price) / pip_value
            hit_target = current_price <= trade['target']
            hit_stop = current_price >= trade['stop']
        
        # Check exit conditions
        exit_reason = None
        if hit_target:
            exit_reason = 'TARGET'
        elif hit_stop:
            exit_reason = 'STOP'
        elif (datetime.now() - trade['timestamp']).seconds > 300:  # 5 min timeout
            exit_reason = 'TIMEOUT'
        
        if exit_reason:
            # Close trade
            del self.active_trades[trade_id]
            
            # Update performance
            self.performance['total_trades'] += 1
            if pnl_pips > 0:
                self.performance['wins'] += 1
            else:
                self.performance['losses'] += 1
            
            self.performance['total_pips'] += pnl_pips
            self.performance['best_trade'] = max(self.performance['best_trade'], pnl_pips)
            self.performance['worst_trade'] = min(self.performance['worst_trade'], pnl_pips)
            self.performance['daily_pnl'] += pnl_pips * 10  # Approximate $ value
            
            # Log result
            emoji = '✅' if pnl_pips > 0 else '❌'
            logger.info(f"{emoji} CLOSED: {trade['symbol']} {pnl_pips:+.1f} pips ({exit_reason})")
            
            # Check circuit breaker
            if self.performance['daily_pnl'] <= self.max_daily_loss:
                self.circuit_breaker = True
                logger.error("🛑 CIRCUIT BREAKER: Daily loss limit reached\!")
    
    async def scalping_loop(self):
        """Main scalping loop"""
        logger.info("🚀 Starting scalping loop...")
        
        cycle = 0
        while self.running:
            try:
                cycle += 1
                
                # Heartbeat
                self.last_heartbeat = time.time()
                
                # Check circuit breaker
                if self.circuit_breaker:
                    logger.warning("Circuit breaker active, waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Process each symbol
                tasks = []
                
                # Forex pairs
                for symbol in self.forex_pairs:
                    if len(self.active_trades) >= self.max_concurrent_trades:
                        break
                    
                    # Get real-time data
                    price_data = await self.get_market_data(symbol)
                    if price_data:
                        # Analyze for opportunity
                        signal = self.strategy.analyze(symbol, price_data)
                        if signal and signal['confidence'] >= 0.6:
                            # Execute trade
                            await self.execute_trade(symbol, signal)
                
                # Stocks
                for symbol in self.stocks:
                    if len(self.active_trades) >= self.max_concurrent_trades:
                        break
                    
                    price_data = await self.get_market_data(symbol)
                    if price_data:
                        signal = self.strategy.analyze(symbol, price_data)
                        if signal and signal['confidence'] >= 0.6:
                            await self.execute_trade(symbol, signal)
                
                # Manage open trades
                for trade_id, trade in list(self.active_trades.items()):
                    current_data = await self.get_market_data(trade['symbol'])
                    if current_data:
                        await self.manage_trade(trade_id, current_data['price'])
                
                # Performance update every 20 trades
                if self.performance['total_trades'] > 0 and self.performance['total_trades'] % 20 == 0:
                    self.log_performance()
                
                # Intelligent delay based on market activity
                if len(self.active_trades) > 0:
                    await asyncio.sleep(1)  # Fast when trading
                else:
                    await asyncio.sleep(2)  # Slower when waiting
                
            except Exception as e:
                logger.error(f"Loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)
    
    def log_performance(self):
        """Log performance metrics"""
        p = self.performance
        win_rate = (p['wins'] / p['total_trades'] * 100) if p['total_trades'] > 0 else 0
        
        logger.info("=" * 60)
        logger.info("📊 PERFORMANCE UPDATE")
        logger.info(f"  Trades: {p['total_trades']} ({p['wins']}W / {p['losses']}L)")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Total: {p['total_pips']:.1f} pips")
        logger.info(f"  Best: {p['best_trade']:.1f} / Worst: {p['worst_trade']:.1f}")
        logger.info(f"  Daily P&L: ${p['daily_pnl']:.2f}")
        logger.info(f"  Active: {len(self.active_trades)} trades")
        logger.info("=" * 60)
    
    async def watchdog(self):
        """Watchdog to detect freezes"""
        while self.running:
            await asyncio.sleep(10)
            
            if time.time() - self.last_heartbeat > 30:
                logger.error("⚠️ Scalper appears frozen, restarting...")
                self.running = False
                break
    
    def run(self):
        """Run the scalper"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run scalping and watchdog concurrently
            tasks = [
                loop.create_task(self.scalping_loop()),
                loop.create_task(self.watchdog())
            ]
            loop.run_until_complete(asyncio.gather(*tasks))
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.running = False
            
            # Final performance
            self.log_performance()
            
            # Save state
            state = {
                'performance': self.performance,
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                with open('/tmp/scalper_state.json', 'w') as f:
                    json.dump(state, f)
                logger.info("State saved")
            except:
                pass
            
            loop.close()
            logger.info("👋 Trinity Scalper V2.0 shutdown complete")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     ⚡ TRINITY SCALPER V2.0 - PREMIUM EDITION               ║
    ║                                                              ║
    ║     75 API calls/minute | Real execution | Paper trading    ║
    ║     OANDA Forex | Alpaca Stocks | 5 pip targets            ║
    ║                                                              ║
    ║     "Zero humans, infinite intelligence"                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Install aiohttp if needed
        import aiohttp
    except ImportError:
        logger.info("Installing aiohttp...")
        os.system("pip3 install aiohttp")
    
    # Run scalper
    scalper = TrinityScalperV2()
    scalper.run()

if __name__ == "__main__":
    main()
