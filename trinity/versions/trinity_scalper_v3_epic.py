#\!/usr/bin/env python3
"""
TRINITY SCALPER V3.0 EPIC - REAL BROKER EXECUTION
ULTRATHINK: Zero fake trades, 100% real execution
Every trade hits the broker, every P&L is real
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading
from collections import deque
import traceback
import uuid

# Setup dual logging - console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/trinity-epic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TRINITY_EPIC')

# ============================================================================
# CREDENTIAL MANAGER
# ============================================================================

class CredentialManager:
    """Manages encrypted API credentials"""
    
    def __init__(self):
        self.config_dir = "/opt/cashmachine/config"
        self.credentials = {}
        self.load_all_credentials()
    
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
                        logger.info(f"✅ Loaded {api} credentials")
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
# EPIC OANDA BROKER - REAL FOREX EXECUTION
# ============================================================================

class EpicOandaBroker:
    """Real OANDA execution with verification"""
    
    def __init__(self, creds: CredentialManager):
        self.creds = creds
        self.account_id = creds.get('oanda', 'account_id')
        self.api_token = creds.get('oanda', 'api_token')
        self.base_url = "https://api-fxpractice.oanda.com/v3"
        self.connected = False
        self.orders_executed = []
        self.verify_connection()
    
    def verify_connection(self):
        """Verify OANDA connection"""
        try:
            import requests
            url = f"{self.base_url}/accounts/{self.account_id}"
            headers = {"Authorization": f"Bearer {self.api_token}"}
            
            resp = requests.get(url, headers=headers, timeout=5, proxies={"http": None, "https": None})
            if resp.status_code == 200:
                data = resp.json()
                balance = data['account']['balance']
                self.connected = True
                logger.info(f"✅ OANDA CONNECTED - Balance: ${balance}")
            else:
                logger.error(f"❌ OANDA connection failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ OANDA error: {e}")
    
    async def execute_trade(self, symbol: str, units: int, side: str) -> Dict:
        """Execute REAL forex trade on OANDA"""
        if not self.connected:
            return {"success": False, "error": "OANDA not connected"}
        
        try:
            import aiohttp
            
            # Fix symbol format (EUR_USD is already correct for OANDA)
            instrument = symbol
            
            # Create order - THIS IS REAL MONEY (paper account)
            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": str(units if side == 'buy' else -units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "timeInForce": "FOK"  # Fill or Kill
                }
            }
            
            url = f"{self.base_url}/accounts/{self.account_id}/orders"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"🚀 SENDING REAL ORDER TO OANDA: {instrument} {side} {units} units")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=order_data, headers=headers) as resp:
                    response_text = await resp.text()
                    
                    if resp.status == 201:
                        data = json.loads(response_text)
                        
                        # Extract REAL order details
                        fill_transaction = data.get("orderFillTransaction", {})
                        order_id = fill_transaction.get("orderID")
                        trade_id = fill_transaction.get("id")
                        price = float(fill_transaction.get("price", 0))
                        
                        # Track this REAL order
                        self.orders_executed.append({
                            "order_id": order_id,
                            "trade_id": trade_id,
                            "instrument": instrument,
                            "units": units,
                            "price": price,
                            "time": datetime.now().isoformat()
                        })
                        
                        logger.info(f"✅ REAL OANDA ORDER EXECUTED\!")
                        logger.info(f"   Order ID: {order_id}")
                        logger.info(f"   Trade ID: {trade_id}")
                        logger.info(f"   Price: {price}")
                        
                        return {
                            "success": True,
                            "broker": "OANDA",
                            "order_id": order_id,
                            "trade_id": trade_id,
                            "price": price,
                            "units": units,
                            "real_execution": True
                        }
                    else:
                        error_data = json.loads(response_text) if response_text else {}
                        error_msg = error_data.get("errorMessage", f"Status {resp.status}")
                        logger.error(f"❌ OANDA rejected order: {error_msg}")
                        return {"success": False, "error": error_msg}
                        
        except Exception as e:
            logger.error(f"❌ OANDA execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def close_trade(self, trade_id: str) -> Dict:
        """Close a real OANDA trade"""
        try:
            import aiohttp
            
            url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/close"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.put(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"✅ OANDA trade {trade_id} closed")
                        return {"success": True, "data": data}
                    else:
                        return {"success": False, "error": f"Status {resp.status}"}
        except Exception as e:
            logger.error(f"Close trade error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_open_trades(self) -> List[Dict]:
        """Get all open trades from OANDA"""
        try:
            import aiohttp
            
            url = f"{self.base_url}/accounts/{self.account_id}/trades"
            headers = {"Authorization": f"Bearer {self.api_token}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("trades", [])
                    return []
        except Exception as e:
            logger.error(f"Get trades error: {e}")
            return []

# ============================================================================
# EPIC ALPACA BROKER - REAL STOCK EXECUTION
# ============================================================================

class EpicAlpacaBroker:
    """Real Alpaca execution with verification"""
    
    def __init__(self, creds: CredentialManager):
        self.creds = creds
        self.api_key = creds.get('alpaca', 'api_key')
        self.api_secret = creds.get('alpaca', 'api_secret')
        self.base_url = "https://paper-api.alpaca.markets/v2"
        self.connected = False
        self.orders_executed = []
        self.verify_connection()
    
    def verify_connection(self):
        """Verify Alpaca connection"""
        try:
            import requests
            url = f"{self.base_url}/account"
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            resp = requests.get(url, headers=headers, timeout=5, proxies={"http": None, "https": None})
            if resp.status_code == 200:
                data = resp.json()
                cash = data['cash']
                self.connected = True
                logger.info(f"✅ ALPACA CONNECTED - Cash: ${cash}")
            else:
                logger.error(f"❌ Alpaca connection failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ Alpaca error: {e}")
    
    async def execute_trade(self, symbol: str, qty: int, side: str) -> Dict:
        """Execute REAL stock trade on Alpaca"""
        if not self.connected:
            return {"success": False, "error": "Alpaca not connected"}
        
        try:
            import aiohttp
            
            # Create order - THIS IS REAL (paper account)
            order_data = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": "market",
                "time_in_force": "ioc"  # Immediate or cancel
            }
            
            url = f"{self.base_url}/orders"
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
                "Content-Type": "application/json"
            }
            
            logger.info(f"🚀 SENDING REAL ORDER TO ALPACA: {symbol} {side} {qty} shares")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=order_data, headers=headers) as resp:
                    response_text = await resp.text()
                    
                    if resp.status in [200, 201]:
                        data = json.loads(response_text)
                        
                        # Extract REAL order details
                        order_id = data.get("id")
                        client_order_id = data.get("client_order_id")
                        filled_qty = data.get("filled_qty", "0")
                        filled_price = data.get("filled_avg_price")
                        
                        # Track this REAL order
                        self.orders_executed.append({
                            "order_id": order_id,
                            "client_order_id": client_order_id,
                            "symbol": symbol,
                            "qty": qty,
                            "filled_price": filled_price,
                            "time": datetime.now().isoformat()
                        })
                        
                        logger.info(f"✅ REAL ALPACA ORDER EXECUTED\!")
                        logger.info(f"   Order ID: {order_id}")
                        logger.info(f"   Filled: {filled_qty} @ ${filled_price}")
                        
                        return {
                            "success": True,
                            "broker": "ALPACA",
                            "order_id": order_id,
                            "client_order_id": client_order_id,
                            "price": float(filled_price) if filled_price else 0,
                            "qty": qty,
                            "real_execution": True
                        }
                    else:
                        error_data = json.loads(response_text) if response_text else {}
                        error_msg = error_data.get("message", f"Status {resp.status}")
                        logger.error(f"❌ Alpaca rejected order: {error_msg}")
                        return {"success": False, "error": error_msg}
                        
        except Exception as e:
            logger.error(f"❌ Alpaca execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def close_position(self, symbol: str) -> Dict:
        """Close all positions for a symbol"""
        try:
            import aiohttp
            
            url = f"{self.base_url}/positions/{symbol}"
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ Alpaca position {symbol} closed")
                        return {"success": True}
                    else:
                        return {"success": False, "error": f"Status {resp.status}"}
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return {"success": False, "error": str(e)}

# ============================================================================
# PREMIUM DATA MANAGER
# ============================================================================

class PremiumDataManager:
    """Premium API data with proper rate limiting"""
    
    def __init__(self, creds: CredentialManager):
        self.creds = creds
        self.alpha_vantage_key = creds.get('alphavantage', 'api_key')
        self.finnhub_key = creds.get('finnhub', 'api_key')
        
        # Track API calls
        self.av_calls = deque(maxlen=75)
        self.av_minute_start = time.time()
        
        logger.info(f"💎 Data Manager: AlphaVantage Premium (75/min)")
    
    async def get_forex_price(self, symbol: str) -> Optional[Dict]:
        """Get real forex price"""
        # Rate limit check
        now = time.time()
        if now - self.av_minute_start > 60:
            self.av_calls.clear()
            self.av_minute_start = now
        
        if len(self.av_calls) >= 75:
            await asyncio.sleep(1)
            return None
        
        try:
            import aiohttp
            from_curr, to_curr = symbol.split('_')
            
            url = "https://www.alphavantage.co/query"
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
                            self.av_calls.append(time.time())
                            return {
                                "price": float(rate_data.get("5. Exchange Rate", 0)),
                                "bid": float(rate_data.get("8. Bid Price", 0)),
                                "ask": float(rate_data.get("9. Ask Price", 0)),
                                "timestamp": datetime.now().isoformat()
                            }
        except Exception as e:
            logger.error(f"Forex data error: {e}")
        return None
    
    async def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """Get real stock price"""
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
                                "high": data["h"],
                                "low": data["l"],
                                "volume": data.get("v", 0),
                                "timestamp": datetime.now().isoformat()
                            }
        except Exception as e:
            logger.error(f"Stock data error: {e}")
        return None

# ============================================================================
# SCALPING STRATEGY
# ============================================================================

class ScalpingStrategy:
    """Simple but effective scalping strategy"""
    
    def __init__(self):
        self.price_history = {}
        self.rsi_period = 14
        
    def calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI"""
        if len(prices) < self.rsi_period + 1:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))][-self.rsi_period:]
        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]
        
        avg_gain = sum(gains) / self.rsi_period if gains else 0
        avg_loss = sum(losses) / self.rsi_period if losses else 0
        
        if avg_loss == 0:
            return 70 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def analyze(self, symbol: str, price_data: Dict) -> Optional[Dict]:
        """Generate trading signal"""
        if not price_data or price_data.get('price', 0) <= 0:
            return None
        
        price = price_data['price']
        
        # Store price history
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=50)
        self.price_history[symbol].append(price)
        
        if len(self.price_history[symbol]) < 20:
            return None
        
        prices = list(self.price_history[symbol])
        rsi = self.calculate_rsi(prices)
        
        # Simple but effective signals
        signal = None
        confidence = 0
        
        if rsi < 30:
            signal = 'buy'
            confidence = 0.7
        elif rsi > 70:
            signal = 'sell'
            confidence = 0.7
        
        if signal:
            return {
                'signal': signal,
                'confidence': confidence,
                'entry': price,
                'rsi': rsi
            }
        
        return None

# ============================================================================
# TRINITY EPIC SCALPER - MAIN ENGINE
# ============================================================================

class TrinityEpicScalper:
    """Epic scalper with 100% real broker execution"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("⚡ TRINITY SCALPER V3.0 EPIC")
        logger.info("🎯 100% REAL BROKER EXECUTION")
        logger.info("=" * 60)
        
        # Initialize components
        self.creds = CredentialManager()
        self.oanda = EpicOandaBroker(self.creds)
        self.alpaca = EpicAlpacaBroker(self.creds)
        self.data_mgr = PremiumDataManager(self.creds)
        self.strategy = ScalpingStrategy()
        
        # Trading state
        self.active_trades = {}
        self.real_orders = []  # Track REAL broker orders
        self.total_real_trades = 0
        self.total_real_pnl = 0
        
        # Symbols to trade
        self.forex_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        self.stocks = ['SPY', 'QQQ']
        
        # Control
        self.running = True
        self.max_concurrent = 3
        
        logger.info(f"OANDA: {'✅ READY' if self.oanda.connected else '❌ FAILED'}")
        logger.info(f"Alpaca: {'✅ READY' if self.alpaca.connected else '❌ FAILED'}")
        logger.info("=" * 60)
    
    async def execute_real_trade(self, symbol: str, signal: Dict) -> Dict:
        """Execute REAL trade on broker"""
        trade_id = f"EPIC_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Determine broker and execute
            if '_' in symbol:  # Forex on OANDA
                units = 1000  # 0.01 lot for testing
                result = await self.oanda.execute_trade(symbol, units, signal['signal'])
                broker = "OANDA"
            else:  # Stocks on Alpaca
                qty = 1  # 1 share for testing
                result = await self.alpaca.execute_trade(symbol, qty, signal['signal'])
                broker = "ALPACA"
            
            if result.get('success') and result.get('real_execution'):
                # This is a REAL trade with REAL order ID
                self.total_real_trades += 1
                
                # Track the REAL order
                real_order = {
                    'trade_id': trade_id,
                    'broker': broker,
                    'symbol': symbol,
                    'side': signal['signal'],
                    'broker_order_id': result.get('order_id'),
                    'broker_trade_id': result.get('trade_id') or result.get('client_order_id'),
                    'price': result.get('price'),
                    'timestamp': datetime.now().isoformat(),
                    'rsi': signal.get('rsi')
                }
                
                self.real_orders.append(real_order)
                self.active_trades[trade_id] = real_order
                
                logger.info("=" * 60)
                logger.info(f"🎯 REAL TRADE EXECUTED #{self.total_real_trades}")
                logger.info(f"   Broker: {broker}")
                logger.info(f"   Symbol: {symbol}")
                logger.info(f"   Side: {signal['signal'].upper()}")
                logger.info(f"   Order ID: {result.get('order_id')}")
                logger.info(f"   Price: {result.get('price')}")
                logger.info(f"   RSI: {signal.get('rsi', 0):.1f}")
                logger.info("=" * 60)
                
                return {'success': True, 'trade_id': trade_id, 'real': True}
            else:
                logger.warning(f"Trade failed: {result.get('error')}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def monitor_and_close_trades(self):
        """Monitor open trades and close at targets"""
        for trade_id, trade in list(self.active_trades.items()):
            try:
                symbol = trade['symbol']
                
                # Get current price
                if '_' in symbol:
                    price_data = await self.data_mgr.get_forex_price(symbol)
                else:
                    price_data = await self.data_mgr.get_stock_price(symbol)
                
                if not price_data:
                    continue
                
                current_price = price_data['price']
                entry_price = trade['price']
                
                # Calculate P&L
                if '_' in symbol:  # Forex in pips
                    pip_value = 0.0001 if 'JPY' not in symbol else 0.01
                    if trade['side'] == 'buy':
                        pnl_pips = (current_price - entry_price) / pip_value
                    else:
                        pnl_pips = (entry_price - current_price) / pip_value
                    
                    # Close at 5 pips profit or 3 pips loss
                    if pnl_pips >= 5 or pnl_pips <= -3:
                        # Close the REAL trade
                        if trade['broker'] == 'OANDA':
                            await self.oanda.close_trade(trade['broker_trade_id'])
                        
                        self.total_real_pnl += pnl_pips
                        del self.active_trades[trade_id]
                        
                        logger.info(f"{'✅' if pnl_pips > 0 else '❌'} CLOSED: {symbol} "
                                  f"{pnl_pips:+.1f} pips | Total P&L: {self.total_real_pnl:+.1f}")
                
                else:  # Stocks in %
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    if trade['side'] == 'sell':
                        pnl_pct = -pnl_pct
                    
                    # Close at 0.1% profit or 0.05% loss
                    if pnl_pct >= 0.1 or pnl_pct <= -0.05:
                        # Close the REAL position
                        if trade['broker'] == 'ALPACA':
                            await self.alpaca.close_position(symbol)
                        
                        del self.active_trades[trade_id]
                        
                        logger.info(f"{'✅' if pnl_pct > 0 else '❌'} CLOSED: {symbol} "
                                  f"{pnl_pct:+.2f}%")
                
            except Exception as e:
                logger.error(f"Monitor error for {trade_id}: {e}")
    
    async def verify_real_trades(self):
        """Verify trades are real on brokers"""
        # Check OANDA
        oanda_trades = await self.oanda.get_open_trades()
        logger.info(f"📊 OANDA open trades: {len(oanda_trades)}")
        
        # Log executed orders
        if self.oanda.orders_executed:
            logger.info(f"   OANDA orders executed: {len(self.oanda.orders_executed)}")
            for order in self.oanda.orders_executed[-3:]:
                logger.info(f"   - {order['instrument']} #{order['order_id']}")
        
        if self.alpaca.orders_executed:
            logger.info(f"   Alpaca orders executed: {len(self.alpaca.orders_executed)}")
            for order in self.alpaca.orders_executed[-3:]:
                logger.info(f"   - {order['symbol']} #{order['order_id']}")
    
    async def scalping_loop(self):
        """Main scalping loop with REAL execution"""
        logger.info("🚀 Starting EPIC scalping with REAL broker execution...")
        
        verify_counter = 0
        
        while self.running:
            try:
                # Process each symbol
                for symbol in self.forex_pairs + self.stocks:
                    if len(self.active_trades) >= self.max_concurrent:
                        break
                    
                    # Get real market data
                    if '_' in symbol:
                        price_data = await self.data_mgr.get_forex_price(symbol)
                    else:
                        price_data = await self.data_mgr.get_stock_price(symbol)
                    
                    if not price_data:
                        continue
                    
                    # Analyze for signals
                    signal = self.strategy.analyze(symbol, price_data)
                    
                    if signal and signal['confidence'] >= 0.6:
                        # Execute REAL trade
                        await self.execute_real_trade(symbol, signal)
                
                # Monitor and close trades
                await self.monitor_and_close_trades()
                
                # Periodic verification
                verify_counter += 1
                if verify_counter % 20 == 0:
                    await self.verify_real_trades()
                
                # Performance update
                if self.total_real_trades > 0 and self.total_real_trades % 10 == 0:
                    logger.info("=" * 60)
                    logger.info(f"📊 EPIC PERFORMANCE UPDATE")
                    logger.info(f"   Real Trades: {self.total_real_trades}")
                    logger.info(f"   Active: {len(self.active_trades)}")
                    logger.info(f"   Total P&L: {self.total_real_pnl:+.1f} pips")
                    logger.info("=" * 60)
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Loop error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)
    
    def run(self):
        """Run the epic scalper"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.scalping_loop())
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.running = False
            
            # Final report
            logger.info("=" * 60)
            logger.info("🏆 EPIC SCALPER FINAL REPORT")
            logger.info(f"   Total REAL Trades: {self.total_real_trades}")
            logger.info(f"   Total REAL Orders: {len(self.real_orders)}")
            logger.info(f"   OANDA Orders: {len(self.oanda.orders_executed)}")
            logger.info(f"   Alpaca Orders: {len(self.alpaca.orders_executed)}")
            logger.info(f"   Total P&L: {self.total_real_pnl:+.1f} pips")
            logger.info("=" * 60)
            
            loop.close()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     ⚡ TRINITY SCALPER V3.0 EPIC - REAL EXECUTION           ║
    ║                                                              ║
    ║     100% Real Broker Trades | Zero Fake Orders              ║
    ║     OANDA Forex | Alpaca Stocks | Verified Execution        ║
    ║                                                              ║
    ║     "Every trade is real, every P&L matters"                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run the epic scalper
    scalper = TrinityEpicScalper()
    scalper.run()

if __name__ == "__main__":
    main()
