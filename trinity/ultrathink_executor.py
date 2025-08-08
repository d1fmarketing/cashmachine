#!/usr/bin/env python3
"""
ULTRATHINK REAL EXECUTOR - Paper Trading with REAL API Calls
NO SIMULATION - REAL TRADES ONLY
"""

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import requests
import redis.asyncio as redis
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment
load_dotenv('/opt/cashmachine/trinity/.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ULTRATHINK_EXECUTOR')

# CRITICAL: Force paper trading mode
TRADING_MODE = os.environ.get('TRADING_MODE', 'paper')
if TRADING_MODE not in ['paper', 'live']:
    logger.error("❌ TRADING_MODE must be 'paper' or 'live'")
    sys.exit(1)

logger.info(f"🚀 ULTRATHINK EXECUTOR - MODE: {TRADING_MODE}")

# ============================================================================
# ALPACA PAPER TRADING CLIENT
# ============================================================================

class AlpacaClient:
    """Real Alpaca paper trading API client - NO SIMULATION"""
    
    def __init__(self):
        self.api_key = os.environ.get('ALPACA_API_KEY')
        self.api_secret = os.environ.get('ALPACA_API_SECRET')
        self.base_url = 'https://paper-api.alpaca.markets'
        
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ Alpaca credentials missing!")
        
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test Alpaca API connection"""
        try:
            resp = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers,
                timeout=10
            )
            resp.raise_for_status()
            account = resp.json()
            logger.info(f"✅ Alpaca connected - Balance: ${account.get('cash', 0)}")
        except Exception as e:
            logger.error(f"❌ Alpaca connection failed: {e}")
            raise
    
    def place_market_order(self, symbol: str, side: str, qty: float) -> Dict:
        """Place REAL market order on Alpaca"""
        idempotency_key = f"ultra-{uuid.uuid4()}"
        
        order_data = {
            'symbol': symbol,
            'qty': str(qty),
            'side': side,
            'type': 'market',
            'time_in_force': 'gtc'
        }
        
        logger.info(f"📤 Placing REAL Alpaca order: {side} {qty} {symbol}")
        
        try:
            resp = requests.post(
                f"{self.base_url}/v2/orders",
                json=order_data,
                headers={**self.headers, 'Idempotency-Key': idempotency_key},
                timeout=10
            )
            resp.raise_for_status()
            order = resp.json()
            
            logger.info(f"✅ REAL ORDER PLACED - ID: {order['id']}, Status: {order['status']}")
            
            return {
                'success': True,
                'order_id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'qty': order['qty'],
                'status': order['status'],
                'idempotency_key': idempotency_key,
                'broker': 'ALPACA',
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Alpaca order failed: {e.response.text}")
            return {
                'success': False,
                'error': str(e),
                'broker': 'ALPACA'
            }
    
    def get_recent_orders(self, limit: int = 5) -> list:
        """Get recent orders to verify trades"""
        try:
            resp = requests.get(
                f"{self.base_url}/v2/orders",
                params={'status': 'all', 'limit': limit},
                headers=self.headers,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

# ============================================================================
# OANDA PRACTICE TRADING CLIENT
# ============================================================================

class OandaClient:
    """Real OANDA practice trading API client - NO SIMULATION"""
    
    def __init__(self):
        self.api_token = os.environ.get('OANDA_API_TOKEN')
        self.account_id = os.environ.get('OANDA_ACCOUNT_ID')
        self.base_url = 'https://api-fxpractice.oanda.com'
        
        if not self.api_token or not self.account_id:
            raise ValueError("❌ OANDA credentials missing!")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test OANDA API connection"""
        try:
            resp = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}",
                headers=self.headers,
                timeout=10
            )
            resp.raise_for_status()
            account = resp.json()['account']
            logger.info(f"✅ OANDA connected - Balance: {account.get('balance', 0)}")
        except Exception as e:
            logger.error(f"❌ OANDA connection failed: {e}")
            raise
    
    def place_market_order(self, instrument: str, units: int) -> Dict:
        """Place REAL market order on OANDA"""
        order_data = {
            'order': {
                'instrument': instrument,
                'units': str(units),
                'type': 'MARKET',
                'timeInForce': 'FOK',
                'positionFill': 'DEFAULT'
            }
        }
        
        logger.info(f"📤 Placing REAL OANDA order: {units} {instrument}")
        
        try:
            resp = requests.post(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders",
                json=order_data,
                headers=self.headers,
                timeout=10
            )
            resp.raise_for_status()
            result = resp.json()
            
            if 'orderFillTransaction' in result:
                fill = result['orderFillTransaction']
                logger.info(f"✅ REAL ORDER FILLED - ID: {fill['id']}, Units: {fill['units']}")
                
                return {
                    'success': True,
                    'order_id': fill['id'],
                    'instrument': fill['instrument'],
                    'units': fill['units'],
                    'price': fill.get('price', 'N/A'),
                    'broker': 'OANDA',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Order created but not filled: {result}")
                return {
                    'success': False,
                    'error': 'Order not filled',
                    'broker': 'OANDA'
                }
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ OANDA order failed: {e.response.text}")
            return {
                'success': False,
                'error': str(e),
                'broker': 'OANDA'
            }

# ============================================================================
# ULTRATHINK EXECUTOR
# ============================================================================

class UltraThinkExecutor:
    """Main executor for ULTRATHINK - REAL TRADES ONLY"""
    
    def __init__(self):
        self.alpaca = AlpacaClient()
        self.oanda = OandaClient()
        self.redis_client = None
        self.executed_trades = []
        
    async def connect_redis(self):
        """Connect to Redis"""
        self.redis_client = await redis.Redis(
            host='10.100.2.200',
            port=6379,
            decode_responses=True
        )
        await self.redis_client.ping()
        logger.info("✅ Connected to Redis")
    
    async def execute_signal(self, signal: Dict):
        """Execute trading signal with REAL API calls"""
        
        if signal.get('signal') == 'HOLD':
            logger.info("Signal is HOLD - no trade")
            return
        
        # Determine trade parameters
        symbol = signal.get('symbol', 'SOLUSD')
        side = 'buy' if signal['signal'] == 'BUY' else 'sell'
        confidence = float(signal.get('confidence', 0.5))
        
        # Calculate position size (small for testing)
        if confidence > 0.7:
            qty = 0.5  # Higher confidence
        elif confidence > 0.5:
            qty = 0.2  # Medium confidence
        else:
            qty = 0.1  # Low confidence (test size)
        
        logger.info(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                   EXECUTING REAL TRADE                        ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ Signal: {signal['signal']:8} Confidence: {confidence:.2%}              ║
        ║ Symbol: {symbol:8} Side: {side:8} Qty: {qty}              ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Execute on Alpaca (crypto/stocks)
        if symbol in ['SOLUSD', 'BTCUSD', 'ETHUSD', 'SPY', 'QQQ']:
            result = self.alpaca.place_market_order(symbol, side, qty)
        # Execute on OANDA (forex)
        elif symbol in ['EUR_USD', 'GBP_USD', 'USD_JPY']:
            units = int(qty * 1000) if side == 'buy' else -int(qty * 1000)
            result = self.oanda.place_market_order(symbol, units)
        else:
            logger.warning(f"Unknown symbol: {symbol}")
            return
        
        # Store execution result
        if result['success']:
            # Store in Redis
            trade_record = {
                'id': str(uuid.uuid4()),
                'order_id': result['order_id'],
                'symbol': symbol,
                'side': side,
                'qty': str(qty),
                'confidence': str(confidence),
                'broker': result['broker'],
                'status': 'EXECUTED',
                'timestamp': result['timestamp']
            }
            
            # Add to Redis list
            await self.redis_client.lpush(
                'trinity:real_trades',
                json.dumps(trade_record)
            )
            
            # Update metrics with REAL trade
            await self._update_real_metrics(trade_record)
            
            self.executed_trades.append(trade_record)
            
            logger.info(f"""
            ✅✅✅ REAL TRADE EXECUTED ✅✅✅
            Order ID: {result['order_id']}
            Broker: {result['broker']}
            Total Real Trades Today: {len(self.executed_trades)}
            """)
        else:
            logger.error(f"❌ Trade execution failed: {result.get('error')}")
    
    async def _update_real_metrics(self, trade: Dict):
        """Update metrics with REAL trades only"""
        metrics = await self.redis_client.hgetall('ultrathink:real_metrics')
        
        total_trades = int(metrics.get('total_trades', 0)) + 1
        
        await self.redis_client.hset(
            'ultrathink:real_metrics',
            mapping={
                'total_trades': str(total_trades),
                'last_trade_time': trade['timestamp'],
                'last_trade_symbol': trade['symbol'],
                'last_trade_side': trade['side']
            }
        )
    
    async def verify_with_broker(self):
        """Verify trades with broker API"""
        logger.info("🔍 Verifying trades with broker...")
        
        # Check Alpaca
        alpaca_orders = self.alpaca.get_recent_orders(5)
        logger.info(f"Alpaca recent orders: {len(alpaca_orders)}")
        
        for order in alpaca_orders[:2]:  # Show last 2
            logger.info(f"  - {order['side']} {order['qty']} {order['symbol']} - {order['status']}")
        
        return len(alpaca_orders)
    
    async def monitor_signals(self):
        """Monitor and execute ULTRATHINK signals"""
        logger.info("""
        ╔══════════════════════════════════════════════════════════════╗
        ║         ULTRATHINK REAL EXECUTOR STARTED                      ║
        ║                                                                ║
        ║  Mode: PAPER TRADING (REAL API CALLS)                        ║
        ║  Brokers: Alpaca & OANDA                                     ║
        ║  NO SIMULATION - REAL TRADES ONLY                            ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        while True:
            try:
                # Get latest signal
                signal_data = await self.redis_client.hgetall('ultrathink:signals')
                
                if signal_data and signal_data.get('signal'):
                    # Check if signal is fresh (within 60 seconds)
                    signal_time = signal_data.get('timestamp', '')
                    if signal_time:
                        from datetime import datetime, timedelta
                        signal_dt = datetime.fromisoformat(signal_time)
                        if datetime.now() - signal_dt < timedelta(seconds=60):
                            await self.execute_signal(signal_data)
                        else:
                            logger.debug("Signal too old, waiting for fresh signal")
                
                # Verify with broker every 5 iterations
                if len(self.executed_trades) % 5 == 1 and len(self.executed_trades) > 0:
                    await self.verify_with_broker()
                
                # Wait before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5)
    
    async def run(self):
        """Main execution loop"""
        await self.connect_redis()
        
        # Run a test trade first
        logger.info("📊 Running test trade to verify connection...")
        test_signal = {
            'signal': 'BUY',
            'symbol': 'SOLUSD',
            'confidence': '0.5'
        }
        await self.execute_signal(test_signal)
        
        # Verify the test trade
        broker_count = await self.verify_with_broker()
        if broker_count > 0:
            logger.info("✅ Test trade successful! Starting monitoring...")
            await self.monitor_signals()
        else:
            logger.error("❌ Test trade not found at broker. Check credentials!")

async def main():
    """Main entry point"""
    # Ensure we're in paper mode
    os.environ['TRADING_MODE'] = 'paper'
    
    executor = UltraThinkExecutor()
    await executor.run()

if __name__ == "__main__":
    asyncio.run(main())