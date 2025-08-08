#\!/usr/bin/env python3
"""
TRINITY SCALPER - Ultra-fast Black Box Scalping System
ULTRATHINK: Microsecond decisions, continuous profits
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
import threading

# Setup logging
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - SCALPER - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/var/log/trinity-scalper.log'),
            logging.StreamHandler()
        ]
    )
except:
    # Fallback to console only
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - SCALPER - %(levelname)s - %(message)s'
    )

logger = logging.getLogger('TRINITY_SCALPER')

class SimpleCredentialManager:
    """Simple credential manager"""
    def __init__(self):
        self.config_dir = "/opt/cashmachine/config"
        self.credentials = {}
        self.load_credentials()
    
    def load_credentials(self):
        """Load API credentials"""
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
                        logger.info(f"✅ {api} credentials loaded")
                except Exception as e:
                    logger.warning(f"Could not load {api}: {e}")
        except ImportError:
            logger.warning("Cryptography not available, using demo mode")
    
    def get_credential(self, api: str, key: str) -> Optional[str]:
        """Get credential"""
        if api in self.credentials:
            return self.credentials[api].get(key)
        return "demo"  # Fallback to demo

class SimpleBrain:
    """Simple Redis brain interface"""
    def __init__(self):
        self.patterns = []
        self.trades = []
        try:
            import redis
            self.redis = redis.Redis(host='10.100.2.200', port=6379, decode_responses=True)
            self.redis.ping()
            self.redis_available = True
            logger.info("✅ Connected to Redis brain")
        except:
            self.redis_available = False
            logger.info("⚠️ Redis not available, using local memory")
    
    def learn_from_trade(self, trade_data: Dict):
        """Store trade result"""
        self.trades.append(trade_data)
        if self.redis_available:
            try:
                self.redis.lpush('scalp_trades', json.dumps(trade_data))
                self.redis.ltrim('scalp_trades', 0, 999)  # Keep last 1000
            except:
                pass

class ScalpingStrategy:
    """High-frequency scalping strategy"""
    
    def __init__(self):
        self.tick_buffer = {}
        self.spread_threshold = 0.0002  # 2 pips
        self.profit_target = 0.0005  # 5 pips
        self.stop_loss = 0.0003  # 3 pips
        
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = sum([c for c in changes[-period:] if c > 0]) / period
        losses = sum([-c for c in changes[-period:] if c < 0]) / period
        
        if losses == 0:
            return 100 if gains > 0 else 50
        rs = gains / losses
        return 100 - (100 / (1 + rs))
    
    def detect_opportunity(self, symbol: str, price: float, bid: float, ask: float) -> Optional[Dict]:
        """Detect scalp opportunity"""
        # Store price
        if symbol not in self.tick_buffer:
            self.tick_buffer[symbol] = []
        self.tick_buffer[symbol].append(price)
        
        if len(self.tick_buffer[symbol]) > 50:
            self.tick_buffer[symbol] = self.tick_buffer[symbol][-50:]
        
        if len(self.tick_buffer[symbol]) < 20:
            return None
        
        prices = self.tick_buffer[symbol]
        rsi = self.calculate_rsi(prices)
        
        # Micro trend
        micro_trend = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] > 0 else 0
        
        # Check spread
        spread = (ask - bid) / bid if bid > 0 else 1
        if spread > self.spread_threshold:
            return None
        
        # Generate signals
        signal = None
        confidence = 0
        
        if rsi < 30 and micro_trend < -0.0001:
            signal = 'buy'
            confidence = 0.7
        elif rsi > 70 and micro_trend > 0.0001:
            signal = 'sell'
            confidence = 0.7
        elif abs(micro_trend) > 0.0003:
            signal = 'buy' if micro_trend > 0 else 'sell'
            confidence = 0.6
        
        if signal:
            return {
                'signal': signal,
                'confidence': confidence,
                'entry': price,
                'target': price * (1 + self.profit_target if signal == 'buy' else 1 - self.profit_target),
                'stop': price * (1 - self.stop_loss if signal == 'buy' else 1 + self.stop_loss),
                'rsi': rsi
            }
        return None

class TrinityScalper:
    """Main scalping daemon"""
    
    def __init__(self):
        logger.info("⚡ TRINITY SCALPER INITIALIZING...")
        
        self.creds = SimpleCredentialManager()
        self.brain = SimpleBrain()
        self.strategy = ScalpingStrategy()
        
        self.active_scalps = {}
        self.stats = {
            'total': 0, 'wins': 0, 'losses': 0,
            'total_pips': 0, 'best': 0, 'worst': 0
        }
        
        self.pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'SPY', 'QQQ']
        self.last_api_call = {}
        self.running = True
        
        logger.info("✅ TRINITY SCALPER READY")
        self.log_status()
    
    def log_status(self):
        """Log status"""
        logger.info("=" * 60)
        logger.info("⚡ SCALPING SYSTEM ONLINE")
        logger.info(f"  Pairs: {', '.join(self.pairs)}")
        logger.info(f"  Target: 5 pips | Stop: 3 pips")
        logger.info(f"  Brain: {'Redis' if self.brain.redis_available else 'Local'}")
        logger.info(f"  APIs: {list(self.creds.credentials.keys())}")
        logger.info("=" * 60)
    
    async def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current price"""
        try:
            import requests
            
            # Rate limiting
            api = 'alphavantage' if '_' in symbol else 'finnhub'
            cooldown = 150 if api == 'alphavantage' else 1
            
            now = time.time()
            if api in self.last_api_call:
                if now - self.last_api_call[api] < cooldown:
                    return None
            
            self.last_api_call[api] = now
            
            # Try real API
            if '_' in symbol and 'alphavantage' in self.creds.credentials:
                from_curr, to_curr = symbol.split('_')
                url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE"
                url += f"&from_currency={from_curr}&to_currency={to_curr}"
                url += f"&apikey={self.creds.get_credential('alphavantage', 'api_key')}"
                
                try:
                    resp = requests.get(url, timeout=2, proxies={'http': None, 'https': None})
                    if resp.status_code == 200:
                        data = resp.json()
                        if 'Realtime Currency Exchange Rate' in data:
                            rate = data['Realtime Currency Exchange Rate']
                            price = float(rate.get('5. Exchange Rate', 0))
                            if price > 0:
                                return {
                                    'price': price,
                                    'bid': price - 0.00005,
                                    'ask': price + 0.00005
                                }
                except:
                    pass
            
        except ImportError:
            pass
        
        # Simulate price
        base = {'EUR_USD': 1.10, 'GBP_USD': 1.25, 'USD_JPY': 145, 'SPY': 450, 'QQQ': 380}
        price = base.get(symbol, 100) * (1 + random.gauss(0, 0.0005))
        spread = 0.0001 if '_' in symbol else 0.01
        
        return {
            'price': price,
            'bid': price - spread/2,
            'ask': price + spread/2
        }
    
    async def manage_scalp(self, trade_id: str, info: Dict):
        """Manage open scalp"""
        tick = await self.get_price(info['symbol'])
        if not tick:
            return
        
        current = tick['price']
        entry = info['entry']
        
        # Check exit
        pnl = (current - entry if info['signal'] == 'buy' else entry - current) * 10000
        exit_reason = None
        
        if info['signal'] == 'buy':
            if current >= info['target']:
                exit_reason = 'target'
            elif current <= info['stop']:
                exit_reason = 'stop'
        else:
            if current <= info['target']:
                exit_reason = 'target'
            elif current >= info['stop']:
                exit_reason = 'stop'
        
        # Time exit
        if (datetime.now() - info['time']).seconds > 300:
            exit_reason = 'timeout'
        
        if exit_reason:
            del self.active_scalps[trade_id]
            self.stats['total'] += 1
            
            if pnl > 0:
                self.stats['wins'] += 1
            else:
                self.stats['losses'] += 1
            
            self.stats['total_pips'] += pnl
            self.stats['best'] = max(self.stats['best'], pnl)
            self.stats['worst'] = min(self.stats['worst'], pnl)
            
            logger.info(f"{'✅' if pnl > 0 else '❌'} {info['symbol']} closed: "
                       f"{pnl:.1f} pips ({exit_reason})")
            
            self.brain.learn_from_trade({
                'symbol': info['symbol'],
                'pnl': pnl,
                'reason': exit_reason
            })
    
    async def run_scalping(self):
        """Main loop"""
        logger.info("⚡ Starting scalping...")
        
        while self.running:
            try:
                # Check each pair
                for symbol in self.pairs:
                    tick = await self.get_price(symbol)
                    if not tick:
                        continue
                    
                    # Check opportunity
                    opp = self.strategy.detect_opportunity(
                        symbol, tick['price'], tick['bid'], tick['ask']
                    )
                    
                    if opp and len(self.active_scalps) < 3:
                        trade_id = f"S{int(time.time() * 1000)}"
                        self.active_scalps[trade_id] = {
                            'symbol': symbol,
                            'signal': opp['signal'],
                            'entry': opp['entry'],
                            'target': opp['target'],
                            'stop': opp['stop'],
                            'time': datetime.now()
                        }
                        
                        logger.info(f"⚡ NEW SCALP: {symbol} {opp['signal'].upper()} "
                                   f"@ {opp['entry']:.5f} RSI:{opp['rsi']:.0f}")
                    
                    await asyncio.sleep(0.1)
                
                # Manage open trades
                for tid, info in list(self.active_scalps.items()):
                    await self.manage_scalp(tid, info)
                
                # Log stats
                if self.stats['total'] > 0 and self.stats['total'] % 20 == 0:
                    self.log_stats()
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(5)
    
    def log_stats(self):
        """Log performance"""
        s = self.stats
        wr = (s['wins'] / s['total'] * 100) if s['total'] > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"⚡ PERFORMANCE: {s['total']} trades, {wr:.1f}% win rate")
        logger.info(f"  Total: {s['total_pips']:.1f} pips")
        logger.info(f"  Best: +{s['best']:.1f} | Worst: {s['worst']:.1f}")
        logger.info(f"  Active: {len(self.active_scalps)}")
        logger.info("=" * 60)
    
    def run(self):
        """Run daemon"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.run_scalping())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False
            self.log_stats()
            loop.close()

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     ⚡ TRINITY SCALPER - BLACK BOX SCALPING                 ║
    ║                                                              ║
    ║     5 pip targets | 3 pip stops | Paper trading             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    TrinityScalper().run()

if __name__ == "__main__":
    main()
