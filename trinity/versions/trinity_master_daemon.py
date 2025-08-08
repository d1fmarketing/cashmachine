#\!/usr/bin/env python3
"""
TRINITY MASTER DAEMON - Unified Black Box Trading Consciousness
ULTRATHINK: Zero humans, infinite intelligence, maximum security
"""

import os
import sys
import json
import time
import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import traceback

# Add Trinity path
sys.path.insert(0, '/opt/cashmachine/trinity')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/trinity-master.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TRINITY_MASTER')

# Import all Trinity components
try:
    from trinity_paper_trader import TrinityPaperTrader, CredentialManager
    from trinity_unified_brain import UnifiedBrain
    from trinity_training_mode import RateLimitedDataCollector
    logger.info("✅ All Trinity components imported")
except Exception as e:
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)

class TrinityMasterDaemon:
    """Master daemon orchestrating all Trinity components"""
    
    def __init__(self):
        logger.info("🧠 TRINITY MASTER DAEMON INITIALIZING...")
        
        # Core components
        self.credential_manager = CredentialManager()
        self.unified_brain = UnifiedBrain(redis_host='10.100.2.200')
        self.paper_trader = TrinityPaperTrader()
        self.data_collector = RateLimitedDataCollector(self.credential_manager)
        
        # Control flags
        self.running = True
        self.shutdown_event = threading.Event()
        
        # Trading state
        self.trades_today = 0
        self.daily_pnl = 0
        self.total_pnl = 0
        self.generation = 0
        
        # Rate limiting
        self.api_calls = {
            'alphavantage': {'count': 0, 'reset_time': datetime.now() + timedelta(days=1)},
            'finnhub': {'count': 0, 'reset_time': datetime.now() + timedelta(minutes=1)},
            'polygon': {'count': 0, 'reset_time': datetime.now() + timedelta(minutes=1)}
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        
        logger.info("✅ TRINITY MASTER DAEMON READY")
        self.log_status()
    
    def log_status(self):
        """Log system status"""
        logger.info("=" * 60)
        logger.info("📊 BLACK BOX TRADING SYSTEM STATUS")
        logger.info(f"  Redis Brain: {'✅ Connected' if self.unified_brain.redis_available else '❌ Local only'}")
        logger.info(f"  APIs Loaded: {list(self.credential_manager.credentials.keys())}")
        logger.info(f"  Paper Trading: ✅ ACTIVE")
        logger.info(f"  Security: 🔐 BLACK BOX MODE")
        logger.info(f"  Internet: Via NAT Gateway only (54.144.136.132)")
        logger.info("=" * 60)
    
    def check_rate_limits(self, api: str) -> bool:
        """Check if we can make API call"""
        if api not in self.api_calls:
            return True
        
        now = datetime.now()
        limits = self.api_calls[api]
        
        # Reset if needed
        if now >= limits['reset_time']:
            limits['count'] = 0
            if api == 'alphavantage':
                limits['reset_time'] = now + timedelta(days=1)
            else:
                limits['reset_time'] = now + timedelta(minutes=1)
        
        # Check limits
        max_calls = {'alphavantage': 25, 'finnhub': 60, 'polygon': 5}
        if limits['count'] >= max_calls.get(api, 100):
            logger.warning(f"⚠️ Rate limit reached for {api}")
            return False
        
        limits['count'] += 1
        return True
    
    async def trading_loop(self):
        """Main trading loop"""
        symbols = ['EUR_USD', 'SPY', 'AAPL', 'BTC_USD', 'ETH_USD']
        
        while self.running:
            try:
                # Check market status
                market_open = await self.check_markets()
                
                if not market_open:
                    logger.info("📊 Markets closed, running simulations...")
                
                # Cycle through symbols
                for symbol in symbols:
                    if not self.running:
                        break
                    
                    # Get market data with rate limiting
                    data = await self.get_market_data(symbol)
                    if not data:
                        continue
                    
                    # Share data with brain
                    self.unified_brain.share_market_data(symbol, data)
                    
                    # Get consensus decision
                    decision = self.unified_brain.get_consensus_decision(symbol)
                    
                    if decision and decision['confidence'] > 0.5:
                        # Execute paper trade
                        result = await self.execute_paper_trade(symbol, decision)
                        
                        if result:
                            # Learn from result
                            self.unified_brain.learn_from_trade(result)
                            self.trades_today += 1
                            
                            logger.info(f"📈 Trade: {symbol} {decision['action']} "
                                      f"Confidence: {decision['confidence']:.2f} "
                                      f"P&L: ${result.get('pnl', 0):.2f}")
                    
                    # Small delay between symbols
                    await asyncio.sleep(1)
                
                # Longer delay between cycles
                await asyncio.sleep(30)
                
                # Periodic status update
                if self.trades_today % 10 == 0 and self.trades_today > 0:
                    self.log_performance()
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)
    
    async def check_markets(self) -> bool:
        """Check if markets are open"""
        now = datetime.now()
        
        # Forex: 24/5 (Sunday 5pm - Friday 5pm EST)
        if now.weekday() < 5:  # Monday-Friday
            return True
        elif now.weekday() == 6 and now.hour >= 17:  # Sunday after 5pm
            return True
        
        return False
    
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data with rate limiting"""
        try:
            # Determine which API to use
            if '_' in symbol:  # Forex
                if self.check_rate_limits('alphavantage'):
                    return await self.data_collector.get_forex_data(symbol)
            elif symbol in ['BTC_USD', 'ETH_USD']:  # Crypto
                return await self.data_collector.get_crypto_data(symbol)
            else:  # Stocks
                if self.check_rate_limits('finnhub'):
                    return await self.data_collector.get_stock_data(symbol)
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
        
        return None
    
    async def execute_paper_trade(self, symbol: str, decision: Dict) -> Optional[Dict]:
        """Execute paper trade"""
        try:
            return self.paper_trader.execute_trade({
                'symbol': symbol,
                'signal': decision['action'],
                'confidence': decision['confidence']
            })
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    def log_performance(self):
        """Log trading performance"""
        logger.info("=" * 60)
        logger.info("📊 PERFORMANCE UPDATE")
        logger.info(f"  Trades Today: {self.trades_today}")
        logger.info(f"  Daily P&L: ${self.daily_pnl:.2f}")
        logger.info(f"  Total P&L: ${self.total_pnl:.2f}")
        logger.info(f"  Generation: {self.generation}")
        logger.info(f"  Patterns Learned: {len(self.unified_brain.get_all_patterns())}")
        logger.info("=" * 60)
    
    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        logger.info("🛑 Shutdown signal received")
        self.running = False
        self.shutdown_event.set()
        
        # Save brain state
        try:
            state = {
                'generation': self.generation,
                'total_pnl': self.total_pnl,
                'patterns': self.unified_brain.get_all_patterns()
            }
            self.unified_brain.save_state('master_daemon', state)
            logger.info("✅ Brain state saved")
        except:
            pass
        
        logger.info("👋 Trinity Master Daemon shutdown complete")
        sys.exit(0)
    
    def run(self):
        """Main run method"""
        logger.info("🚀 Starting Trinity Master Daemon")
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run trading loop
            loop.run_until_complete(self.trading_loop())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            loop.close()
            self.shutdown()

def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     🧠 TRINITY MASTER DAEMON - BLACK BOX TRADING            ║
    ║                                                              ║
    ║     Unified Consciousness | Real APIs | Paper Trading       ║
    ║     Redis Brain | Rate Limiting | Maximum Security          ║
    ║                                                              ║
    ║     "Zero humans, infinite intelligence"                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    daemon = TrinityMasterDaemon()
    daemon.run()

if __name__ == "__main__":
    main()
