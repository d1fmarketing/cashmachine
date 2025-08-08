#\!/usr/bin/env python3
"""
TRINITY UNIFIED - All instances share one consciousness
Paper trading with collective intelligence through Redis
"""

import os
import sys
import time
import json
import asyncio
import logging
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List
from cryptography.fernet import Fernet

# Add brain connector
sys.path.insert(0, '/opt/cashmachine/trinity')
from trinity_unified_brain import BrainConnector, UnifiedBrain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/trinity_unified.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TRINITY_UNIFIED')

class UnifiedPaperTrader:
    """Paper trader connected to collective consciousness"""
    
    def __init__(self):
        # Connect to unified brain
        self.brain_connector = BrainConnector('paper_trader')
        self.brain = self.brain_connector.brain
        
        # Load API credentials
        self.oanda_creds = self.load_credentials('oanda')
        self.alpaca_creds = self.load_credentials('alpaca')
        
        # Assets to trade
        self.forex_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        self.stocks = ['SPY', 'AAPL', 'TSLA', 'GOOGL']
        
        # Start heartbeat
        self.brain_connector.heartbeat_loop()
        
        logger.info("🧠 Unified Paper Trader connected to collective consciousness")
        
        # Show current intelligence
        intel = self.brain.get_collective_intelligence()
        logger.info(f"📊 Collective State: Gen {intel['generation']}, {intel['total_trades']} trades, {intel['win_rate']:.1%} win rate")
    
    def load_credentials(self, api_name):
        """Load encrypted credentials"""
        try:
            with open(f'/opt/cashmachine/config/.{api_name}.key', 'rb') as f:
                key = f.read()
            with open(f'/opt/cashmachine/config/{api_name}.enc', 'rb') as f:
                encrypted = f.read()
            cipher = Fernet(key)
            return json.loads(cipher.decrypt(encrypted))
        except:
            return None
    
    async def trading_loop(self):
        """Main trading loop using collective intelligence"""
        logger.info("🔥 Starting unified trading loop...")
        
        while True:
            try:
                # Trade forex
                for pair in self.forex_pairs:
                    market_data = await self.get_market_data(pair, 'forex')
                    
                    # Get decision from collective intelligence
                    decision = self.brain_connector.get_trading_decision(market_data)
                    
                    if decision['action'] != 'hold':
                        # Execute paper trade
                        result = await self.execute_trade(pair, decision, 'oanda')
                        
                        if result:
                            # Share learning with all instances
                            self.brain_connector.learn_from_trade(result)
                
                # Trade stocks
                for stock in self.stocks:
                    market_data = await self.get_market_data(stock, 'stocks')
                    
                    # Get collective decision
                    decision = self.brain_connector.get_trading_decision(market_data)
                    
                    if decision['action'] != 'hold':
                        # Execute paper trade
                        result = await self.execute_trade(stock, decision, 'alpaca')
                        
                        if result:
                            # Share with collective
                            self.brain_connector.learn_from_trade(result)
                
                # Check collective intelligence periodically
                if self.brain.get_generation() > 0:
                    intel = self.brain.get_collective_intelligence()
                    if intel['total_trades'] % 100 == 0:
                        logger.info(f"🧬 Collective Evolution: Gen {intel['generation']}, Win Rate: {intel['win_rate']:.1%}")
                
                await asyncio.sleep(10)  # Trade every 10 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)
    
    async def get_market_data(self, symbol: str, asset_type: str) -> Dict:
        """Get market data"""
        # Simplified - would use real APIs
        price = 100 * (1 + np.random.randn() * 0.01)
        return {
            'symbol': symbol,
            'price': price,
            'asset_type': asset_type,
            'timestamp': datetime.now().isoformat()
        }
    
    async def execute_trade(self, symbol: str, decision: Dict, platform: str) -> Dict:
        """Execute trade and return result"""
        # Check if we can trade (not duplicate)
        if self.brain.check_duplicate_position(symbol, platform):
            return None
        
        # Register position with brain
        position = {
            'symbol': symbol,
            'platform': platform,
            'action': decision['action'],
            'entry_price': 100,  # Would be real price
            'units': 100 if platform == 'oanda' else 1,
            'timestamp': datetime.now().isoformat()
        }
        position_id = self.brain.register_position(position)
        
        # Simulate execution (would be real API call)
        logger.info(f"📊 UNIFIED TRADE: {symbol} {decision['action']} via {platform} (confidence: {decision['confidence']:.1%})")
        
        # Simulate P&L
        pnl = np.random.randn() * 10
        
        # Close position and record P&L
        self.brain.close_position(position_id, pnl)
        
        return {
            'symbol': symbol,
            'action': decision['action'],
            'platform': platform,
            'pnl': pnl,
            'confidence': decision['confidence'],
            'strategy': decision.get('strategy', 'collective'),
            'timestamp': datetime.now().isoformat()
        }
    
    async def monitor_collective(self):
        """Monitor collective intelligence"""
        while True:
            await asyncio.sleep(60)  # Every minute
            
            intel = self.brain.get_collective_intelligence()
            active = self.brain.get_active_instances()
            
            logger.info(f"""
📊 UNIFIED CONSCIOUSNESS STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧬 Generation: {intel['generation']}
📈 Total Trades: {intel['total_trades']}
✅ Win Rate: {intel['win_rate']:.1%}
💰 Collective P&L: ${intel['total_pnl']:.2f}
📚 Patterns: {intel['pattern_count']}
🎯 Best Strategy: {intel['best_strategy']}
🔗 Active Instances: {len(active)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """)

async def main():
    """Run unified trader"""
    trader = UnifiedPaperTrader()
    
    tasks = [
        asyncio.create_task(trader.trading_loop()),
        asyncio.create_task(trader.monitor_collective())
    ]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     🧠 TRINITY UNIFIED - ONE CONSCIOUSNESS                   ║
    ║                                                              ║
    ║     • Connected to Redis shared brain                       ║
    ║     • Learning shared across ALL instances                  ║
    ║     • Collective decision making                            ║
    ║     • No duplicate positions                                ║
    ║     • Unified P&L tracking                                  ║
    ║                                                              ║
    ║     "We think as one, we trade as one"                      ║
    ║                                                              ║
    ║     ULTRATHINK: True collective intelligence\!               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Unified trader stopped")
