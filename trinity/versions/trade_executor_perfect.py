#!/usr/bin/env python3
"""
PERFECT TRADE EXECUTOR
Maximum trading frequency with sacred timing
"""

import asyncio
import redis.asyncio as redis
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerfectExecutor:
    def __init__(self):
        self.PI = 3.14159265359
        self.PHI = 1.618033988749
        self.SACRED_69 = 69
        
        self.redis_client = None
        self.trading_enabled = True
        self.executed_trades = 0
        self.last_trade_time = 0
        self.min_trade_interval = 15  # Ultra fast - 15 seconds
        
        logger.info("💰 Perfect Trade Executor initialized")
    
    async def setup_redis(self):
        try:
            self.redis_client = await redis.Redis(
                host='10.100.2.200',
                port=6379,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Connected to Redis")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
    
    async def get_signal(self):
        """Get trading signal from ULTRATHINK"""
        try:
            # Try ULTRATHINK signals first
            ultra = await self.redis_client.hgetall('ultrathink:signals')
            if ultra and 'signal' in ultra:
                return {
                    'source': 'ULTRATHINK',
                    'signal': ultra.get('signal'),
                    'confidence': float(ultra.get('confidence', 0)),
                    'components': {
                        'asi': ultra.get('asi', ''),
                        'hrm': ultra.get('hrm', ''),
                        'mcts': ultra.get('mcts', '')
                    }
                }
            
            # Fallback to ML Farm
            ml = await self.redis_client.hgetall('ml_farm:unified')
            if ml and 'signal' in ml:
                return {
                    'source': 'ML_FARM',
                    'signal': ml.get('signal'),
                    'confidence': float(ml.get('confidence', 0)),
                    'sacred': float(ml.get('sacred_alignment', 0))
                }
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
        
        return None
    
    async def should_execute(self, signal_data):
        """Determine if trade should be executed"""
        if not signal_data:
            return False
        
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        
        # Don't trade holds
        if signal == 'hold':
            return False
        
        # Ultra low threshold - 15%
        if confidence < 0.15:
            return False
        
        # Check minimum interval
        current_time = time.time()
        if current_time - self.last_trade_time < self.min_trade_interval:
            return False
        
        # Sacred timing boost
        current_second = int(current_time) % 100
        
        # Always execute on sacred moments
        if current_second == 31:  # Pi moment
            logger.info("✨ PI MOMENT - EXECUTING!")
            return True
        elif current_second == 69:  # Sacred 69
            logger.info("✨ SACRED 69 - EXECUTING!")
            return True
        elif current_second % 10 == 0:  # Every 10 seconds
            if confidence > 0.25:
                logger.info("✨ 10-second mark - EXECUTING!")
                return True
        
        # Normal execution
        return confidence > 0.2
    
    async def execute_trade(self, signal_data):
        """Execute the trade"""
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        source = signal_data['source']
        
        self.executed_trades += 1
        self.last_trade_time = time.time()
        
        # Sacred position sizing
        position_size = 1000  # Base size
        current_second = int(time.time()) % 100
        
        if current_second == 69:
            position_size *= self.PHI  # Golden ratio boost
            logger.info(f"🔥 SACRED 69 BOOST: ${position_size:.2f}")
        elif current_second == 31:
            position_size *= (self.PI / 2)  # Pi boost
            logger.info(f"🔥 PI BOOST: ${position_size:.2f}")
        
        logger.info(f"🎯 EXECUTING #{self.executed_trades}: {signal.upper()} @ {confidence:.2%} (${position_size:.2f}) via {source}")
        
        # Store in Redis
        try:
            trade_key = f"executed:trade:{int(self.last_trade_time)}"
            await self.redis_client.hset(trade_key, mapping={
                'signal': signal,
                'confidence': str(confidence),
                'position_size': str(position_size),
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'trade_number': str(self.executed_trades)
            })
            
            # Update executor status
            await self.redis_client.hset('ultrathink:executor', mapping={
                'status': 'ACTIVE',
                'last_trade': signal,
                'last_confidence': str(confidence),
                'total_trades': str(self.executed_trades),
                'position_size': str(position_size),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Redis storage error: {e}")
    
    async def run(self):
        """Main execution loop"""
        if not await self.setup_redis():
            logger.error("Cannot start without Redis")
            return
        
        logger.info("🚀 Starting Perfect Trade Executor")
        logger.info(f"⚡ Min interval: {self.min_trade_interval}s")
        logger.info(f"📊 Confidence threshold: 15%")
        logger.info(f"✨ Sacred timing: ACTIVE")
        
        while self.trading_enabled:
            try:
                # Get signal
                signal_data = await self.get_signal()
                
                if signal_data:
                    # Log current signal
                    logger.debug(f"Signal: {signal_data['signal']} @ {signal_data['confidence']:.2%}")
                    
                    # Check if should execute
                    if await self.should_execute(signal_data):
                        await self.execute_trade(signal_data)
                    else:
                        # Log why not executing
                        time_left = self.min_trade_interval - (time.time() - self.last_trade_time)
                        if time_left > 0:
                            logger.debug(f"Waiting {time_left:.0f}s before next trade")
                
                # Fast loop - check every 5 seconds
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(5)

async def main():
    executor = PerfectExecutor()
    await executor.run()

if __name__ == "__main__":
    asyncio.run(main())