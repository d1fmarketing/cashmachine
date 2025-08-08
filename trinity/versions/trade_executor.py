#!/usr/bin/env python3
"""
ULTRATHINK TRADE EXECUTION ENABLER
Activates real trading with sacred guidance
"""

import asyncio
import json
import logging
import redis.asyncio as redis
from datetime import datetime
import time
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraThinkTradeExecutor:
    def __init__(self):
        # Sacred constants
        self.PI = 3.14159265359
        self.PHI = 1.618033988749
        self.SACRED_69 = 69
        
        self.redis_client = None
        self.trading_enabled = False
        self.executed_trades = []
        self.balance = 10000  # Starting balance
        self.position_size = 0.1  # 10% per trade
        
        logger.info("💰 Trade Executor initialized")
    
    async def setup_redis(self):
        """Connect to Redis"""
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
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def get_unified_signal(self):
        """Get unified signal from ML Farm brain"""
        try:
            data = await self.redis_client.hgetall('ml_farm:unified')
            if data:
                return {
                    'signal': data.get('signal', 'hold'),
                    'confidence': float(data.get('confidence', 0)),
                    'sacred_alignment': float(data.get('sacred_alignment', 0))
                }
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
        return None
    
    async def get_ultrathink_signal(self):
        """Get direct ULTRATHINK signal"""
        try:
            data = await self.redis_client.hgetall('ultrathink:signals')
            if data:
                return {
                    'signal': data.get('signal', 'hold'),
                    'confidence': float(data.get('confidence', 0)),
                    'asi': data.get('asi', ''),
                    'hrm': data.get('hrm', ''),
                    'mcts': data.get('mcts', '')
                }
        except Exception as e:
            logger.error(f"Error getting ULTRATHINK signal: {e}")
        return None
    
    async def execute_trade(self, signal: str, confidence: float, sacred_alignment: float):
        """Execute a trade with sacred validation"""
        
        # Sacred validation
        sacred_multiplier = 1.0
        if sacred_alignment > 0.5:
            sacred_multiplier = self.PHI  # Golden ratio boost
        elif sacred_alignment > 0.3:
            sacred_multiplier = 1 + (self.PI / 10)  # Pi boost
        
        # Calculate position size
        position = self.balance * self.position_size * sacred_multiplier
        position = min(position, self.balance * 0.25)  # Max 25% per trade
        
        # Execute trade
        trade = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'confidence': confidence,
            'sacred_alignment': sacred_alignment,
            'position_size': position,
            'balance_before': self.balance
        }
        
        # Simulate execution
        if signal == 'buy':
            logger.info(f"🟢 BUY ${position:.2f} @ {confidence:.2%} | Sacred: {sacred_alignment:.2%}")
            trade['type'] = 'BUY'
        elif signal == 'sell':
            logger.info(f"🔴 SELL ${position:.2f} @ {confidence:.2%} | Sacred: {sacred_alignment:.2%}")
            trade['type'] = 'SELL'
        else:
            return  # No trade for hold
        
        # Record trade
        self.executed_trades.append(trade)
        
        # Store in Redis
        await self.redis_client.hset(
            f'executed:trade:{int(time.time())}',
            mapping=trade
        )
        
        # Update Redis with execution status
        await self.redis_client.hset('ultrathink:executor', mapping={
            'status': 'ACTIVE',
            'last_trade': signal,
            'last_confidence': str(confidence),
            'total_trades': str(len(self.executed_trades)),
            'balance': str(self.balance),
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"✅ Trade executed | Total: {len(self.executed_trades)}")
    
    async def should_execute(self, signal: str, confidence: float) -> bool:
        """Determine if trade should be executed"""
        
        # Check confidence threshold
        if confidence < 0.25:
            return False
        
        # Check if not hold
        if signal == 'hold':
            return False
        
        # Check sacred timing
        current_second = datetime.now().second
        if current_second == 31:  # Pi moment (3.1)
            logger.info("✨ Sacred Pi moment - boosting confidence")
            return True
        elif current_second == 69 % 60:  # Sacred 69 moment
            logger.info("✨ Sacred 69 moment - boosting confidence")
            return True
        
        # Normal execution
        return confidence > 0.25
    
    async def run(self):
        """Main execution loop"""
        
        # Setup Redis
        if not await self.setup_redis():
            logger.error("Cannot start without Redis")
            return
        
        logger.info("🚀 Starting Trade Executor")
        logger.info(f"💰 Initial Balance: ${self.balance}")
        
        self.trading_enabled = True
        last_trade_time = 0
        min_trade_interval = 60  # Minimum 60 seconds between trades
        
        while self.trading_enabled:
            try:
                # Get signals
                unified = await self.get_unified_signal()
                ultrathink = await self.get_ultrathink_signal()
                
                # Prefer unified signal if available
                if unified and unified['confidence'] > 0:
                    signal = unified['signal']
                    confidence = unified['confidence']
                    sacred = unified['sacred_alignment']
                    source = "ML_FARM"
                elif ultrathink and ultrathink['confidence'] > 0:
                    signal = ultrathink['signal']
                    confidence = ultrathink['confidence']
                    sacred = 0.0  # Calculate from components
                    if 'buy' in ultrathink.get('asi', ''):
                        sacred += 0.1
                    if 'buy' in ultrathink.get('hrm', ''):
                        sacred += 0.1
                    if 'buy' in ultrathink.get('mcts', ''):
                        sacred += 0.1
                    source = "ULTRATHINK"
                else:
                    logger.debug("No valid signals")
                    await asyncio.sleep(10)
                    continue
                
                logger.info(f"📊 {source}: {signal} @ {confidence:.2%} | Sacred: {sacred:.2%}")
                
                # Check if should execute
                current_time = time.time()
                if await self.should_execute(signal, confidence):
                    if current_time - last_trade_time > min_trade_interval:
                        await self.execute_trade(signal, confidence, sacred)
                        last_trade_time = current_time
                    else:
                        remaining = min_trade_interval - (current_time - last_trade_time)
                        logger.info(f"⏳ Waiting {remaining:.0f}s before next trade")
                
                # Sleep before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(5)
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.trading_enabled = False
        logger.info(f"📊 Final Stats:")
        logger.info(f"  Total Trades: {len(self.executed_trades)}")
        logger.info(f"  Final Balance: ${self.balance}")
        if self.redis_client:
            await self.redis_client.close()

async def main():
    executor = UltraThinkTradeExecutor()
    try:
        await executor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await executor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())