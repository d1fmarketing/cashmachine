
import redis
import json
import time
import threading
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class UltrathinkRedisIntegration:
    def __init__(self):
        # Redis connection with proper socket settings
        self.redis_client = redis.Redis(
            host='10.100.2.200',
            port=6379,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
            
        # Channels for brain communication
        self.CHANNELS = {
            'market_data': 'ultrathink:market:data',
            'asi_signals': 'ultrathink:asi:signals',
            'hrm_signals': 'ultrathink:hrm:signals',
            'mcts_signals': 'ultrathink:mcts:signals',
            'brain_commands': 'ultrathink:brain:commands',
            'trade_signals': 'ultrathink:trade:signals',
            'heartbeat': 'ultrathink:heartbeat',
        }
        
        self.running = True
        
    def publish_signal(self, channel: str, data: Dict[str, Any]):
        """Publish signal to Redis for all systems to see"""
        if not self.redis_client:
            return
            
        try:
            message = {
                'timestamp': datetime.now().isoformat(),
                'source': 'trinity_main',
                'data': data
            }
            self.redis_client.publish(self.CHANNELS.get(channel, channel), json.dumps(message))
            logger.debug(f"Published to {channel}: {data.get('symbol', '')} {data.get('signal', '')}")
        except Exception as e:
            logger.debug(f"Redis publish error: {e}")
            
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get latest market data from Redis cache"""
        if not self.redis_client:
            return {}
            
        try:
            # Check multiple keys for data
            keys = [
                f'market_data:{symbol}',
                f'ultrathink:data:{symbol}',
                f'price:{symbol}'
            ]
            
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    try:
                        return json.loads(data)
                    except:
                        return {'price': float(data)}
                    
            # Try hash
            hash_data = self.redis_client.hgetall(f'market:{symbol}')
            if hash_data:
                return hash_data
                
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
            
        return {}
        
    def store_trade_history(self, trade: Dict[str, Any]):
        """Store trade in Redis for learning"""
        if not self.redis_client:
            return
            
        try:
            # Add to sorted set by timestamp
            score = time.time()
            self.redis_client.zadd('ultrathink:trades', {json.dumps(trade): score})
            
            # Update stats
            self.redis_client.hincrby('ultrathink:stats', 'total_trades', 1)
            if trade.get('profit', 0) > 0:
                self.redis_client.hincrby('ultrathink:stats', 'winning_trades', 1)
            else:
                self.redis_client.hincrby('ultrathink:stats', 'losing_trades', 1)
                
        except Exception as e:
            logger.debug(f"Store trade error: {e}")
            
    def heartbeat(self):
        """Send heartbeat to show we're alive"""
        while self.running:
            try:
                if self.redis_client:
                    self.redis_client.setex(
                        'ultrathink:trinity:heartbeat',
                        60,
                        json.dumps({
                            'timestamp': datetime.now().isoformat(),
                            'status': 'alive',
                            'instance': 'trinity_main'
                        })
                    )
            except:
                pass
            time.sleep(30)
            
    def start_heartbeat(self):
        """Start heartbeat thread"""
        thread = threading.Thread(target=self.heartbeat, daemon=True)
        thread.start()
        
    def close(self):
        """Clean shutdown"""
        self.running = False
        if self.redis_client:
            self.redis_client.close()
