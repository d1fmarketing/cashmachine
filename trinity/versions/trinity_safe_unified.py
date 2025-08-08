#\!/usr/bin/env python3
"""
SAFER UNIFIED TRINITY - With local fallback and conflict prevention
Each instance can operate independently if Redis fails
"""

import redis
import json
import time
import logging
import pickle
import os
from typing import Dict, Optional, Any
from threading import Lock

logger = logging.getLogger('SAFE_TRINITY')

class SafeUnifiedBrain:
    """Safer brain with local caching and fallback"""
    
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.lock = Lock()
        self.local_cache = {}
        self.cache_file = f'/tmp/trinity_cache_{instance_id}.pkl'
        
        # Try Redis connection
        self.redis_available = False
        self.redis = None
        self.connect_redis()
        
        # Load local cache
        self.load_local_cache()
        
        logger.info(f"🛡️ Safe Brain initialized: {instance_id}")
        logger.info(f"  Redis: {'✅ Connected' if self.redis_available else '❌ Using local cache'}")
    
    def connect_redis(self):
        """Try to connect to Redis with fallback"""
        try:
            self.redis = redis.Redis(
                host='10.100.2.200',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=2
            )
            self.redis.ping()
            self.redis_available = True
        except:
            self.redis_available = False
            logger.warning("Redis unavailable - using local mode")
    
    def load_local_cache(self):
        """Load local cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.local_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.local_cache)} items from local cache")
            except:
                self.local_cache = {}
    
    def save_local_cache(self):
        """Save local cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.local_cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def safe_set(self, key: str, value: Any) -> bool:
        """Safely set value with local fallback"""
        with self.lock:
            # Always update local cache
            self.local_cache[key] = value
            self.save_local_cache()
            
            # Try to update Redis
            if self.redis_available:
                try:
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    self.redis.set(key, value)
                    return True
                except:
                    self.redis_available = False
                    logger.warning("Redis write failed - using local only")
            
            return False
    
    def safe_get(self, key: str) -> Any:
        """Safely get value with local fallback"""
        # Try Redis first
        if self.redis_available:
            try:
                value = self.redis.get(key)
                if value:
                    # Update local cache
                    self.local_cache[key] = value
                    return value
            except:
                self.redis_available = False
        
        # Fallback to local cache
        return self.local_cache.get(key)
    
    def safe_increment(self, key: str, amount: int = 1) -> int:
        """Safely increment counter"""
        with self.lock:
            current = int(self.safe_get(key) or 0)
            new_value = current + amount
            self.safe_set(key, new_value)
            return new_value
    
    def record_trade_safe(self, trade: Dict) -> bool:
        """Record trade with conflict prevention"""
        trade_id = f"trade_{self.instance_id}_{int(time.time() * 1000)}"
        
        # Add to local history
        if 'trades' not in self.local_cache:
            self.local_cache['trades'] = []
        self.local_cache['trades'].append(trade)
        
        # Keep only last 1000 trades locally
        if len(self.local_cache['trades']) > 1000:
            self.local_cache['trades'] = self.local_cache['trades'][-1000:]
        
        # Try to sync with Redis
        if self.redis_available:
            try:
                # Use Redis transaction for safety
                pipe = self.redis.pipeline()
                pipe.hset(f'trinity:trade:{trade_id}', mapping=trade)
                pipe.hincrby('trinity:metrics', 'total_trades', 1)
                if trade.get('pnl', 0) > 0:
                    pipe.hincrby('trinity:metrics', 'wins', 1)
                pipe.execute()
                return True
            except:
                self.redis_available = False
        
        return False
    
    def get_safe_metrics(self) -> Dict:
        """Get metrics with fallback to local"""
        metrics = {
            'total_trades': 0,
            'wins': 0,
            'generation': 0,
            'redis_status': self.redis_available
        }
        
        # Try Redis
        if self.redis_available:
            try:
                redis_metrics = self.redis.hgetall('trinity:metrics')
                metrics.update(redis_metrics)
            except:
                pass
        
        # Merge with local
        local_trades = len(self.local_cache.get('trades', []))
        local_wins = sum(1 for t in self.local_cache.get('trades', []) if t.get('pnl', 0) > 0)
        
        metrics['local_trades'] = local_trades
        metrics['local_wins'] = local_wins
        metrics['win_rate'] = local_wins / local_trades if local_trades > 0 else 0
        
        return metrics
    
    def sync_with_redis(self):
        """Periodically try to sync with Redis"""
        if not self.redis_available:
            self.connect_redis()
            
            if self.redis_available:
                logger.info("Redis reconnected - syncing local data")
                # Sync local trades to Redis
                for trade in self.local_cache.get('trades', []):
                    self.record_trade_safe(trade)


# Test the safer approach
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     🛡️ SAFER UNIFIED TRINITY - With Fallback                 ║
    ║                                                              ║
    ║     • Local cache for each instance                         ║
    ║     • Can operate without Redis                             ║
    ║     • Automatic failover and recovery                       ║
    ║     • Conflict prevention with locks                        ║
    ║     • Data persistence to disk                              ║
    ║                                                              ║
    ║     "Safe, resilient, unified intelligence"                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Test with and without Redis
    brain = SafeUnifiedBrain("test_instance")
    
    # Test operations
    brain.safe_set('test_key', 'test_value')
    print(f"Test get: {brain.safe_get('test_key')}")
    
    # Test trade recording
    brain.record_trade_safe({
        'symbol': 'EUR_USD',
        'action': 'buy',
        'pnl': 10.5,
        'timestamp': time.time()
    })
    
    # Get metrics
    metrics = brain.get_safe_metrics()
    print("\n📊 Safe Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Safer unified system ready\!")
