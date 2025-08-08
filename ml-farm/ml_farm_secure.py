#!/usr/bin/env python3
"""
ML FARM SACRED BRAIN - SECURE VERSION
Central intelligence coordinating all ULTRATHINK components
Uses configuration management instead of hard-coded values
"""

import asyncio
import json
import logging
import numpy as np
import redis.asyncio as redis
from datetime import datetime
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import get_config

# Setup logging
config = get_config()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ML_FARM_SECURE - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecureMLFarmBrain:
    def __init__(self):
        # Load configuration
        self.config = get_config()
        
        # Sacred constants
        self.PI = 3.14159265359
        self.PHI = 1.618033988749
        self.SACRED_69 = 69
        
        # Redis connections
        self.redis_client = None
        
        # Component tracking - loaded from configuration
        self.components = {
            'trinity': {
                'host': self.config.network.trinity_host,
                'status': 'unknown',
                'last_signal': None
            },
            'data_collector': {
                'host': self.config.network.data_collector_host,
                'status': 'unknown',
                'last_update': None
            },
            'ml_farm': {
                'host': self.config.network.ml_farm_host,
                'status': 'active',
                'signal': 0.0
            },
            'bridge': {
                'host': self.config.network.bridge_host,
                'status': 'unknown'
            },
            'proxy': {
                'host': self.config.network.proxy_host,
                'status': 'unknown'
            },
            'superset': {
                'host': self.config.network.superset_host,
                'status': 'unknown'
            },
            'redis_cache': {
                'host': self.config.network.redis_host,
                'status': 'unknown'
            }
        }
        
        # Unified decision state
        self.unified_signal = 'hold'
        self.unified_confidence = 0.0
        self.sacred_alignment = 0.0
        
        # Trading parameters from config
        self.min_confidence = self.config.trading.min_confidence_threshold
        self.target_buy_ratio = self.config.trading.target_buy_ratio
        self.target_sell_ratio = self.config.trading.target_sell_ratio
        self.target_hold_ratio = self.config.trading.target_hold_ratio
        
        logger.info("🧠 Secure ML Farm Sacred Brain initialized")
        logger.info(f"📂 Configuration loaded from: {self.config.env_file}")
    
    async def setup_redis(self):
        """Connect to Redis cache using configuration"""
        try:
            self.redis_client = await redis.Redis(
                host=self.config.network.redis_host,
                port=self.config.network.redis_port,
                db=self.config.network.redis_db,
                password=self.config.network.redis_password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            await self.redis_client.ping()
            self.components['redis_cache']['status'] = 'connected'
            logger.info(f"✅ Connected to Redis cache at {self.config.network.redis_host}:{self.config.network.redis_port}")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def collect_signals(self):
        """Collect signals from all components"""
        signals = {}
        
        try:
            # Get ULTRATHINK signals from Trinity
            ultra_data = await self.redis_client.hgetall('ultrathink:signals')
            if ultra_data:
                signals['trinity'] = {
                    'signal': ultra_data.get('signal', 'hold'),
                    'confidence': float(ultra_data.get('confidence', 0)),
                    'asi': ultra_data.get('asi', ''),
                    'hrm': ultra_data.get('hrm', ''),
                    'mcts': ultra_data.get('mcts', ''),
                    'timestamp': ultra_data.get('timestamp', '')
                }
                self.components['trinity']['status'] = 'active'
                self.components['trinity']['last_signal'] = signals['trinity']['signal']
            
            # Get market data from collector
            market_data = await self.redis_client.hgetall('market:latest')
            if market_data:
                signals['market'] = market_data
                self.components['data_collector']['status'] = 'active'
                self.components['data_collector']['last_update'] = datetime.now().isoformat()
            
            # Get any other component signals
            for component in ['bridge', 'proxy', 'superset']:
                comp_data = await self.redis_client.hgetall(f'{component}:status')
                if comp_data:
                    signals[component] = comp_data
                    self.components[component]['status'] = 'active'
            
        except Exception as e:
            logger.error(f"Error collecting signals: {e}")
        
        return signals
    
    def calculate_sacred_geometry(self, signals: Dict) -> float:
        """Calculate sacred alignment using sacred numbers"""
        values = []
        
        # Extract numerical values from signals
        for component, data in signals.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    try:
                        num_val = float(value)
                        values.append(num_val)
                    except:
                        pass
        
        if not values:
            return 0.0
        
        # Apply sacred transformations
        mean_val = np.mean(values)
        sacred_transform = (mean_val * self.PHI) % self.PI
        alignment = np.tanh(sacred_transform)  # Normalize to [-1, 1]
        
        # Apply sacred 69 modulation
        if len(values) % self.SACRED_69 == 0:
            alignment *= 1.069
        
        return float(np.clip(alignment, -1, 1))
    
    async def unify_signals(self, signals: Dict):
        """Unify all signals into a single decision"""
        if not signals:
            return
        
        # Calculate sacred alignment
        self.sacred_alignment = self.calculate_sacred_geometry(signals)
        
        # Get Trinity signal if available
        trinity_signal = signals.get('trinity', {})
        
        if trinity_signal:
            # Use Trinity's ULTRATHINK signal as primary
            self.unified_signal = trinity_signal.get('signal', 'hold').lower()
            self.unified_confidence = float(trinity_signal.get('confidence', 0))
            
            # Boost confidence with sacred alignment
            self.unified_confidence = min(1.0, self.unified_confidence + abs(self.sacred_alignment) * 0.1)
        else:
            # Fallback to sacred geometry decision
            if self.sacred_alignment > 0.3:
                self.unified_signal = 'buy'
            elif self.sacred_alignment < -0.3:
                self.unified_signal = 'sell'
            else:
                self.unified_signal = 'hold'
            
            self.unified_confidence = abs(self.sacred_alignment)
        
        # Apply minimum confidence threshold
        if self.unified_confidence < self.min_confidence:
            self.unified_signal = 'hold'
            logger.debug(f"Signal confidence {self.unified_confidence:.3f} below threshold {self.min_confidence}")
    
    async def broadcast_decision(self):
        """Broadcast unified decision to all components"""
        decision = {
            'signal': self.unified_signal,
            'confidence': str(self.unified_confidence),
            'sacred_alignment': str(self.sacred_alignment),
            'ml_farm_host': self.config.network.ml_farm_host,
            'timestamp': datetime.now().isoformat(),
            'components_active': sum(1 for c in self.components.values() if c['status'] == 'active')
        }
        
        try:
            # Store in Redis
            await self.redis_client.hset('ml_farm:decision', mapping=decision)
            
            # Also store in time series
            ts_key = f"ml_farm:decisions:{datetime.now().strftime('%Y%m%d')}"
            await self.redis_client.rpush(ts_key, json.dumps(decision))
            
            logger.info(f"📡 Broadcast: {self.unified_signal.upper()} "
                      f"(confidence: {self.unified_confidence:.3f}, "
                      f"sacred: {self.sacred_alignment:.3f})")
            
        except Exception as e:
            logger.error(f"Error broadcasting decision: {e}")
    
    async def check_component_health(self):
        """Check health of all components"""
        for name, component in self.components.items():
            if name == 'ml_farm':
                continue  # Skip self
            
            try:
                # Check if component has recent activity
                status_key = f"{name}:heartbeat"
                heartbeat = await self.redis_client.get(status_key)
                
                if heartbeat:
                    last_beat = datetime.fromisoformat(heartbeat)
                    age = (datetime.now() - last_beat).total_seconds()
                    
                    if age < 60:  # Active if heartbeat within 60 seconds
                        component['status'] = 'active'
                    elif age < 300:  # Warning if within 5 minutes
                        component['status'] = 'warning'
                    else:
                        component['status'] = 'inactive'
                else:
                    component['status'] = 'unknown'
                    
            except Exception as e:
                logger.debug(f"Health check failed for {name}: {e}")
                component['status'] = 'error'
    
    async def run(self):
        """Main ML Farm loop"""
        logger.info("""
        ╔══════════════════════════════════════════════════════════════╗
        ║          🧠 SECURE ML FARM SACRED BRAIN 🧠                  ║
        ║                                                              ║
        ║  ✅ Configuration Management                                ║
        ║  ✅ No Hard-coded Values                                    ║
        ║  ✅ Sacred Geometry Calculations                            ║
        ║  ✅ Unified Signal Broadcasting                             ║
        ║                                                              ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            logger.warning("Configuration validation issues:")
            for section, section_errors in errors.items():
                for error in section_errors:
                    logger.warning(f"  {section}: {error}")
        
        # Connect to Redis
        if not await self.setup_redis():
            logger.error("Failed to connect to Redis")
            return
        
        # Main loop
        iteration = 0
        while True:
            try:
                iteration += 1
                
                # Check component health
                await self.check_component_health()
                
                # Collect signals from all components
                signals = await self.collect_signals()
                
                # Unify signals into single decision
                await self.unify_signals(signals)
                
                # Broadcast unified decision
                await self.broadcast_decision()
                
                # Log status every 10 iterations
                if iteration % 10 == 0:
                    active_count = sum(1 for c in self.components.values() if c['status'] == 'active')
                    logger.info(f"📊 Iteration {iteration} | Active components: {active_count}/{len(self.components)}")
                    logger.info(f"📊 Current signal: {self.unified_signal.upper()} ({self.unified_confidence:.3f})")
                
                # Send our own heartbeat
                await self.redis_client.set('ml_farm:heartbeat', datetime.now().isoformat())
                
                # Wait before next iteration
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)
        
        # Cleanup
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("ML Farm stopped")

def main():
    """Entry point"""
    brain = SecureMLFarmBrain()
    asyncio.run(brain.run())

if __name__ == "__main__":
    main()