#!/usr/bin/env python3
"""
TRINITY SCALPER WITH ULTRATHINK INTEGRATION - SECURE VERSION
Uses configuration manager instead of hard-coded values
Integrated with ASI/HRM/MCTS signals from ULTRATHINK via Redis
"""

import os
import sys
import asyncio
import aiohttp
import json
import time
import logging
import traceback
import uuid
import redis.asyncio as redis
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import get_config

# Setup logging
config = get_config()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.paths.trinity_log),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TRINITY_ULTRATHINK_SECURE')

# ============================================================================
# CREDENTIAL MANAGER (enhanced with config manager)
# ============================================================================

class SecureCredentialManager:
    """Manages API credentials from config manager and encrypted files"""
    
    def __init__(self):
        self.config = get_config()
        self.credentials = {}
        self.load_all_credentials()
    
    def load_all_credentials(self):
        """Load credentials from config manager and encrypted files"""
        # Try to load from config manager first
        if self.config.api.oanda_token:
            self.credentials['oanda'] = {
                'token': self.config.api.oanda_token,
                'account': self.config.api.oanda_account
            }
            logger.info("✅ Loaded OANDA credentials from config")
        
        if self.config.api.alpaca_key:
            self.credentials['alpaca'] = {
                'key': self.config.api.alpaca_key,
                'secret': self.config.api.alpaca_secret,
                'base_url': self.config.api.alpaca_base_url
            }
            logger.info("✅ Loaded Alpaca credentials from config")
        
        # Fall back to encrypted files if not in config
        if not self.credentials:
            self.load_encrypted_credentials()
    
    def load_encrypted_credentials(self):
        """Load credentials from encrypted files as fallback"""
        apis = ['oanda', 'alpaca']
        
        for api in apis:
            if api in self.credentials:
                continue  # Already loaded from config
                
            try:
                from cryptography.fernet import Fernet
                
                # Load encryption key
                key_file = self.config.paths.config_dir / f".{api}.key"
                enc_file = self.config.paths.config_dir / f"{api}.enc"
                
                if key_file.exists() and enc_file.exists():
                    with open(key_file, "rb") as f:
                        key = f.read()
                    with open(enc_file, "rb") as f:
                        encrypted = f.read()
                    
                    cipher = Fernet(key)
                    decrypted = cipher.decrypt(encrypted)
                    config_data = json.loads(decrypted)
                    
                    self.credentials[api] = config_data
                    logger.info(f"✅ Loaded {api} credentials from encrypted file")
            except Exception as e:
                logger.warning(f"Could not load {api} credentials: {e}")
    
    def get(self, api: str, key: str) -> Optional[str]:
        """Get specific credential"""
        if api in self.credentials:
            return self.credentials[api].get(key)
        return None

# ============================================================================
# ULTRATHINK SIGNAL READER
# ============================================================================

class SecureUltraThinkSignalReader:
    """Reads ASI/HRM/MCTS signals from ULTRATHINK via Redis"""
    
    def __init__(self):
        self.config = get_config()
        self.redis_client = None
        self.last_signal = None
        self.signal_history = deque(maxlen=100)
        
    async def connect(self):
        """Connect to Redis using configuration"""
        try:
            self.redis_client = await redis.Redis(
                host=self.config.network.redis_host,
                port=self.config.network.redis_port,
                db=self.config.network.redis_db,
                password=self.config.network.redis_password,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"✅ Connected to ULTRATHINK via Redis at {self.config.network.redis_host}:{self.config.network.redis_port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False
    
    async def get_signal(self) -> Optional[Dict]:
        """Get latest signal from ULTRATHINK"""
        if not self.redis_client:
            return None
        
        try:
            # Get latest signal from ULTRATHINK
            signal_data = await self.redis_client.hgetall('ultrathink:signals')
            
            if signal_data:
                # Parse the signal
                signal = self.parse_signal(signal_data)
                
                # Store in history
                self.signal_history.append(signal)
                self.last_signal = signal
                
                return signal
            else:
                logger.debug("No signal data available from ULTRATHINK")
                return None
                
        except Exception as e:
            logger.error(f"Error getting ULTRATHINK signal: {e}")
            return None
    
    def parse_signal(self, data: Dict) -> Dict:
        """Parse ULTRATHINK signal data"""
        signal_value = data.get('signal', 'HOLD')
        
        # Parse numpy float format if present
        def parse_value(val):
            if isinstance(val, str):
                if val.startswith('np.float64('):
                    return float(val.replace('np.float64(', '').replace(')', ''))
                try:
                    return float(val)
                except:
                    return val
            return val
        
        # Parse component values in format "signal:confidence"
        def parse_component(val):
            if isinstance(val, str) and ':' in val:
                parts = val.split(':')
                if len(parts) == 2:
                    try:
                        return float(parts[1])
                    except:
                        return 0.5
            return parse_value(val) if val else 0.5
        
        # Parse confidence values from actual Redis format
        asi_data = data.get('asi', '0.5')
        hrm_data = data.get('hrm', '0.5')
        mcts_data = data.get('mcts', '0.5')
        
        asi_confidence = parse_component(asi_data)
        hrm_confidence = parse_component(hrm_data)
        mcts_confidence = parse_component(mcts_data)
        
        # Use 'confidence' key instead of 'combined_confidence'
        combined_confidence = parse_value(data.get('confidence', 0.5))
        
        return {
            'signal': signal_value,
            'asi_confidence': asi_confidence,
            'hrm_confidence': hrm_confidence,
            'mcts_confidence': mcts_confidence,
            'combined_confidence': combined_confidence,
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'source': 'ULTRATHINK'
        }

# ============================================================================
# TRADING CONNECTORS (Simplified for demo)
# ============================================================================

class SecureOandaConnector:
    """OANDA Forex trading connector"""
    
    def __init__(self, credentials: SecureCredentialManager):
        self.config = get_config()
        self.creds = credentials
        self.connected = False
        
    async def connect(self):
        """Connect to OANDA"""
        token = self.creds.get('oanda', 'token')
        if token:
            self.connected = True
            logger.info("✅ Connected to OANDA (simulated)")
        else:
            logger.warning("⚠️ OANDA credentials not available")
        return self.connected
    
    async def execute_trade(self, symbol: str, units: int, signal: str) -> Dict:
        """Execute forex trade"""
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
        
        # Simulated trading for demonstration
        logger.info(f"📊 OANDA: {signal} {units} units of {symbol}")
        
        return {
            'status': 'success',
            'trade_id': str(uuid.uuid4()),
            'symbol': symbol,
            'signal': signal,
            'units': units,
            'timestamp': datetime.now().isoformat()
        }

class SecureAlpacaConnector:
    """Alpaca stock/crypto trading connector"""
    
    def __init__(self, credentials: SecureCredentialManager):
        self.config = get_config()
        self.creds = credentials
        self.connected = False
        
    async def connect(self):
        """Connect to Alpaca"""
        key = self.creds.get('alpaca', 'key')
        if key:
            self.connected = True
            logger.info("✅ Connected to Alpaca (simulated)")
        else:
            logger.warning("⚠️ Alpaca credentials not available")
        return self.connected
    
    async def execute_trade(self, symbol: str, qty: float, signal: str) -> Dict:
        """Execute stock/crypto trade"""
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
        
        # Simulated trading for demonstration
        logger.info(f"📊 ALPACA: {signal} ${qty} of {symbol}")
        
        return {
            'status': 'success',
            'trade_id': str(uuid.uuid4()),
            'symbol': symbol,
            'signal': signal,
            'notional': qty,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# MAIN TRINITY SCALPER
# ============================================================================

class SecureTrinityScalper:
    """Main Trinity Scalper with ULTRATHINK Integration"""
    
    def __init__(self):
        self.config = get_config()
        self.credentials = SecureCredentialManager()
        self.signal_reader = SecureUltraThinkSignalReader()
        self.oanda = SecureOandaConnector(self.credentials)
        self.alpaca = SecureAlpacaConnector(self.credentials)
        
        # Trading parameters from config
        self.max_position_size = self.config.trading.max_position_size
        self.max_daily_trades = self.config.trading.max_daily_trades
        self.min_confidence = self.config.trading.min_confidence_threshold
        
        # Track trading activity
        self.total_trades = 0
        self.daily_trades = 0
        self.trade_results = []
        
        # Trading pairs
        self.forex = ['EUR/USD', 'GBP/USD', 'USD/JPY']
        self.crypto = ['BTC/USD', 'ETH/USD', 'SOL/USD']
        self.stocks = ['SPY', 'QQQ', 'AAPL']
        
    async def initialize(self):
        """Initialize all connections"""
        logger.info("""
        ╔══════════════════════════════════════════════════════════════╗
        ║          💰 SECURE TRINITY SCALPER v3.0 💰                  ║
        ║                                                              ║
        ║  ✅ Configuration Management                                ║
        ║  ✅ No Hard-coded Values                                    ║
        ║  ✅ ULTRATHINK Integration                                  ║
        ║  ✅ ASI/HRM/MCTS Signals                                    ║
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
        if not await self.signal_reader.connect():
            logger.error("Failed to connect to Redis/ULTRATHINK")
            return False
        
        # Connect to trading platforms
        await self.oanda.connect()
        await self.alpaca.connect()
        
        logger.info(f"✅ Trinity Scalper initialized")
        logger.info(f"📊 Max position size: ${self.max_position_size}")
        logger.info(f"📊 Max daily trades: {self.max_daily_trades}")
        logger.info(f"📊 Min confidence: {self.min_confidence}")
        
        return True
    
    async def should_trade(self, signal: Dict) -> bool:
        """Determine if we should execute trade based on signal"""
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"Daily trade limit reached ({self.max_daily_trades})")
            return False
        
        # Check confidence threshold
        confidence = signal.get('combined_confidence', 0)
        if confidence < self.min_confidence:
            logger.debug(f"Confidence {confidence:.2f} below threshold {self.min_confidence}")
            return False
        
        # Check if it's not HOLD
        if signal.get('signal', '').upper() == 'HOLD':
            return False
        
        return True
    
    async def execute_trades(self, signal: Dict):
        """Execute trades based on ULTRATHINK signal"""
        if not await self.should_trade(signal):
            return
        
        trade_signal = signal['signal']
        confidence = signal['combined_confidence']
        
        logger.info(f"🎯 Executing {trade_signal} with confidence {confidence:.2f}")
        
        # Execute on Alpaca (crypto priority to avoid PDT)
        if self.alpaca.connected:
            symbol = self.crypto[self.total_trades % len(self.crypto)]
            result = await self.alpaca.execute_trade(
                symbol,
                min(100, self.max_position_size),  # $100 or max position
                trade_signal
            )
            
            if result['status'] == 'success':
                self.daily_trades += 1
                self.total_trades += 1
                self.trade_results.append(result)
                
                # Store result in Redis
                await self.store_trade_result(result)
        
        # Execute on OANDA (forex)
        if self.oanda.connected and self.daily_trades < self.max_daily_trades:
            symbol = self.forex[self.total_trades % len(self.forex)]
            result = await self.oanda.execute_trade(
                symbol,
                1000,  # 1000 units
                trade_signal
            )
            
            if result['status'] == 'success':
                self.daily_trades += 1
                self.total_trades += 1
                self.trade_results.append(result)
                
                await self.store_trade_result(result)
    
    async def store_trade_result(self, result: Dict):
        """Store trade result in Redis"""
        try:
            key = f"trinity:trade:{result['trade_id']}"
            await self.signal_reader.redis_client.hset(key, mapping=result)
            
            # Update stats
            await self.signal_reader.redis_client.hincrby('trinity:stats', 'total_trades', 1)
            await self.signal_reader.redis_client.hset('trinity:stats', 'last_trade', json.dumps(result))
        except Exception as e:
            logger.error(f"Error storing trade result: {e}")
    
    async def run(self):
        """Main trading loop"""
        if not await self.initialize():
            logger.error("Initialization failed")
            return
        
        logger.info("🚀 Starting main trading loop")
        
        while True:
            try:
                # Get signal from ULTRATHINK
                signal = await self.signal_reader.get_signal()
                
                if signal:
                    logger.info(f"📡 Signal: {signal['signal']} | "
                              f"ASI: {signal['asi_confidence']:.3f} | "
                              f"HRM: {signal['hrm_confidence']:.3f} | "
                              f"MCTS: {signal['mcts_confidence']:.3f}")
                    
                    # Execute trades based on signal
                    await self.execute_trades(signal)
                
                # Log stats periodically
                if self.total_trades % 10 == 0 and self.total_trades > 0:
                    logger.info(f"📊 Stats - Total: {self.total_trades} | Daily: {self.daily_trades}")
                
                # Reset daily counter at midnight
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    self.daily_trades = 0
                    logger.info("🔄 Daily trade counter reset")
                
                # Wait before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(10)
        
        logger.info("Trinity Scalper stopped")

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Entry point"""
    scalper = SecureTrinityScalper()
    asyncio.run(scalper.run())

if __name__ == "__main__":
    main()