#!/usr/bin/env python3
"""
Secure Data Collector with Configuration Management
- No hard-coded API keys
- Uses configuration manager
- Proper error handling
"""

import asyncio
import aiohttp
import redis.asyncio as redis
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureDataCollector:
    """Secure data collector using configuration management"""
    
    def __init__(self):
        # Load configuration
        self.config = get_config()
        self.redis_client = None
        
        # API configurations loaded from config manager
        self.apis = self._initialize_apis()
        
        # Trading pairs
        self.forex_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
        self.crypto_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD']
        self.stocks = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
        
        # Rate limiting tracking
        self.api_calls = {}
        
        logger.info(f"SecureDataCollector initialized with config from {self.config.env_file}")
    
    def _initialize_apis(self) -> Dict:
        """Initialize API configurations from config manager"""
        apis = {}
        
        # Yahoo Finance (no key required)
        apis['yahoo'] = {
            'enabled': self.config.get('YAHOO_FINANCE_ENABLED', 'true').lower() == 'true',
            'rate_limit': 0,
            'last_call': 0
        }
        
        # AlphaVantage
        av_config = self.config.get_api_config('alphavantage')
        if av_config.get('key'):
            apis['alphavantage'] = {
                'enabled': True,
                'key': av_config['key'],
                'base_url': av_config['base_url'],
                'rate_limit': av_config['rate_limit'],
                'last_call': 0,
                'call_count': 0
            }
        else:
            logger.warning("AlphaVantage API key not configured")
        
        # Polygon
        polygon_config = self.config.get_api_config('polygon')
        if polygon_config.get('key'):
            apis['polygon'] = {
                'enabled': True,
                'key': polygon_config['key'],
                'base_url': polygon_config['base_url'],
                'rate_limit': polygon_config['rate_limit'],
                'last_call': 0
            }
        else:
            logger.warning("Polygon API key not configured")
        
        # Finnhub
        finnhub_config = self.config.get_api_config('finnhub')
        if finnhub_config.get('key'):
            apis['finnhub'] = {
                'enabled': True,
                'key': finnhub_config['key'],
                'base_url': finnhub_config['base_url'],
                'rate_limit': finnhub_config['rate_limit'],
                'last_call': 0
            }
        else:
            logger.warning("Finnhub API key not configured")
        
        # CoinGecko
        coingecko_config = self.config.get_api_config('coingecko')
        apis['coingecko'] = {
            'enabled': True,
            'base_url': coingecko_config.get('base_url', 'https://api.coingecko.com/api/v3'),
            'rate_limit': coingecko_config.get('rate_limit', 50),
            'last_call': 0
        }
        
        return apis
    
    async def setup_redis(self):
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
            logger.info(f"✅ Connected to Redis at {self.config.network.redis_host}:{self.config.network.redis_port}")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    def can_call_api(self, api_name: str) -> bool:
        """Check if API can be called based on rate limiting"""
        api = self.apis.get(api_name)
        if not api or not api.get('enabled'):
            return False
        
        rate_limit = api.get('rate_limit', 0)
        if rate_limit == 0:
            return True
        
        current_time = time.time()
        last_call = api.get('last_call', 0)
        
        # Check if enough time has passed
        time_between_calls = 60 / rate_limit  # seconds between calls
        if current_time - last_call >= time_between_calls:
            api['last_call'] = current_time
            return True
        
        return False
    
    async def fetch_yahoo_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Yahoo Finance"""
        if not self.can_call_api('yahoo'):
            return None
        
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'interval': '1m',
                'range': '1d'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data['chart']['result'][0]
                        quote = result['indicators']['quote'][0]
                        
                        # Get latest values
                        close_prices = quote['close']
                        volumes = quote['volume']
                        
                        # Filter out None values
                        close_prices = [p for p in close_prices if p is not None]
                        volumes = [v for v in volumes if v is not None]
                        
                        if close_prices:
                            return {
                                'symbol': symbol,
                                'price': close_prices[-1],
                                'volume': volumes[-1] if volumes else 0,
                                'high': max(close_prices),
                                'low': min(close_prices),
                                'source': 'yahoo',
                                'timestamp': datetime.now().isoformat()
                            }
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
        
        return None
    
    async def fetch_alphavantage_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from AlphaVantage"""
        if not self.can_call_api('alphavantage'):
            return None
        
        api = self.apis['alphavantage']
        
        try:
            url = api['base_url']
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': api['key']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'Global Quote' in data:
                            quote = data['Global Quote']
                            return {
                                'symbol': symbol,
                                'price': float(quote['05. price']),
                                'volume': int(quote['06. volume']),
                                'high': float(quote['03. high']),
                                'low': float(quote['04. low']),
                                'change': float(quote['09. change']),
                                'change_percent': quote['10. change percent'],
                                'source': 'alphavantage',
                                'timestamp': datetime.now().isoformat()
                            }
        except Exception as e:
            logger.error(f"AlphaVantage error for {symbol}: {e}")
        
        return None
    
    async def fetch_coingecko_data(self, symbol: str) -> Optional[Dict]:
        """Fetch crypto data from CoinGecko"""
        if not self.can_call_api('coingecko'):
            return None
        
        # Map symbols to CoinGecko IDs
        coin_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana'
        }
        
        coin_id = coin_map.get(symbol.split('/')[0])
        if not coin_id:
            return None
        
        api = self.apis['coingecko']
        
        try:
            url = f"{api['base_url']}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if coin_id in data:
                            coin_data = data[coin_id]
                            return {
                                'symbol': symbol,
                                'price': coin_data['usd'],
                                'volume': coin_data.get('usd_24h_vol', 0),
                                'change_24h': coin_data.get('usd_24h_change', 0),
                                'source': 'coingecko',
                                'timestamp': datetime.now().isoformat()
                            }
        except Exception as e:
            logger.error(f"CoinGecko error for {symbol}: {e}")
        
        return None
    
    async def collect_market_data(self) -> Dict:
        """Collect data from all available sources"""
        all_data = {
            'forex': {},
            'crypto': {},
            'stocks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Collect stock data
        for symbol in self.stocks:
            # Try Yahoo first
            data = await self.fetch_yahoo_data(symbol)
            if data:
                all_data['stocks'][symbol] = data
            # Try AlphaVantage as backup
            elif 'alphavantage' in self.apis:
                data = await self.fetch_alphavantage_data(symbol)
                if data:
                    all_data['stocks'][symbol] = data
        
        # Collect crypto data
        for symbol in self.crypto_pairs:
            data = await self.fetch_coingecko_data(symbol)
            if data:
                all_data['crypto'][symbol] = data
        
        # Store in Redis
        if self.redis_client and any(all_data['stocks']) or any(all_data['crypto']):
            try:
                # Store latest market data
                await self.redis_client.hset('market:latest', mapping={
                    'stocks': json.dumps(all_data['stocks']),
                    'crypto': json.dumps(all_data['crypto']),
                    'forex': json.dumps(all_data['forex']),
                    'timestamp': all_data['timestamp']
                })
                
                # Store individual symbols
                for symbol, data in all_data['stocks'].items():
                    await self.redis_client.hset(f'market:stock:{symbol}', mapping=data)
                
                for symbol, data in all_data['crypto'].items():
                    await self.redis_client.hset(f'market:crypto:{symbol}', mapping=data)
                
                logger.info(f"📊 Stored market data: {len(all_data['stocks'])} stocks, {len(all_data['crypto'])} crypto")
            except Exception as e:
                logger.error(f"Redis storage error: {e}")
        
        return all_data
    
    async def run(self):
        """Main collection loop"""
        # Validate configuration
        errors = self.config.validate()
        if errors:
            logger.error("Configuration errors detected:")
            for section, section_errors in errors.items():
                for error in section_errors:
                    logger.error(f"  {section}: {error}")
            
            # Continue anyway with available configuration
            logger.warning("Continuing with partial configuration...")
        
        # Connect to Redis
        if not await self.setup_redis():
            logger.error("Failed to connect to Redis, exiting")
            return
        
        logger.info("🚀 Secure Data Collector started")
        logger.info(f"📊 Monitoring: {len(self.stocks)} stocks, {len(self.crypto_pairs)} crypto pairs")
        
        # Collection loop
        collection_interval = 60  # seconds
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logger.info(f"--- Iteration {iteration} ---")
                
                # Collect market data
                market_data = await self.collect_market_data()
                
                # Log summary
                total_collected = (
                    len(market_data.get('stocks', {})) + 
                    len(market_data.get('crypto', {})) + 
                    len(market_data.get('forex', {}))
                )
                logger.info(f"✅ Collected {total_collected} data points")
                
                # Log API usage
                for api_name, api_config in self.apis.items():
                    if api_config.get('enabled'):
                        call_count = api_config.get('call_count', 0)
                        if call_count > 0:
                            logger.debug(f"  {api_name}: {call_count} calls")
                
                # Wait before next collection
                await asyncio.sleep(collection_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Collection error: {e}")
                await asyncio.sleep(10)
        
        # Cleanup
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Data collector stopped")


def main():
    """Entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          📊 SECURE DATA COLLECTOR v2.0 📊                   ║
    ║                                                              ║
    ║  ✅ No hard-coded API keys                                  ║
    ║  ✅ Configuration management                                ║
    ║  ✅ Proper rate limiting                                    ║
    ║  ✅ Error handling and recovery                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    collector = SecureDataCollector()
    asyncio.run(collector.run())


if __name__ == "__main__":
    main()