#!/usr/bin/env python3
"""
ULTRATHINK DATA COLLECTOR - 9 APIs
Complete data collection from all available sources
"""

import asyncio
import aiohttp
import redis.asyncio as redis
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NineAPICollector:
    """Collector for all 9 API services"""
    
    def __init__(self):
        self.redis_client = None
        
        # API configurations
        self.apis = {
            'alphavantage': {
                'keys': ['demo', 'YOUR_KEY1', 'YOUR_KEY2'],  # Multiple keys for rotation
                'url': 'https://www.alphavantage.co/query',
                'active': True,
                'priority': 1
            },
            'polygon': {
                'key': 'beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq',
                'url': 'https://api.polygon.io/v2/aggs/ticker/{}/prev',
                'active': True,
                'priority': 2
            },
            'finnhub': {
                'key': 'ct3k1a9r01qvltqha3u0ct3k1a9r01qvltqha3ug',
                'url': 'https://finnhub.io/api/v1/quote',
                'active': True,
                'priority': 3
            },
            'yahoo': {
                'url': 'https://query1.finance.yahoo.com/v8/finance/chart/{}',
                'active': True,
                'priority': 4
            },
            'coingecko': {
                'url': 'https://api.coingecko.com/api/v3/simple/price',
                'active': True,
                'priority': 5
            },
            'binance': {
                'url': 'https://api.binance.com/api/v3/ticker/price',
                'active': False,  # Geo-blocked
                'priority': 10
            },
            'coinbase': {
                'url': 'https://api.coinbase.com/v2/exchange-rates',
                'active': True,
                'priority': 6
            },
            'kraken': {
                'url': 'https://api.kraken.com/0/public/Ticker',
                'active': True,
                'priority': 7
            },
            'cryptocompare': {
                'url': 'https://min-api.cryptocompare.com/data/price',
                'active': True,
                'priority': 8
            }
        }
        
        # Symbols to track
        self.stock_symbols = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'META', 'QQQ']
        self.crypto_symbols = ['BTC', 'ETH', 'SOL', 'MATIC', 'LINK']
        
        # API rotation counters
        self.av_key_index = 0
        self.api_call_counts = {api: 0 for api in self.apis}
        self.api_errors = {api: 0 for api in self.apis}
        
        logger.info("📡 9-API Data Collector initialized")
    
    async def setup_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = await redis.Redis(
                host='10.100.2.200',
                port=6379,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Redis connected")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def fetch_alphavantage(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch from Alpha Vantage"""
        try:
            # Rotate API keys
            keys = self.apis['alphavantage']['keys']
            key = keys[self.av_key_index % len(keys)]
            self.av_key_index += 1
            
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': key
            }
            
            async with session.get(self.apis['alphavantage']['url'], params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'Global Quote' in data:
                        quote = data['Global Quote']
                        return {
                            'source': 'alphavantage',
                            'symbol': symbol,
                            'price': float(quote.get('05. price', 0)),
                            'volume': float(quote.get('06. volume', 0)),
                            'change': float(quote.get('09. change', 0)),
                            'change_percent': quote.get('10. change percent', '0%')
                        }
        except Exception as e:
            self.api_errors['alphavantage'] += 1
            logger.debug(f"Alpha Vantage error for {symbol}: {e}")
        return None
    
    async def fetch_polygon(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch from Polygon.io"""
        try:
            url = self.apis['polygon']['url'].format(symbol)
            params = {'apiKey': self.apis['polygon']['key']}
            
            async with session.get(url, params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'results' in data and len(data['results']) > 0:
                        result = data['results'][0]
                        return {
                            'source': 'polygon',
                            'symbol': symbol,
                            'price': result.get('c', 0),  # Close price
                            'volume': result.get('v', 0),
                            'high': result.get('h', 0),
                            'low': result.get('l', 0)
                        }
        except Exception as e:
            self.api_errors['polygon'] += 1
            logger.debug(f"Polygon error for {symbol}: {e}")
        return None
    
    async def fetch_finnhub(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch from Finnhub"""
        try:
            params = {
                'symbol': symbol,
                'token': self.apis['finnhub']['key']
            }
            
            async with session.get(self.apis['finnhub']['url'], params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'source': 'finnhub',
                        'symbol': symbol,
                        'price': data.get('c', 0),  # Current price
                        'high': data.get('h', 0),
                        'low': data.get('l', 0),
                        'previous_close': data.get('pc', 0)
                    }
        except Exception as e:
            self.api_errors['finnhub'] += 1
            logger.debug(f"Finnhub error for {symbol}: {e}")
        return None
    
    async def fetch_yahoo(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch from Yahoo Finance"""
        try:
            url = self.apis['yahoo']['url'].format(symbol)
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            async with session.get(url, headers=headers, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'chart' in data and 'result' in data['chart']:
                        result = data['chart']['result'][0]
                        meta = result.get('meta', {})
                        return {
                            'source': 'yahoo',
                            'symbol': symbol,
                            'price': meta.get('regularMarketPrice', 0),
                            'volume': meta.get('regularMarketVolume', 0),
                            'previous_close': meta.get('previousClose', 0)
                        }
        except Exception as e:
            self.api_errors['yahoo'] += 1
            logger.debug(f"Yahoo error for {symbol}: {e}")
        return None
    
    async def fetch_coingecko(self, session: aiohttp.ClientSession, crypto_id: str):
        """Fetch from CoinGecko"""
        try:
            params = {
                'ids': crypto_id.lower(),
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            async with session.get(self.apis['coingecko']['url'], params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if crypto_id.lower() in data:
                        crypto_data = data[crypto_id.lower()]
                        return {
                            'source': 'coingecko',
                            'symbol': crypto_id,
                            'price': crypto_data.get('usd', 0),
                            'volume_24h': crypto_data.get('usd_24h_vol', 0),
                            'change_24h': crypto_data.get('usd_24h_change', 0)
                        }
        except Exception as e:
            self.api_errors['coingecko'] += 1
            logger.debug(f"CoinGecko error for {crypto_id}: {e}")
        return None
    
    async def fetch_coinbase(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch from Coinbase"""
        try:
            params = {'currency': symbol}
            
            async with session.get(self.apis['coinbase']['url'], params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data and 'rates' in data['data']:
                        rates = data['data']['rates']
                        if 'USD' in rates:
                            return {
                                'source': 'coinbase',
                                'symbol': symbol,
                                'price': float(rates['USD'])
                            }
        except Exception as e:
            self.api_errors['coinbase'] += 1
            logger.debug(f"Coinbase error for {symbol}: {e}")
        return None
    
    async def fetch_kraken(self, session: aiohttp.ClientSession, pair: str):
        """Fetch from Kraken"""
        try:
            params = {'pair': f'{pair}USD'}
            
            async with session.get(self.apis['kraken']['url'], params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'result' in data:
                        for key, value in data['result'].items():
                            return {
                                'source': 'kraken',
                                'symbol': pair,
                                'price': float(value['c'][0]),  # Last trade closed
                                'volume': float(value['v'][1]),  # Volume last 24h
                                'high_24h': float(value['h'][1]),
                                'low_24h': float(value['l'][1])
                            }
        except Exception as e:
            self.api_errors['kraken'] += 1
            logger.debug(f"Kraken error for {pair}: {e}")
        return None
    
    async def fetch_cryptocompare(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch from CryptoCompare"""
        try:
            params = {
                'fsym': symbol,
                'tsyms': 'USD'
            }
            
            async with session.get(self.apis['cryptocompare']['url'], params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'USD' in data:
                        return {
                            'source': 'cryptocompare',
                            'symbol': symbol,
                            'price': data['USD']
                        }
        except Exception as e:
            self.api_errors['cryptocompare'] += 1
            logger.debug(f"CryptoCompare error for {symbol}: {e}")
        return None
    
    async def collect_all_data(self):
        """Collect data from all active APIs"""
        collected_data = {}
        
        async with aiohttp.ClientSession() as session:
            # Collect stock data
            tasks = []
            for symbol in self.stock_symbols:
                if self.apis['alphavantage']['active']:
                    tasks.append(self.fetch_alphavantage(session, symbol))
                if self.apis['polygon']['active']:
                    tasks.append(self.fetch_polygon(session, symbol))
                if self.apis['finnhub']['active']:
                    tasks.append(self.fetch_finnhub(session, symbol))
                if self.apis['yahoo']['active']:
                    tasks.append(self.fetch_yahoo(session, symbol))
            
            # Collect crypto data
            for symbol in self.crypto_symbols:
                crypto_id = 'bitcoin' if symbol == 'BTC' else 'ethereum' if symbol == 'ETH' else symbol.lower()
                
                if self.apis['coingecko']['active']:
                    tasks.append(self.fetch_coingecko(session, crypto_id))
                if self.apis['coinbase']['active']:
                    tasks.append(self.fetch_coinbase(session, symbol))
                if self.apis['kraken']['active']:
                    tasks.append(self.fetch_kraken(session, 'XBT' if symbol == 'BTC' else symbol))
                if self.apis['cryptocompare']['active']:
                    tasks.append(self.fetch_cryptocompare(session, symbol))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if result and not isinstance(result, Exception):
                    symbol = result.get('symbol')
                    source = result.get('source')
                    if symbol:
                        if symbol not in collected_data:
                            collected_data[symbol] = {}
                        collected_data[symbol][source] = result
            
        return collected_data
    
    async def aggregate_and_store(self, data: Dict):
        """Aggregate data from multiple sources and store in Redis"""
        if not self.redis_client:
            return
        
        for symbol, sources in data.items():
            if not sources:
                continue
            
            # Aggregate prices from all sources
            prices = []
            volumes = []
            
            for source_data in sources.values():
                if 'price' in source_data and source_data['price'] > 0:
                    prices.append(source_data['price'])
                if 'volume' in source_data and source_data.get('volume', 0) > 0:
                    volumes.append(source_data['volume'])
            
            if prices:
                # Calculate aggregated values
                avg_price = sum(prices) / len(prices)
                avg_volume = sum(volumes) / len(volumes) if volumes else 0
                
                # Store in Redis
                try:
                    await self.redis_client.hset(f'market:{symbol}', mapping={
                        'price': str(avg_price),
                        'volume': str(avg_volume),
                        'sources': str(len(sources)),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Store individual source data
                    for source, source_data in sources.items():
                        await self.redis_client.hset(f'market:{symbol}:{source}', mapping={
                            k: str(v) for k, v in source_data.items()
                        })
                    
                    logger.info(f"✅ Stored {symbol}: ${avg_price:.2f} from {len(sources)} sources")
                    
                except Exception as e:
                    logger.error(f"Redis storage error for {symbol}: {e}")
    
    async def run(self):
        """Main collection loop"""
        await self.setup_redis()
        
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"📡 Collection cycle {iteration}")
                
                # Collect from all APIs
                start_time = time.time()
                data = await self.collect_all_data()
                collection_time = time.time() - start_time
                
                # Count successful API calls
                total_data_points = sum(len(sources) for sources in data.values())
                logger.info(f"📊 Collected {total_data_points} data points in {collection_time:.2f}s")
                
                # Aggregate and store
                await self.aggregate_and_store(data)
                
                # Report API health
                if iteration % 10 == 0:
                    logger.info("📈 API Health Report:")
                    for api, errors in self.api_errors.items():
                        if self.apis[api]['active']:
                            status = "✅" if errors == 0 else "⚠️" if errors < 5 else "❌"
                            logger.info(f"  {status} {api}: {errors} errors")
                
                # Sleep before next cycle
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Collection error: {e}")
                await asyncio.sleep(10)

async def main():
    logger.info("🚀 Starting 9-API Data Collector")
    collector = NineAPICollector()
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())