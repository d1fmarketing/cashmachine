#!/usr/bin/env python3
"""
Fix Polygon.io rate limiting with multiple API keys and alternatives
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

class FixedDataCollector:
    """Data collector with proper API rotation"""
    
    def __init__(self):
        self.redis_client = None
        
        # MULTIPLE API KEYS FOR ROTATION
        self.polygon_keys = [
            'beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq',  # Original (rate limited)
            # Add more Polygon keys here if available
        ]
        self.polygon_key_index = 0
        self.polygon_blocked_until = 0
        
        # Alpha Vantage keys (PAID KEY FROM TRINITY)
        self.alpha_keys = [
            '4DCP9RES6PLJBO56',  # REAL WORKING PAID KEY
        ]
        self.alpha_key_index = 0
        
        # Symbols to track
        self.symbols = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'META', 'QQQ', 'AMC', 'GME']
        
        logger.info("📡 Fixed Data Collector initialized")
    
    async def setup_redis(self):
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
            logger.error(f"Redis connection failed: {e}")
            return False
    
    async def fetch_yahoo(self, session: aiohttp.ClientSession, symbol: str):
        """Yahoo Finance - NO API KEY NEEDED!"""
        try:
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with session.get(url, headers=headers, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'chart' in data and 'result' in data['chart']:
                        result = data['chart']['result'][0]
                        meta = result.get('meta', {})
                        
                        price = meta.get('regularMarketPrice', 0)
                        if price > 0:
                            return {
                                'source': 'yahoo',
                                'symbol': symbol,
                                'price': price,
                                'volume': meta.get('regularMarketVolume', 0),
                                'previousClose': meta.get('previousClose', 0),
                                'change': price - meta.get('previousClose', price)
                            }
                elif resp.status == 429:
                    logger.warning(f"Yahoo rate limited for {symbol}")
                    await asyncio.sleep(1)
        except Exception as e:
            logger.debug(f"Yahoo error for {symbol}: {e}")
        return None
    
    async def fetch_alphavantage(self, session: aiohttp.ClientSession, symbol: str):
        """Alpha Vantage with key rotation"""
        try:
            # Rotate keys
            key = self.alpha_keys[self.alpha_key_index % len(self.alpha_keys)]
            self.alpha_key_index += 1
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': key
            }
            
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'Global Quote' in data:
                        quote = data['Global Quote']
                        price = float(quote.get('05. price', 0))
                        if price > 0:
                            return {
                                'source': 'alphavantage',
                                'symbol': symbol,
                                'price': price,
                                'volume': float(quote.get('06. volume', 0)),
                                'change': float(quote.get('09. change', 0)),
                                'changePercent': quote.get('10. change percent', '0%')
                            }
                    elif 'Note' in data:
                        logger.warning(f"AlphaVantage rate limit: {data['Note'][:50]}")
        except Exception as e:
            logger.debug(f"AlphaVantage error for {symbol}: {e}")
        return None
    
    async def fetch_polygon_carefully(self, session: aiohttp.ClientSession, symbol: str):
        """Try Polygon but with rate limit awareness"""
        
        # Check if we're still blocked
        if time.time() < self.polygon_blocked_until:
            return None
        
        try:
            key = self.polygon_keys[self.polygon_key_index % len(self.polygon_keys)]
            url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/prev'
            params = {'apiKey': key}
            
            async with session.get(url, params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'results' in data and data['results']:
                        result = data['results'][0]
                        return {
                            'source': 'polygon',
                            'symbol': symbol,
                            'price': result.get('c', 0),
                            'volume': result.get('v', 0),
                            'high': result.get('h', 0),
                            'low': result.get('l', 0)
                        }
                elif resp.status == 429:
                    # Rate limited - back off for 5 minutes
                    self.polygon_blocked_until = time.time() + 300
                    logger.warning(f"Polygon rate limited - backing off for 5 minutes")
                    # Try next key
                    self.polygon_key_index += 1
        except Exception as e:
            logger.debug(f"Polygon error for {symbol}: {e}")
        return None
    
    async def fetch_finnhub(self, session: aiohttp.ClientSession, symbol: str):
        """Finnhub as backup"""
        try:
            url = 'https://finnhub.io/api/v1/quote'
            params = {
                'symbol': symbol,
                'token': 'ct3k1a9r01qvltqha3u0ct3k1a9r01qvltqha3ug'
            }
            
            async with session.get(url, params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('c', 0) > 0:
                        return {
                            'source': 'finnhub',
                            'symbol': symbol,
                            'price': data.get('c', 0),
                            'high': data.get('h', 0),
                            'low': data.get('l', 0),
                            'previousClose': data.get('pc', 0)
                        }
        except Exception as e:
            logger.debug(f"Finnhub error for {symbol}: {e}")
        return None
    
    async def collect_symbol(self, session: aiohttp.ClientSession, symbol: str):
        """Collect data for one symbol from multiple sources"""
        results = []
        
        # Try all sources in parallel
        tasks = [
            self.fetch_yahoo(session, symbol),
            self.fetch_alphavantage(session, symbol),
            self.fetch_polygon_carefully(session, symbol),
            self.fetch_finnhub(session, symbol)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if response and not isinstance(response, Exception):
                results.append(response)
        
        return results
    
    async def aggregate_and_store(self, symbol: str, sources: List[Dict]):
        """Aggregate data from multiple sources"""
        if not sources or not self.redis_client:
            return
        
        # Get all prices
        prices = [s['price'] for s in sources if s.get('price', 0) > 0]
        volumes = [s.get('volume', 0) for s in sources if s.get('volume', 0) > 0]
        
        if prices:
            avg_price = sum(prices) / len(prices)
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            
            # Store in Redis
            try:
                await self.redis_client.hset(f'market:{symbol}', mapping={
                    'price': str(avg_price),
                    'volume': str(avg_volume),
                    'sources': str(len(sources)),
                    'timestamp': datetime.now().isoformat(),
                    'providers': ','.join([s['source'] for s in sources])
                })
                
                logger.info(f"✅ {symbol}: ${avg_price:.2f} from {len(sources)} sources ({','.join([s['source'] for s in sources])})")
                
            except Exception as e:
                logger.error(f"Redis storage error: {e}")
    
    async def run(self):
        """Main collection loop"""
        await self.setup_redis()
        
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"📡 Collection cycle {iteration}")
                
                if self.polygon_blocked_until > time.time():
                    remaining = int(self.polygon_blocked_until - time.time())
                    logger.warning(f"⚠️ Polygon blocked for {remaining}s - using alternatives")
                
                async with aiohttp.ClientSession() as session:
                    # Collect all symbols
                    tasks = []
                    for symbol in self.symbols:
                        tasks.append(self.collect_symbol(session, symbol))
                    
                    all_results = await asyncio.gather(*tasks)
                    
                    # Store results
                    for symbol, sources in zip(self.symbols, all_results):
                        if sources:
                            await self.aggregate_and_store(symbol, sources)
                
                # Report API status
                logger.info(f"📊 Collected data for {len([r for r in all_results if r])} symbols")
                
                # Sleep before next cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Collection error: {e}")
                await asyncio.sleep(10)

async def main():
    logger.info("🚀 Starting Fixed Data Collector (Polygon workaround)")
    collector = FixedDataCollector()
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())