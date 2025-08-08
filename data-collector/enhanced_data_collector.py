#!/usr/bin/env python3
"""
Enhanced Data Collector - Gets ALL stocks and publishes to Redis
Uses multiple APIs to avoid rate limits
"""

import redis
import json
import time
import requests
import logging
from datetime import datetime
import threading
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ENHANCED_COLLECTOR')

class EnhancedDataCollector:
    def __init__(self):
        # Redis connection
        self.redis_client = redis.Redis(
            host='10.100.2.200',
            port=6379,
            decode_responses=True
        )
        
        # Multiple API keys for redundancy
        self.apis = {
            'polygon': {
                'url': 'https://api.polygon.io/v2/aggs/ticker/{}/prev',
                'key': 'beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq'
            },
            'alphavantage': [
                'PKGXVRHYGL3DT8QQ795W',  # Key 1
                'BPP52CKE3K2X88UH',      # Key 2
                'K8V0U0N3Z20D66Z0',      # Key 3
                'demo'                    # Fallback
            ],
            'finnhub': {
                'url': 'https://finnhub.io/api/v1/quote',
                'key': 'crrss69r01qme9drfkogcrrss69r01qme9drfkp0'
            },
            'yahoo': {
                'url': 'https://query1.finance.yahoo.com/v8/finance/chart/{}'
            }
        }
        
        # Stocks to collect
        self.stocks = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'GOOGL', 'MSFT', 'NVDA', 'META']
        self.crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        
        # API key rotation
        self.av_key_index = 0
        
        # Price cache
        self.price_cache = {}
        
    def get_polygon_data(self, symbol):
        """Get data from Polygon (best for stocks)"""
        try:
            url = self.apis['polygon']['url'].format(symbol)
            params = {'apiKey': self.apis['polygon']['key']}
            
            resp = requests.get(url, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('results'):
                    result = data['results'][0]
                    return {
                        'price': result['c'],
                        'volume': result['v'],
                        'high': result['h'],
                        'low': result['l'],
                        'source': 'polygon'
                    }
        except Exception as e:
            logger.debug(f"Polygon error for {symbol}: {e}")
        return None
        
    def get_alphavantage_data(self, symbol):
        """Get data from AlphaVantage with key rotation"""
        try:
            # Rotate API keys
            key = self.apis['alphavantage'][self.av_key_index]
            self.av_key_index = (self.av_key_index + 1) % len(self.apis['alphavantage'])
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': key
            }
            
            resp = requests.get(url, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    return {
                        'price': float(quote.get('05. price', 0)),
                        'volume': int(quote.get('06. volume', 0)),
                        'high': float(quote.get('03. high', 0)),
                        'low': float(quote.get('04. low', 0)),
                        'source': 'alphavantage'
                    }
        except Exception as e:
            logger.debug(f"AlphaVantage error for {symbol}: {e}")
        return None
        
    def get_finnhub_data(self, symbol):
        """Get data from Finnhub"""
        try:
            url = self.apis['finnhub']['url']
            params = {
                'symbol': symbol,
                'token': self.apis['finnhub']['key']
            }
            
            resp = requests.get(url, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('c'):
                    return {
                        'price': data['c'],
                        'high': data['h'],
                        'low': data['l'],
                        'source': 'finnhub'
                    }
        except Exception as e:
            logger.debug(f"Finnhub error for {symbol}: {e}")
        return None
        
    def get_yahoo_data(self, symbol):
        """Get data from Yahoo Finance (no API key needed)"""
        try:
            url = self.apis['yahoo']['url'].format(symbol)
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    meta = result.get('meta', {})
                    return {
                        'price': meta.get('regularMarketPrice', 0),
                        'volume': meta.get('regularMarketVolume', 0),
                        'source': 'yahoo'
                    }
        except Exception as e:
            logger.debug(f"Yahoo error for {symbol}: {e}")
        return None
        
    def collect_stock_data(self, symbol):
        """Collect data for a single stock using multiple sources"""
        # Try multiple sources in order
        data = None
        
        # 1. Try Polygon first (best for stocks)
        data = self.get_polygon_data(symbol)
        
        # 2. Try Yahoo (no rate limit)
        if not data:
            data = self.get_yahoo_data(symbol)
            
        # 3. Try Finnhub
        if not data:
            data = self.get_finnhub_data(symbol)
            
        # 4. Try AlphaVantage with key rotation
        if not data:
            data = self.get_alphavantage_data(symbol)
            
        # 5. Use cached price if all APIs fail
        if not data and symbol in self.price_cache:
            data = self.price_cache[symbol].copy()
            data['source'] = 'cache'
            
        if data and data.get('price', 0) > 0:
            # Update cache
            self.price_cache[symbol] = data.copy()
            
            # Add metadata
            data['symbol'] = symbol
            data['timestamp'] = datetime.now().isoformat()
            
            # Store in Redis
            self.redis_client.set(
                f'market_data:{symbol}',
                json.dumps(data),
                ex=60  # 1 minute TTL
            )
            
            # Also store in hash for ULTRATHINK
            self.redis_client.hset(
                f'market:{symbol}',
                'price', data['price']
            )
            self.redis_client.hset(
                f'market:{symbol}',
                'volume', data.get('volume', 0)
            )
            
            # Publish to channel
            self.redis_client.publish(
                'ultrathink:market:data',
                json.dumps(data)
            )
            
            logger.info(f"✅ {symbol}: ${data['price']:.2f} from {data['source']}")
            return True
        else:
            logger.warning(f"⚠️ No data for {symbol}")
            return False
            
    def run(self):
        """Main collection loop"""
        logger.info("🚀 Enhanced Data Collector starting...")
        logger.info(f"📊 Collecting: {', '.join(self.stocks + self.crypto)}")
        
        while True:
            try:
                # Collect all stocks
                success_count = 0
                for symbol in self.stocks:
                    if self.collect_stock_data(symbol):
                        success_count += 1
                    time.sleep(0.5)  # Small delay between requests
                    
                # Collect crypto (using Yahoo with -USD suffix)
                for crypto in self.crypto:
                    if self.collect_stock_data(crypto):
                        success_count += 1
                    time.sleep(0.5)
                    
                logger.info(f"📈 Collected {success_count}/{len(self.stocks + self.crypto)} symbols")
                
                # Update stats in Redis
                self.redis_client.hset(
                    'ultrathink:collector:stats',
                    'last_update', datetime.now().isoformat()
                )
                self.redis_client.hset(
                    'ultrathink:collector:stats',
                    'symbols_collected', success_count
                )
                
                # Wait before next cycle
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Collection error: {e}")
                time.sleep(5)

if __name__ == '__main__':
    collector = EnhancedDataCollector()
    collector.run()