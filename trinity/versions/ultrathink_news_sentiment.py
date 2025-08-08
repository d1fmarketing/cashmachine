#!/usr/bin/env python3
"""
ULTRATHINK News & Sentiment Aggregator
Integrates Benzinga, NewsAPI, and Tiingo for market sentiment
"""

import asyncio
import aiohttp
import redis.asyncio as redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NEWS_SENTIMENT')

class NewsSentimentAggregator:
    """Aggregates news and sentiment from multiple sources"""
    
    def __init__(self):
        # API configurations
        self.benzinga_key = 'bz.CW3UBPZ7QJBBHKGRPLBEUVALVZO6AQOS'
        self.newsapi_key = '64b2a2a8adb240fe9ba8b80b62878a21'
        self.tiingo_token = 'ea97772d4100918051b77b585f6ba9b2a0c7a094'
        
        self.redis_client = None
        self.sentiment_scores = {}
        
    async def connect_redis(self):
        """Connect to Redis"""
        self.redis_client = await redis.Redis(
            host='10.100.2.200',
            port=6379,
            decode_responses=True
        )
        await self.redis_client.ping()
        logger.info("✅ Connected to Redis")
        
    async def fetch_benzinga_news(self, session: aiohttp.ClientSession, tickers: List[str]):
        """Fetch news from Benzinga"""
        try:
            url = 'https://api.benzinga.com/api/v2/news'
            params = {
                'token': self.benzinga_key,
                'tickers': ','.join(tickers),
                'displayOutput': 'full',
                'pageSize': 20
            }
            
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Analyze sentiment
                    sentiment = self.analyze_sentiment(data)
                    
                    # Store in Redis
                    await self.redis_client.hset(
                        'news:benzinga',
                        mapping={
                            'sentiment': sentiment,
                            'timestamp': datetime.now().isoformat(),
                            'article_count': len(data) if isinstance(data, list) else 0
                        }
                    )
                    
                    logger.info(f"📰 Benzinga: {len(data) if isinstance(data, list) else 0} articles, sentiment: {sentiment:.2f}")
                    return sentiment
        except Exception as e:
            logger.error(f"Benzinga error: {e}")
        return 0.5
    
    async def fetch_newsapi(self, session: aiohttp.ClientSession, query: str):
        """Fetch news from NewsAPI"""
        try:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'apiKey': self.newsapi_key,
                'q': query,
                'sortBy': 'relevancy',
                'pageSize': 20,
                'from': (datetime.now() - timedelta(days=1)).isoformat()
            }
            
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    articles = data.get('articles', [])
                    
                    # Analyze sentiment
                    sentiment = self.analyze_news_sentiment(articles)
                    
                    # Store in Redis
                    await self.redis_client.hset(
                        'news:newsapi',
                        mapping={
                            'sentiment': sentiment,
                            'timestamp': datetime.now().isoformat(),
                            'article_count': len(articles)
                        }
                    )
                    
                    logger.info(f"📰 NewsAPI: {len(articles)} articles, sentiment: {sentiment:.2f}")
                    return sentiment
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        return 0.5
    
    async def fetch_tiingo_news(self, session: aiohttp.ClientSession, tickers: List[str]):
        """Fetch news from Tiingo"""
        try:
            url = 'https://api.tiingo.com/tiingo/news'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Token {self.tiingo_token}'
            }
            params = {
                'tickers': ','.join(tickers),
                'limit': 20
            }
            
            async with session.get(url, headers=headers, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Analyze sentiment
                    sentiment = self.analyze_tiingo_sentiment(data)
                    
                    # Store in Redis
                    await self.redis_client.hset(
                        'news:tiingo',
                        mapping={
                            'sentiment': sentiment,
                            'timestamp': datetime.now().isoformat(),
                            'article_count': len(data)
                        }
                    )
                    
                    logger.info(f"📰 Tiingo: {len(data)} articles, sentiment: {sentiment:.2f}")
                    return sentiment
        except Exception as e:
            logger.error(f"Tiingo error: {e}")
        return 0.5
    
    def analyze_sentiment(self, articles) -> float:
        """Analyze sentiment from news articles"""
        if not articles:
            return 0.5
        
        # Simple sentiment analysis based on keywords
        positive_words = ['gain', 'rise', 'up', 'high', 'bull', 'profit', 'surge', 'rally', 'boom', 'growth']
        negative_words = ['loss', 'fall', 'down', 'low', 'bear', 'crash', 'decline', 'drop', 'recession', 'crisis']
        
        positive_count = 0
        negative_count = 0
        
        for article in (articles if isinstance(articles, list) else []):
            text = str(article.get('title', '') + ' ' + article.get('description', '')).lower()
            
            for word in positive_words:
                positive_count += text.count(word)
            for word in negative_words:
                negative_count += text.count(word)
        
        if positive_count + negative_count == 0:
            return 0.5
        
        # Return sentiment score between 0 and 1
        return positive_count / (positive_count + negative_count)
    
    def analyze_news_sentiment(self, articles: List[Dict]) -> float:
        """Analyze NewsAPI articles sentiment"""
        if not articles:
            return 0.5
        
        sentiments = []
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            
            # Simple scoring
            score = 0.5
            if any(word in title + description for word in ['bull', 'gain', 'rise', 'surge']):
                score = 0.7
            elif any(word in title + description for word in ['bear', 'fall', 'crash', 'decline']):
                score = 0.3
            
            sentiments.append(score)
        
        return sum(sentiments) / len(sentiments) if sentiments else 0.5
    
    def analyze_tiingo_sentiment(self, articles: List[Dict]) -> float:
        """Analyze Tiingo news sentiment"""
        if not articles:
            return 0.5
        
        sentiments = []
        for article in articles:
            title = article.get('title', '').lower()
            
            # Check for sentiment indicators
            if 'upgrade' in title or 'outperform' in title:
                sentiments.append(0.8)
            elif 'downgrade' in title or 'underperform' in title:
                sentiments.append(0.2)
            else:
                sentiments.append(0.5)
        
        return sum(sentiments) / len(sentiments) if sentiments else 0.5
    
    async def aggregate_sentiment(self) -> Dict:
        """Aggregate sentiment from all sources"""
        # Get all sentiment scores
        benzinga = await self.redis_client.hget('news:benzinga', 'sentiment')
        newsapi = await self.redis_client.hget('news:newsapi', 'sentiment')
        tiingo = await self.redis_client.hget('news:tiingo', 'sentiment')
        
        scores = []
        if benzinga:
            scores.append(float(benzinga))
        if newsapi:
            scores.append(float(newsapi))
        if tiingo:
            scores.append(float(tiingo))
        
        if scores:
            avg_sentiment = sum(scores) / len(scores)
        else:
            avg_sentiment = 0.5
        
        # Determine market sentiment
        if avg_sentiment > 0.6:
            market_sentiment = 'BULLISH'
        elif avg_sentiment < 0.4:
            market_sentiment = 'BEARISH'
        else:
            market_sentiment = 'NEUTRAL'
        
        result = {
            'average_sentiment': avg_sentiment,
            'market_sentiment': market_sentiment,
            'benzinga': float(benzinga) if benzinga else 0.5,
            'newsapi': float(newsapi) if newsapi else 0.5,
            'tiingo': float(tiingo) if tiingo else 0.5,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store aggregated sentiment
        await self.redis_client.hset('news:sentiment', mapping=result)
        
        return result
    
    async def run(self):
        """Main loop"""
        await self.connect_redis()
        
        logger.info("""
        ╔══════════════════════════════════════════════════════════════╗
        ║          📰 NEWS SENTIMENT AGGREGATOR STARTED 📰            ║
        ║                                                              ║
        ║  Sources:                                                    ║
        ║  ✅ Benzinga Financial News                                 ║
        ║  ✅ NewsAPI Global Coverage                                 ║
        ║  ✅ Tiingo Market Intelligence                              ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Main tickers to track
        tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'BTC']
        queries = ['stock market', 'crypto', 'bitcoin', 'trading']
        
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # Fetch from all sources
                    tasks = [
                        self.fetch_benzinga_news(session, tickers),
                        self.fetch_newsapi(session, queries[0]),
                        self.fetch_tiingo_news(session, tickers)
                    ]
                    
                    await asyncio.gather(*tasks)
                    
                    # Aggregate sentiment
                    sentiment = await self.aggregate_sentiment()
                    
                    logger.info(f"""
                    📊 SENTIMENT UPDATE:
                    Market: {sentiment['market_sentiment']}
                    Score: {sentiment['average_sentiment']:.2%}
                    Benzinga: {sentiment['benzinga']:.2%}
                    NewsAPI: {sentiment['newsapi']:.2%}
                    Tiingo: {sentiment['tiingo']:.2%}
                    """)
                
                # Update every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)

def main():
    aggregator = NewsSentimentAggregator()
    asyncio.run(aggregator.run())

if __name__ == "__main__":
    main()