#!/usr/bin/env python3
"""
ULTRATHINK Finnhub-Backtrader Bridge
Alternative data, transcripts, and SEC filings for AI trading
"""

import backtrader as bt
import pandas as pd
import numpy as np
import finnhub
from datetime import datetime, timedelta
import json
import os
import threading
import queue
from cryptography.fernet import Fernet
from typing import Dict, Optional, List
import websocket
import time

class FinnhubStore:
    """Finnhub data store for Backtrader with unique alternative data"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key or load from encrypted config"""
        
        if not api_key:
            # Load from encrypted config
            api_key = self.load_api_key()
        
        self.api_key = api_key
        
        # Set proxy
        os.environ['http_proxy'] = 'http://10.100.1.72:3128'
        os.environ['https_proxy'] = 'http://10.100.1.72:3128'
        
        # Initialize Finnhub client
        self.client = finnhub.Client(api_key=api_key)
        
        # WebSocket for real-time data
        self.ws = None
        self.ws_thread = None
        self.data_queue = queue.Queue()
        
        # Cache for alternative data
        self.transcripts_cache = {}
        self.filings_cache = {}
        self.news_cache = {}
        
        print(f"✅ Finnhub Store initialized")
        print(f"   Features: Transcripts, SEC Filings, Alternative Data")
    
    def load_api_key(self) -> str:
        """Load API key from encrypted config"""
        config_dir = "/opt/cashmachine/config"
        
        try:
            with open(f"{config_dir}/.finnhub.key", "rb") as f:
                key = f.read()
            with open(f"{config_dir}/finnhub.enc", "rb") as f:
                encrypted = f.read()
            cipher = Fernet(key)
            config = json.loads(cipher.decrypt(encrypted))
            return config['api_key']
        except:
            # Fallback to direct key
            return "d296dn9r01qhoena6u00d296dn9r01qhoena6u0g"
    
    def get_candles(self, symbol: str, resolution: str = 'D', count: int = 100):
        """Get historical candle data"""
        try:
            end_time = int(datetime.now().timestamp())
            
            # Calculate start time based on resolution and count
            if resolution == 'D':
                start_time = int((datetime.now() - timedelta(days=count)).timestamp())
            elif resolution == '60':  # 60 minutes
                start_time = int((datetime.now() - timedelta(hours=count)).timestamp())
            else:
                start_time = int((datetime.now() - timedelta(minutes=count * int(resolution))).timestamp())
            
            # Get candle data
            res = self.client.stock_candles(symbol, resolution, start_time, end_time)
            
            if res['s'] == 'ok':
                # Convert to DataFrame
                df = pd.DataFrame({
                    'datetime': pd.to_datetime(res['t'], unit='s'),
                    'open': res['o'],
                    'high': res['h'],
                    'low': res['l'],
                    'close': res['c'],
                    'volume': res['v']
                })
                df.set_index('datetime', inplace=True)
                
                print(f"📊 Retrieved {len(df)} candles for {symbol}")
                return df
            
        except Exception as e:
            print(f"❌ Error getting candles: {e}")
        
        return None
    
    def get_earnings_transcript(self, symbol: str, year: int = None, quarter: int = None):
        """Get earnings call transcript for sentiment analysis"""
        try:
            # Get list of transcripts
            transcripts = self.client.earnings_call_transcripts(symbol)
            
            if transcripts:
                # Get most recent or specific transcript
                if year and quarter:
                    for t in transcripts:
                        if t.get('year') == year and t.get('quarter') == quarter:
                            # Get full transcript
                            transcript_id = t.get('id')
                            full_transcript = self.client.earnings_call_transcript(transcript_id)
                            self.transcripts_cache[symbol] = full_transcript
                            return full_transcript
                else:
                    # Get most recent
                    if len(transcripts) > 0:
                        transcript_id = transcripts[0].get('id')
                        full_transcript = self.client.earnings_call_transcript(transcript_id)
                        self.transcripts_cache[symbol] = full_transcript
                        return full_transcript
            
        except Exception as e:
            print(f"❌ Error getting transcript: {e}")
        
        return None
    
    def get_sec_filings(self, symbol: str, form: str = None):
        """Get SEC filings for regulatory analysis"""
        try:
            # Get recent filings
            from_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            filings = self.client.filings(symbol=symbol, _from=from_date, to=to_date, form=form)
            
            if filings:
                self.filings_cache[symbol] = filings
                print(f"📋 Retrieved {len(filings)} SEC filings for {symbol}")
                return filings
            
        except Exception as e:
            print(f"❌ Error getting filings: {e}")
        
        return []
    
    def get_insider_transactions(self, symbol: str):
        """Get insider trading data"""
        try:
            transactions = self.client.insider_transactions(symbol)
            
            if transactions and 'data' in transactions:
                print(f"👔 Retrieved {len(transactions['data'])} insider transactions")
                return transactions['data']
            
        except Exception as e:
            print(f"❌ Error getting insider transactions: {e}")
        
        return []
    
    def get_company_news(self, symbol: str, days_back: int = 7):
        """Get recent company news for sentiment analysis"""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            news = self.client.company_news(symbol, _from=from_date, to=to_date)
            
            if news:
                self.news_cache[symbol] = news
                print(f"📰 Retrieved {len(news)} news articles for {symbol}")
                return news
            
        except Exception as e:
            print(f"❌ Error getting news: {e}")
        
        return []
    
    def get_fundamentals(self, symbol: str):
        """Get company fundamentals"""
        try:
            # Basic financials
            financials = self.client.company_basic_financials(symbol, 'all')
            
            # Company profile
            profile = self.client.company_profile2(symbol=symbol)
            
            # Combine data
            fundamentals = {
                'profile': profile,
                'metrics': financials.get('metric', {}),
                'series': financials.get('series', {})
            }
            
            return fundamentals
            
        except Exception as e:
            print(f"❌ Error getting fundamentals: {e}")
        
        return None
    
    def start_websocket(self, symbols: List[str]):
        """Start WebSocket for real-time data"""
        def on_message(ws, message):
            data = json.loads(message)
            self.data_queue.put(data)
        
        def on_error(ws, error):
            print(f"❌ WebSocket error: {error}")
        
        def on_close(ws):
            print("📡 WebSocket closed")
        
        def on_open(ws):
            # Subscribe to symbols
            for symbol in symbols:
                ws.send(json.dumps({'type': 'subscribe', 'symbol': symbol}))
            print(f"📡 WebSocket subscribed to {symbols}")
        
        # Create WebSocket connection
        websocket_url = f"wss://ws.finnhub.io?token={self.api_key}"
        self.ws = websocket.WebSocketApp(
            websocket_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run in separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()
        
        print("📡 WebSocket streaming started")


class FinnhubData(bt.feeds.PandasData):
    """Finnhub data feed for Backtrader"""
    
    params = (
        ('symbol', 'AAPL'),
        ('resolution', 'D'),  # D, W, M, 1, 5, 15, 30, 60
        ('count', 100),
        ('api_key', None),
    )
    
    def __init__(self):
        super(FinnhubData, self).__init__()
        
        # Create store
        self.store = FinnhubStore(api_key=self.params.api_key)
        
        # Get candle data
        df = self.store.get_candles(
            self.params.symbol,
            self.params.resolution,
            self.params.count
        )
        
        if df is not None:
            self.p.dataname = df


class UltrathinkFinnhubStrategy(bt.Strategy):
    """
    ULTRATHINK Strategy using Finnhub alternative data
    Combines price action with transcripts, filings, and news sentiment
    """
    
    params = (
        ('symbol', 'AAPL'),
        ('use_transcripts', True),
        ('use_filings', True),
        ('use_insider', True),
        ('use_news', True),
    )
    
    def __init__(self):
        # Initialize Finnhub store
        self.finnhub_store = FinnhubStore()
        
        # Technical indicators
        self.sma20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma50 = bt.indicators.SMA(self.data.close, period=50)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
        # Alternative data signals
        self.transcript_sentiment = 0
        self.filing_signal = 0
        self.insider_signal = 0
        self.news_sentiment = 0
        
        # Load alternative data
        self.load_alternative_data()
        
        print("🧠 ULTRATHINK Finnhub Strategy initialized")
        print("   Features: Transcripts, SEC Filings, Insider Trading, News")
    
    def load_alternative_data(self):
        """Load and analyze alternative data"""
        
        if self.params.use_transcripts:
            transcript = self.finnhub_store.get_earnings_transcript(self.params.symbol)
            if transcript:
                # Analyze transcript sentiment (simplified)
                self.transcript_sentiment = self.analyze_transcript_sentiment(transcript)
                print(f"📝 Transcript sentiment: {self.transcript_sentiment:.2f}")
        
        if self.params.use_filings:
            filings = self.finnhub_store.get_sec_filings(self.params.symbol)
            if filings:
                self.filing_signal = self.analyze_filings(filings)
                print(f"📋 SEC filing signal: {self.filing_signal:.2f}")
        
        if self.params.use_insider:
            transactions = self.finnhub_store.get_insider_transactions(self.params.symbol)
            if transactions:
                self.insider_signal = self.analyze_insider_trading(transactions)
                print(f"👔 Insider signal: {self.insider_signal:.2f}")
        
        if self.params.use_news:
            news = self.finnhub_store.get_company_news(self.params.symbol)
            if news:
                self.news_sentiment = self.analyze_news_sentiment(news)
                print(f"📰 News sentiment: {self.news_sentiment:.2f}")
    
    def analyze_transcript_sentiment(self, transcript):
        """Analyze earnings call transcript sentiment"""
        if not transcript:
            return 0
        
        # Simplified sentiment analysis
        positive_words = ['growth', 'strong', 'increase', 'positive', 'exceed', 'record']
        negative_words = ['decline', 'weak', 'decrease', 'negative', 'miss', 'challenge']
        
        text = str(transcript).lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count + negative_count > 0:
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            return sentiment
        
        return 0
    
    def analyze_filings(self, filings):
        """Analyze SEC filings for signals"""
        if not filings:
            return 0
        
        # Check for important form types
        important_forms = ['10-K', '10-Q', '8-K', 'DEF 14A']
        signal = 0
        
        for filing in filings[:5]:  # Check recent 5 filings
            if filing.get('form') in important_forms:
                signal += 0.2
        
        return min(signal, 1.0)
    
    def analyze_insider_trading(self, transactions):
        """Analyze insider trading patterns"""
        if not transactions:
            return 0
        
        buy_count = 0
        sell_count = 0
        
        for trans in transactions[:10]:  # Check recent 10 transactions
            if trans.get('transactionType') == 'Buy':
                buy_count += 1
            elif trans.get('transactionType') == 'Sell':
                sell_count += 1
        
        if buy_count + sell_count > 0:
            signal = (buy_count - sell_count) / (buy_count + sell_count)
            return signal
        
        return 0
    
    def analyze_news_sentiment(self, news):
        """Analyze news sentiment"""
        if not news:
            return 0
        
        # Simplified sentiment based on headlines
        positive_words = ['beats', 'surges', 'gains', 'rallies', 'upgrade']
        negative_words = ['misses', 'falls', 'drops', 'downgrade', 'concern']
        
        sentiment_sum = 0
        for article in news[:10]:  # Check recent 10 articles
            headline = article.get('headline', '').lower()
            
            for word in positive_words:
                if word in headline:
                    sentiment_sum += 0.1
                    break
            
            for word in negative_words:
                if word in headline:
                    sentiment_sum -= 0.1
                    break
        
        return max(min(sentiment_sum, 1.0), -1.0)
    
    def next(self):
        """Trading logic combining technical and alternative data"""
        
        # Technical signals
        tech_bullish = self.sma20[0] > self.sma50[0] and self.rsi[0] < 70
        tech_bearish = self.sma20[0] < self.sma50[0] or self.rsi[0] > 80
        
        # Alternative data composite signal
        alt_signal = (
            self.transcript_sentiment * 0.25 +
            self.filing_signal * 0.15 +
            self.insider_signal * 0.35 +
            self.news_sentiment * 0.25
        )
        
        # Combined decision
        if not self.position:
            if tech_bullish and alt_signal > 0.2:
                self.buy()
                print(f"✅ BUY at ${self.data.close[0]:.2f}")
                print(f"   Tech: Bullish, Alt Signal: {alt_signal:.2f}")
        else:
            if tech_bearish or alt_signal < -0.2:
                self.close()
                print(f"📤 CLOSE at ${self.data.close[0]:.2f}")
                print(f"   Tech: Bearish, Alt Signal: {alt_signal:.2f}")


def test_finnhub_integration():
    """Test Finnhub integration with Backtrader"""
    
    print("\n" + "="*60)
    print("🧪 ULTRATHINK Finnhub Integration Test")
    print("="*60)
    
    cerebro = bt.Cerebro()
    
    # Add Finnhub data feed
    data = FinnhubData(
        symbol='AAPL',
        resolution='D',
        count=100
    )
    
    if data.p.dataname is not None:
        cerebro.adddata(data)
        
        # Add strategy
        cerebro.addstrategy(
            UltrathinkFinnhubStrategy,
            symbol='AAPL',
            use_transcripts=True,
            use_filings=True,
            use_insider=True,
            use_news=True
        )
        
        # Set broker parameters
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.001)
        
        print(f"\n💰 Initial Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        
        # Run backtest
        cerebro.run()
        
        print(f"💰 Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        print("\n✅ Finnhub integration test complete!")
        print("📝 Alternative data analysis ready!")
    else:
        print("❌ Failed to load data - check API connection")


if __name__ == "__main__":
    # Test the integration
    test_finnhub_integration()