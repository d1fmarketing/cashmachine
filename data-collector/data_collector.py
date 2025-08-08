#!/usr/bin/env python3
"""
ULTRATHINK Data Collector
Collects market data and feeds it to Redis for AI processing
"""

import json
import time
import redis
import requests
import logging
from datetime import datetime
import threading
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DATA_COLLECTOR - %(levelname)s - %(message)s'
)

class DataCollector:
    def __init__(self):
        logging.info("🔄 Initializing Data Collector...")
        
        # Redis connection
        self.redis_client = redis.Redis(
            host='10.100.2.200',
            port=6379,
            decode_responses=True
        )
        
        # OANDA API Config
        self.oanda_url = "https://api-fxpractice.oanda.com"
        self.account_id = "101-001-27477016-001"
        self.api_key = "01cc03ede7cda93a88e87e4e0f1c6912-1cdac97a23c3e1d80f3c8e759e43f4e0"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Instruments to track
        self.instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']
        
        # Price history
        self.price_history = {inst: [] for inst in self.instruments}
        self.max_history = 100
        
        self.running = False
        
    def start(self):
        """Start data collection"""
        logging.info("🚀 Starting Data Collector...")
        self.running = True
        
        # Start collection threads
        threads = [
            threading.Thread(target=self._collect_prices, daemon=True),
            threading.Thread(target=self._collect_account_data, daemon=True),
            threading.Thread(target=self._process_indicators, daemon=True)
        ]
        
        for t in threads:
            t.start()
            
        logging.info("✅ Data Collector active")
        
        # Main loop
        while self.running:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
                break
                
    def stop(self):
        """Stop data collection"""
        logging.info("Stopping Data Collector...")
        self.running = False
        
    def _collect_prices(self):
        """Collect price data from OANDA"""
        while self.running:
            try:
                for instrument in self.instruments:
                    # Get current price
                    url = f"{self.oanda_url}/v3/accounts/{self.account_id}/pricing"
                    params = {"instruments": instrument}
                    
                    response = requests.get(url, headers=self.headers, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'prices' in data and len(data['prices']) > 0:
                            price_data = data['prices'][0]
                            
                            bid = float(price_data.get('bids', [{'price': '0'}])[0]['price'])
                            ask = float(price_data.get('asks', [{'price': '0'}])[0]['price'])
                            mid = (bid + ask) / 2
                            
                            # Update history
                            self.price_history[instrument].append(mid)
                            if len(self.price_history[instrument]) > self.max_history:
                                self.price_history[instrument].pop(0)
                            
                            # Store in Redis
                            market_data = {
                                'instrument': instrument,
                                'bid': bid,
                                'ask': ask,
                                'price': mid,
                                'spread': ask - bid,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Save current price
                            self.redis_client.hset(
                                f"market:{instrument}",
                                "current",
                                json.dumps(market_data)
                            )
                            
                            # Publish to channel
                            self.redis_client.publish(
                                "market:prices",
                                json.dumps(market_data)
                            )
                            
                            logging.info(f"📊 {instrument}: {mid:.5f} (spread: {(ask-bid)*10000:.1f} pips)")
                    
                time.sleep(2)  # Collect every 2 seconds
                
            except Exception as e:
                logging.error(f"Price collection error: {e}")
                time.sleep(5)
                
    def _collect_account_data(self):
        """Collect account data"""
        while self.running:
            try:
                # Get account summary
                url = f"{self.oanda_url}/v3/accounts/{self.account_id}/summary"
                response = requests.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    account = data.get('account', {})
                    
                    account_data = {
                        'balance': float(account.get('balance', 0)),
                        'unrealized_pl': float(account.get('unrealizedPL', 0)),
                        'nav': float(account.get('NAV', 0)),
                        'margin_used': float(account.get('marginUsed', 0)),
                        'margin_available': float(account.get('marginAvailable', 0)),
                        'open_trades': int(account.get('openTradeCount', 0)),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Store in Redis
                    self.redis_client.hset(
                        "account:summary",
                        "current",
                        json.dumps(account_data)
                    )
                    
                    logging.info(f"💰 Account: Balance=${account_data['balance']:.2f}, Open={account_data['open_trades']}, P&L=${account_data['unrealized_pl']:.2f}")
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logging.error(f"Account data error: {e}")
                time.sleep(30)
                
    def _process_indicators(self):
        """Calculate technical indicators"""
        while self.running:
            try:
                for instrument in self.instruments:
                    prices = self.price_history[instrument]
                    
                    if len(prices) >= 20:
                        # Calculate indicators
                        sma_20 = np.mean(prices[-20:])
                        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
                        
                        # RSI
                        rsi = self._calculate_rsi(prices[-14:])
                        
                        # Volatility
                        volatility = np.std(prices[-20:])
                        
                        # Trend
                        trend = 'up' if prices[-1] > sma_20 else 'down'
                        
                        indicators = {
                            'instrument': instrument,
                            'sma_20': sma_20,
                            'sma_50': sma_50,
                            'rsi': rsi,
                            'volatility': volatility,
                            'trend': trend,
                            'current_price': prices[-1] if prices else 0,
                            'price_history': prices[-20:],  # Last 20 prices
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Store in Redis
                        self.redis_client.hset(
                            f"indicators:{instrument}",
                            "current",
                            json.dumps(indicators)
                        )
                        
                        # Publish to AI
                        self.redis_client.publish(
                            "market:indicators",
                            json.dumps(indicators)
                        )
                        
                        logging.info(f"📈 {instrument}: RSI={rsi:.1f}, Trend={trend}, Vol={volatility:.5f}")
                
                time.sleep(5)  # Calculate every 5 seconds
                
            except Exception as e:
                logging.error(f"Indicator processing error: {e}")
                time.sleep(10)
                
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < 2:
            return 50.0
            
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))
                
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


def main():
    """Main entry point"""
    print("=" * 60)
    print("📊 ULTRATHINK DATA COLLECTOR")
    print("Real-time market data feed for AI processing")
    print("=" * 60)
    
    collector = DataCollector()
    
    try:
        collector.start()
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down Data Collector...")
        collector.stop()
        
    print("✅ Data Collector stopped")


if __name__ == "__main__":
    main()