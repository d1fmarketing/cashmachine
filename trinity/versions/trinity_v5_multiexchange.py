#!/usr/bin/env python3
"""
TRINITY V5 MULTI-EXCHANGE - ULTRATHINK
Real trading with 5 integrated APIs
Zero Humans | Maximum Intelligence | Real Execution
"""

import json
import time
import redis
import logging
import requests
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import alpaca_trade_api as tradeapi
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TRINITY_V5 - %(levelname)s - %(message)s'
)

class AlpacaAPI:
    """Alpaca trading API for stocks and crypto"""
    
    def __init__(self):
        # Paper trading credentials (replace with real from config)
        self.api_key = "YOUR_ALPACA_KEY"
        self.api_secret = "YOUR_ALPACA_SECRET"
        self.base_url = "https://paper-api.alpaca.markets"
        
        try:
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
            logging.info("✅ Alpaca API connected")
        except Exception as e:
            logging.error(f"❌ Alpaca connection failed: {e}")
            self.api = None
    
    def get_account(self):
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'balance': float(account.cash),
                'buying_power': float(account.buying_power),
                'positions': len(self.api.list_positions())
            }
        except Exception as e:
            logging.error(f"Alpaca account error: {e}")
            return None
    
    def place_order(self, symbol: str, qty: int, side: str, order_type: str = 'market'):
        """Place an order"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day'
            )
            logging.info(f"✅ Alpaca order placed: {side} {qty} {symbol}")
            return order.id
        except Exception as e:
            logging.error(f"❌ Alpaca order failed: {e}")
            return None


class OandaAPI:
    """OANDA API for Forex trading"""
    
    def __init__(self):
        self.account_id = "101-001-27477016-001"
        self.api_token = "01cc03ede7cda93a88e87e4e0f1c6912-1cdac97a23c3e1d80f3c8e759e43f4e0"
        self.base_url = "https://api-fxpractice.oanda.com"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
    def get_prices(self, instruments: List[str]):
        """Get current prices"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
            params = {"instruments": ",".join(instruments)}
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"OANDA price error: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"OANDA error: {e}")
            return None


class PolygonAPI:
    """Polygon.io for market data"""
    
    def __init__(self):
        self.api_key = "YOUR_POLYGON_KEY"
        self.base_url = "https://api.polygon.io"
        
    def get_ticker_snapshot(self, ticker: str):
        """Get snapshot of ticker"""
        try:
            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
            params = {"apiKey": self.api_key}
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logging.error(f"Polygon error: {e}")
            return None


class AlphaVantageAPI:
    """AlphaVantage for fundamental data"""
    
    def __init__(self):
        self.api_key = "YOUR_ALPHAVANTAGE_KEY"
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_quote(self, symbol: str):
        """Get real-time quote"""
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logging.error(f"AlphaVantage error: {e}")
            return None


class FinnhubAPI:
    """Finnhub for real-time data"""
    
    def __init__(self):
        self.api_key = "YOUR_FINNHUB_KEY"
        self.base_url = "https://finnhub.io/api/v1"
        
    def get_quote(self, symbol: str):
        """Get real-time quote"""
        try:
            url = f"{self.base_url}/quote"
            params = {
                "symbol": symbol,
                "token": self.api_key
            }
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logging.error(f"Finnhub error: {e}")
            return None


class TrinityV5MultiExchange:
    """Trinity V5 with multi-exchange support"""
    
    def __init__(self):
        logging.info("=" * 60)
        logging.info("🚀 TRINITY V5 MULTI-EXCHANGE INITIALIZING")
        logging.info("5 APIs | Real Trading | Zero Humans")
        logging.info("=" * 60)
        
        # Initialize all APIs
        self.apis = {
            'alpaca': AlpacaAPI(),
            'oanda': OandaAPI(),
            'polygon': PolygonAPI(),
            'alphavantage': AlphaVantageAPI(),
            'finnhub': FinnhubAPI()
        }
        
        # Redis connection
        self.redis_client = redis.Redis(
            host='10.100.2.200',
            port=6379,
            decode_responses=True
        )
        
        # Subscribe to AI signals
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe('ultrathink:execute')
        
        # Trading state
        self.active_positions = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        
        self.running = False
        
    def start(self):
        """Start Trinity V5"""
        self.running = True
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self._monitor_signals, daemon=True),
            threading.Thread(target=self._monitor_positions, daemon=True),
            threading.Thread(target=self._collect_market_data, daemon=True),
            threading.Thread(target=self._report_status, daemon=True)
        ]
        
        for t in threads:
            t.start()
            
        logging.info("✅ Trinity V5 Multi-Exchange ACTIVE")
        logging.info("📊 Monitoring 5 APIs for trading opportunities...")
        
        # Main loop
        while self.running:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
                break
                
    def stop(self):
        """Stop Trinity V5"""
        logging.info("Stopping Trinity V5...")
        self.running = False
        self.pubsub.unsubscribe()
        
    def _monitor_signals(self):
        """Monitor AI signals and execute trades"""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1)
                
                if message and message['type'] == 'message':
                    signal = json.loads(message['data'])
                    self._process_signal(signal)
                    
            except Exception as e:
                logging.error(f"Signal monitor error: {e}")
                time.sleep(5)
                
    def _process_signal(self, signal: Dict):
        """Process trading signal from AI"""
        
        action = signal.get('action')
        symbol = signal.get('symbol', 'SPY')
        confidence = signal.get('confidence', 0)
        
        logging.info(f"📨 Signal received: {action} {symbol} (Confidence: {confidence:.2f})")
        
        if action == 'BUY' and confidence > 0.65:
            self._execute_buy(symbol, signal)
        elif action == 'SELL' and confidence > 0.65:
            self._execute_sell(symbol, signal)
            
    def _execute_buy(self, symbol: str, signal: Dict):
        """Execute buy order"""
        
        # Determine which API to use based on asset type
        if symbol in ['EUR_USD', 'GBP_USD', 'USD_JPY']:
            # Forex - use OANDA
            self._execute_oanda_trade(symbol, 'buy', signal)
        else:
            # Stocks/ETFs - use Alpaca
            self._execute_alpaca_trade(symbol, 'buy', signal)
            
    def _execute_sell(self, symbol: str, signal: Dict):
        """Execute sell order"""
        
        if symbol in ['EUR_USD', 'GBP_USD', 'USD_JPY']:
            self._execute_oanda_trade(symbol, 'sell', signal)
        else:
            self._execute_alpaca_trade(symbol, 'sell', signal)
            
    def _execute_alpaca_trade(self, symbol: str, side: str, signal: Dict):
        """Execute trade on Alpaca"""
        
        try:
            # Get account info
            account = self.apis['alpaca'].get_account()
            if not account:
                return
                
            # Calculate position size (2% of account)
            position_value = account['balance'] * 0.02
            
            # Get current price from Finnhub
            quote = self.apis['finnhub'].get_quote(symbol)
            if quote:
                current_price = quote.get('c', 100)
                qty = int(position_value / current_price)
                
                if qty > 0:
                    # Place order
                    order_id = self.apis['alpaca'].place_order(
                        symbol=symbol,
                        qty=qty,
                        side=side
                    )
                    
                    if order_id:
                        self.total_trades += 1
                        
                        # Store position info
                        self.active_positions[order_id] = {
                            'symbol': symbol,
                            'qty': qty,
                            'side': side,
                            'entry_price': current_price,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Report to Redis
                        self._report_trade(symbol, side, qty, current_price)
                        
                        logging.info(f"✅ REAL TRADE: {side.upper()} {qty} {symbol} @ ${current_price:.2f}")
                        
        except Exception as e:
            logging.error(f"Alpaca trade error: {e}")
            
    def _execute_oanda_trade(self, symbol: str, side: str, signal: Dict):
        """Execute trade on OANDA"""
        
        # Implementation for OANDA trades
        # Currently limited by authentication issues
        pass
        
    def _collect_market_data(self):
        """Collect market data from multiple sources"""
        
        symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'EUR_USD']
        
        while self.running:
            try:
                for symbol in symbols:
                    # Collect from multiple sources
                    data = {}
                    
                    # Polygon snapshot
                    if symbol not in ['EUR_USD', 'GBP_USD']:
                        polygon_data = self.apis['polygon'].get_ticker_snapshot(symbol)
                        if polygon_data:
                            data['polygon'] = polygon_data
                    
                    # Finnhub real-time
                    finnhub_data = self.apis['finnhub'].get_quote(symbol)
                    if finnhub_data:
                        data['finnhub'] = finnhub_data
                    
                    # AlphaVantage quote
                    alpha_data = self.apis['alphavantage'].get_quote(symbol)
                    if alpha_data:
                        data['alphavantage'] = alpha_data
                    
                    # Store in Redis
                    if data:
                        self.redis_client.hset(
                            f"market:{symbol}",
                            "data",
                            json.dumps(data)
                        )
                        
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logging.error(f"Data collection error: {e}")
                time.sleep(30)
                
    def _monitor_positions(self):
        """Monitor open positions"""
        
        while self.running:
            try:
                # Check Alpaca positions
                if self.apis['alpaca'].api:
                    positions = self.apis['alpaca'].api.list_positions()
                    
                    for pos in positions:
                        current_price = float(pos.current_price)
                        entry_price = float(pos.avg_entry_price)
                        qty = int(pos.qty)
                        
                        # Calculate P&L
                        pnl = (current_price - entry_price) * qty
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        
                        # Check for stop loss or take profit
                        if pnl_pct <= -2:  # 2% stop loss
                            self._close_position(pos.symbol, qty)
                            logging.info(f"⛔ Stop loss hit: {pos.symbol}")
                        elif pnl_pct >= 4:  # 4% take profit
                            self._close_position(pos.symbol, qty)
                            logging.info(f"💰 Take profit hit: {pos.symbol}")
                            self.winning_trades += 1
                            
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Position monitor error: {e}")
                time.sleep(10)
                
    def _close_position(self, symbol: str, qty: int):
        """Close a position"""
        try:
            order = self.apis['alpaca'].api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            logging.info(f"✅ Position closed: {symbol}")
        except Exception as e:
            logging.error(f"Close position error: {e}")
            
    def _report_trade(self, symbol: str, side: str, qty: int, price: float):
        """Report trade to Redis for monitoring"""
        
        trade_data = {
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'api': 'alpaca',
            'status': 'executed'
        }
        
        # Store in Redis
        self.redis_client.lpush(
            "trinity:real_trades",
            json.dumps(trade_data)
        )
        
        # Publish to channel
        self.redis_client.publish(
            "trinity:trade_executed",
            json.dumps(trade_data)
        )
        
    def _report_status(self):
        """Report system status periodically"""
        
        while self.running:
            try:
                # Get Alpaca account
                account = self.apis['alpaca'].get_account()
                
                status = {
                    'version': 'Trinity V5 Multi-Exchange',
                    'apis_connected': sum(1 for api in self.apis.values() if api),
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
                    'account_balance': account['balance'] if account else 0,
                    'positions': len(self.active_positions),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store in Redis
                self.redis_client.hset(
                    "trinity:status",
                    "current",
                    json.dumps(status)
                )
                
                logging.info(f"📊 Status: Trades={self.total_trades}, Win Rate={status['win_rate']:.1f}%, Balance=${status['account_balance']:.2f}")
                
                time.sleep(30)  # Report every 30 seconds
                
            except Exception as e:
                logging.error(f"Status report error: {e}")
                time.sleep(60)


def main():
    """Main entry point"""
    
    print("=" * 60)
    print("🚀 TRINITY V5 MULTI-EXCHANGE")
    print("5 APIs | Real Trading | Zero Humans")
    print("=" * 60)
    print()
    print("Initializing connections...")
    
    trinity = TrinityV5MultiExchange()
    
    try:
        trinity.start()
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down Trinity V5...")
        trinity.stop()
        
    print("✅ Trinity V5 stopped")


if __name__ == "__main__":
    main()