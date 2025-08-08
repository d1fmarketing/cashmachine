#!/usr/bin/env python3
"""
TRINITY SIMPLE PROFITABLE - Using requests library
No complex dependencies - just make money NOW
"""

import requests
import redis
import json
import time
import logging
from datetime import datetime
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SimpleProfit:
    def __init__(self):
        # OANDA API settings
        self.account_id = "101-001-27538116-001"
        self.api_token = "01cc03ede7cda93a88e87e4e0f1c6912-1cdac97a23c3e1d80f3c8e759e43f4e0"
        self.base_url = "https://api-fxpractice.oanda.com/v3"
        
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        
        # Redis
        self.redis_client = redis.Redis(host='10.100.3.141', port=6379, decode_responses=True)
        
        # PROFITABLE SETTINGS
        self.TAKE_PROFIT_PIPS = 12  # Smaller but achievable
        self.STOP_LOSS_PIPS = 8     # Tight stop loss
        self.LOT_SIZE = 100         # Very small for safety
        
        self.pairs = ['EUR_USD', 'GBP_USD']  # Focus on major pairs
        self.last_trade_time = {}
        
    def get_price(self, pair):
        """Get current price from OANDA"""
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/pricing"
            params = {'instruments': pair}
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                price_data = data['prices'][0]
                bid = float(price_data['bids'][0]['price'])
                ask = float(price_data['asks'][0]['price'])
                
                # Store in Redis
                self.redis_client.set(f'price:{pair}', json.dumps({
                    'bid': bid,
                    'ask': ask,
                    'timestamp': datetime.now().isoformat()
                }), ex=60)
                
                return bid, ask
            else:
                logging.error(f"Price error: {response.status_code}")
                return None, None
                
        except Exception as e:
            logging.error(f"Get price error: {e}")
            return None, None
    
    def simple_signal(self, pair):
        """Very simple momentum signal"""
        try:
            bid, ask = self.get_price(pair)
            if not bid:
                return 0
            
            # Get previous price from Redis
            prev_data = self.redis_client.get(f'prev_price:{pair}')
            if prev_data:
                prev = json.loads(prev_data)
                prev_bid = prev['bid']
                
                # Simple momentum
                change = (bid - prev_bid) / prev_bid * 10000  # In pips
                
                if change > 2:  # Rising fast
                    return 1  # BUY
                elif change < -2:  # Falling fast
                    return -1  # SELL
            
            # Store current as previous for next iteration
            self.redis_client.set(f'prev_price:{pair}', json.dumps({'bid': bid}), ex=300)
            
            return 0
            
        except Exception as e:
            logging.error(f"Signal error: {e}")
            return 0
    
    def place_order(self, pair, direction):
        """Place simple market order"""
        try:
            bid, ask = self.get_price(pair)
            if not bid:
                return None
            
            if direction == "BUY":
                entry = ask
                units = self.LOT_SIZE
                tp = entry + (self.TAKE_PROFIT_PIPS * 0.0001)
                sl = entry - (self.STOP_LOSS_PIPS * 0.0001)
            else:
                entry = bid
                units = -self.LOT_SIZE
                tp = entry - (self.TAKE_PROFIT_PIPS * 0.0001)
                sl = entry + (self.STOP_LOSS_PIPS * 0.0001)
            
            # Create order
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": pair,
                    "units": str(units),
                    "takeProfitOnFill": {
                        "price": f"{tp:.5f}"
                    },
                    "stopLossOnFill": {
                        "price": f"{sl:.5f}"
                    }
                }
            }
            
            url = f"{self.base_url}/accounts/{self.account_id}/orders"
            response = requests.post(url, headers=self.headers, json=order_data)
            
            if response.status_code == 201:
                result = response.json()
                logging.info(f"✅ TRADE: {pair} {direction} @ {entry:.5f} | TP: {tp:.5f} | SL: {sl:.5f}")
                
                # Store trade in Redis
                trade_info = {
                    'pair': pair,
                    'direction': direction,
                    'entry': entry,
                    'tp': tp,
                    'sl': sl,
                    'timestamp': datetime.now().isoformat()
                }
                
                trade_id = result.get('orderFillTransaction', {}).get('id', random.randint(1000, 9999))
                self.redis_client.set(f'trinity:trades:{trade_id}', json.dumps(trade_info), ex=86400)
                
                return trade_id
            else:
                logging.error(f"Order failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Order error: {e}")
            return None
    
    def check_account(self):
        """Check account balance and P&L"""
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/summary"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                balance = float(data['account']['balance'])
                pl = float(data['account'].get('pl', 0))
                
                logging.info(f"💰 Balance: ${balance:.2f} | P&L: ${pl:.2f}")
                
                # Store in Redis
                self.redis_client.set('trinity:balance', str(balance))
                self.redis_client.set('trinity:daily_pnl', str(pl))
                
                return balance, pl
            
        except Exception as e:
            logging.error(f"Account check error: {e}")
            return 0, 0
    
    def run(self):
        """Main trading loop - simple and profitable"""
        logging.info("🚀 TRINITY SIMPLE PROFIT STARTED")
        logging.info(f"Settings: TP={self.TAKE_PROFIT_PIPS} pips | SL={self.STOP_LOSS_PIPS} pips | Size={self.LOT_SIZE}")
        
        # Initial account check
        self.check_account()
        
        while True:
            try:
                # Check emergency stop
                if self.redis_client.get('trinity:emergency_stop') == '1':
                    logging.warning("Emergency stop active")
                    time.sleep(60)
                    continue
                
                # Trade each pair
                for pair in self.pairs:
                    # Rate limit: 1 trade per pair per 5 minutes
                    last_time = self.last_trade_time.get(pair, 0)
                    if time.time() - last_time < 300:
                        continue
                    
                    # Get signal
                    signal = self.simple_signal(pair)
                    
                    if signal > 0:
                        if self.place_order(pair, "BUY"):
                            self.last_trade_time[pair] = time.time()
                    
                    elif signal < 0:
                        if self.place_order(pair, "SELL"):
                            self.last_trade_time[pair] = time.time()
                
                # Check account every 30 seconds
                if int(time.time()) % 30 == 0:
                    self.check_account()
                
                # Wait before next iteration
                time.sleep(5)
                
            except KeyboardInterrupt:
                logging.info("Stopped by user")
                break
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    trader = SimpleProfit()
    trader.run()