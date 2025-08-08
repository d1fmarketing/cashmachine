#!/usr/bin/env python3
"""
TRINITY PROFITABLE TRADING ENGINE - EMERGENCY FIX
Realistic parameters for immediate profit
"""

import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.accounts import AccountDetails, AccountSummary
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.exceptions import V20Error
import redis
import json
import time
import logging
from datetime import datetime
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ProfitableTrader:
    def __init__(self):
        # OANDA credentials
        self.account_id = "101-001-27538116-001"
        self.api_token = "c0e18946b8655db1424d60dd50299fe8-c8bc1bedc23770af7aed6e30a6f23fa0"
        self.client = API(access_token=self.api_token, environment='practice')
        
        # Redis connection
        self.redis_client = redis.Redis(host='10.100.3.141', port=6379, decode_responses=True)
        
        # REALISTIC PROFITABLE PARAMETERS
        self.TAKE_PROFIT = 15  # 15 pips - achievable
        self.STOP_LOSS = 10    # 10 pips - tight stop
        self.LOT_SIZE = 1000   # Small lot for safety
        
        # Trading pairs
        self.pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']
        self.active_trades = {}
        
    def get_signal(self, pair):
        """Get trading signal from Redis (HRM/ASI/MCTS)"""
        try:
            # Check HRM signal
            hrm_signal = float(self.redis_client.get(f'hrm:signal:{pair}') or 0)
            
            # If HRM is broken (signal=0), use simple price action
            if abs(hrm_signal) < 0.01:
                # Simple momentum strategy
                price_data = self.redis_client.get(f'price:{pair}')
                if price_data:
                    data = json.loads(price_data)
                    current = float(data['bid'])
                    
                    # Get 5-min average
                    avg_key = f'avg_5min:{pair}'
                    avg = float(self.redis_client.get(avg_key) or current)
                    
                    # Update average
                    new_avg = (avg * 0.9) + (current * 0.1)
                    self.redis_client.set(avg_key, str(new_avg), ex=300)
                    
                    # Generate signal based on momentum
                    if current > avg * 1.0001:  # 0.01% above average
                        return 1.0  # BUY
                    elif current < avg * 0.9999:  # 0.01% below average
                        return -1.0  # SELL
            
            return hrm_signal
            
        except Exception as e:
            logging.error(f"Signal error: {e}")
            return 0
    
    def calculate_position_size(self):
        """Conservative position sizing"""
        try:
            # Get account balance
            r = AccountSummary(accountID=self.account_id)
            self.client.request(r)
            balance = float(r.response['account']['balance'])
            
            # Risk only 1% per trade
            risk_amount = balance * 0.01
            position_size = min(self.LOT_SIZE, int(risk_amount / self.STOP_LOSS))
            
            return position_size
            
        except Exception as e:
            logging.error(f"Position size error: {e}")
            return self.LOT_SIZE
    
    def place_order(self, pair, direction):
        """Place order with realistic TP/SL"""
        try:
            # Get current price
            from oandapyV20.endpoints.pricing import PricingInfo
            r = PricingInfo(accountID=self.account_id, params={"instruments": pair})
            self.client.request(r)
            
            price_data = r.response['prices'][0]
            if direction == "BUY":
                entry_price = float(price_data['asks'][0]['price'])
                tp_price = entry_price + (self.TAKE_PROFIT * 0.0001)  # 15 pips
                sl_price = entry_price - (self.STOP_LOSS * 0.0001)    # 10 pips
                units = self.calculate_position_size()
            else:
                entry_price = float(price_data['bids'][0]['price'])
                tp_price = entry_price - (self.TAKE_PROFIT * 0.0001)  # 15 pips
                sl_price = entry_price + (self.STOP_LOSS * 0.0001)    # 10 pips
                units = -self.calculate_position_size()
            
            # Adjust for JPY pairs
            if 'JPY' in pair:
                tp_price = entry_price + (self.TAKE_PROFIT * 0.01) if direction == "BUY" else entry_price - (self.TAKE_PROFIT * 0.01)
                sl_price = entry_price - (self.STOP_LOSS * 0.01) if direction == "BUY" else entry_price + (self.STOP_LOSS * 0.01)
            
            # Create order
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": pair,
                    "units": str(units),
                    "takeProfitOnFill": {"price": f"{tp_price:.5f}"},
                    "stopLossOnFill": {"price": f"{sl_price:.5f}"},
                    "timeInForce": "FOK"
                }
            }
            
            r = OrderCreate(accountID=self.account_id, data=order_data)
            response = self.client.request(r)
            
            trade_id = response['orderFillTransaction']['tradeOpened']['tradeID']
            
            # Log to Redis
            trade_info = {
                'pair': pair,
                'direction': direction,
                'entry': entry_price,
                'tp': tp_price,
                'sl': sl_price,
                'units': units,
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.set(f'trade:{trade_id}', json.dumps(trade_info), ex=86400)
            self.active_trades[trade_id] = trade_info
            
            logging.info(f"✅ TRADE OPENED: {pair} {direction} @ {entry_price:.5f} | TP: {tp_price:.5f} SL: {sl_price:.5f}")
            
            return trade_id
            
        except V20Error as e:
            logging.error(f"Order error: {e}")
            return None
    
    def check_positions(self):
        """Monitor and log P&L"""
        try:
            from oandapyV20.endpoints.trades import TradesList
            r = TradesList(accountID=self.account_id)
            self.client.request(r)
            
            daily_pnl = 0
            for trade in r.response.get('trades', []):
                pnl = float(trade.get('unrealizedPL', 0))
                daily_pnl += pnl
                
                # Update Redis with current P&L
                trade_id = trade['id']
                if trade_id in self.active_trades:
                    self.active_trades[trade_id]['current_pnl'] = pnl
            
            # Store daily P&L
            self.redis_client.set('trinity:daily_pnl', str(daily_pnl))
            
            if daily_pnl != 0:
                status = "🟢 PROFIT" if daily_pnl > 0 else "🔴 LOSS"
                logging.info(f"{status}: Daily P&L = {daily_pnl:.2f}")
            
        except Exception as e:
            logging.error(f"Position check error: {e}")
    
    def run_strategy(self):
        """Main trading loop with realistic logic"""
        logging.info("🚀 PROFITABLE TRINITY STARTED - Realistic Parameters")
        logging.info(f"TP: {self.TAKE_PROFIT} pips | SL: {self.STOP_LOSS} pips")
        
        last_trade_time = {}
        
        while True:
            try:
                # Check if emergency stop is active
                if self.redis_client.get('trinity:emergency_stop') == '1':
                    logging.warning("Emergency stop active - pausing")
                    time.sleep(60)
                    continue
                
                for pair in self.pairs:
                    # Limit trading frequency (1 trade per pair per 5 minutes)
                    last_time = last_trade_time.get(pair, 0)
                    if time.time() - last_time < 300:
                        continue
                    
                    # Get signal
                    signal = self.get_signal(pair)
                    
                    # Trade on strong signals only
                    if signal > 0.5:
                        trade_id = self.place_order(pair, "BUY")
                        if trade_id:
                            last_trade_time[pair] = time.time()
                    
                    elif signal < -0.5:
                        trade_id = self.place_order(pair, "SELL")
                        if trade_id:
                            last_trade_time[pair] = time.time()
                
                # Check positions every iteration
                self.check_positions()
                
                # Wait before next iteration
                time.sleep(10)
                
            except KeyboardInterrupt:
                logging.info("Shutting down...")
                break
            except Exception as e:
                logging.error(f"Strategy error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    trader = ProfitableTrader()
    trader.run_strategy()