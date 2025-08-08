#!/usr/bin/env python3
"""
TRINITY V4 - AI INTEGRATED
Real AI-powered trading with ULTRATHINK Brain integration
Zero Humans | Maximum Intelligence
"""

import json
import time
import redis
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TRINITY_V4 - %(levelname)s - %(message)s'
)

class TrinityV4AI:
    """Trinity V4 with full AI integration"""
    
    def __init__(self):
        logging.info("🚀 Initializing Trinity V4 AI...")
        
        # Redis connection for AI communication
        self.redis_client = redis.Redis(
            host='10.100.2.200',
            port=6379,
            decode_responses=True
        )
        
        # Subscribe to AI signals
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe('ultrathink:execute')
        
        # OANDA Configuration
        self.oanda_url = "https://api-fxpractice.oanda.com"
        self.account_id = "101-001-27477016-001"
        self.api_key = "01cc03ede7cda93a88e87e4e0f1c6912-1cdac97a23c3e1d80f3c8e759e43f4e0"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Trading metrics
        self.trades_executed = 0
        self.trades_won = 0
        self.trades_lost = 0
        self.total_profit = 0
        
        self.running = False
        
    def start(self):
        """Start Trinity V4 AI"""
        logging.info("=" * 60)
        logging.info("🤖 TRINITY V4 AI - ULTRATHINK INTEGRATED")
        logging.info("AI-Powered Trading with Zero Human Intervention")
        logging.info("=" * 60)
        
        self.running = True
        
        # Main trading loop
        while self.running:
            try:
                # Listen for AI signals
                message = self.pubsub.get_message(timeout=1)
                
                if message and message['type'] == 'message':
                    signal = json.loads(message['data'])
                    self.process_ai_signal(signal)
                    
                # Check open positions
                self.monitor_positions()
                
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                time.sleep(5)
                
    def stop(self):
        """Stop Trinity V4"""
        logging.info("Stopping Trinity V4 AI...")
        self.running = False
        self.pubsub.unsubscribe()
        
    def process_ai_signal(self, signal: Dict):
        """Process signal from ULTRATHINK Brain"""
        
        logging.info(f"🧠 AI Signal received: {signal['action']} {signal.get('direction', '')} (Confidence: {signal.get('confidence', 0):.2f})")
        
        if signal['action'] == 'TRADE':
            self.execute_trade(signal)
        elif signal['action'] == 'CLOSE':
            self.close_position(signal.get('instrument'))
        elif signal['action'] == 'MODIFY':
            self.modify_position(signal)
            
    def execute_trade(self, signal: Dict):
        """Execute trade based on AI decision"""
        
        instrument = signal.get('instrument', 'EUR_USD')
        direction = signal.get('direction', 'BUY')
        units = self.calculate_position_size(signal)
        take_profit = signal.get('take_profit', 30)
        stop_loss = signal.get('stop_loss', 15)
        
        # Create order
        order = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": units if direction == 'BUY' else -units,
                "takeProfitOnFill": {
                    "price": self.calculate_tp_price(instrument, direction, take_profit)
                },
                "stopLossOnFill": {
                    "distance": str(stop_loss * 0.0001)
                }
            }
        }
        
        # Send order
        url = f"{self.oanda_url}/v3/accounts/{self.account_id}/orders"
        response = requests.post(url, headers=self.headers, json=order)
        
        if response.status_code == 201:
            result = response.json()
            order_id = result.get('orderFillTransaction', {}).get('id')
            
            self.trades_executed += 1
            
            logging.info(f"✅ Trade #{self.trades_executed} executed: {direction} {units} {instrument}")
            logging.info(f"   TP: {take_profit} pips, SL: {stop_loss} pips")
            
            # Report to brain
            self.report_execution(order_id, signal)
        else:
            logging.error(f"❌ Trade failed: {response.text}")
            
    def calculate_position_size(self, signal: Dict) -> int:
        """Calculate position size based on risk management"""
        
        # Get account balance
        balance = self.get_account_balance()
        
        # Risk per trade (from AI or default)
        risk_percent = signal.get('risk_per_trade', 0.02)
        
        # Calculate units (simplified)
        risk_amount = balance * risk_percent
        stop_loss_pips = signal.get('stop_loss', 15)
        
        # Basic calculation (would be more complex in production)
        units = int(risk_amount * 100)  # Simplified
        
        # Apply limits
        max_units = 10000  # Max position size
        units = min(units, max_units)
        
        return units
        
    def calculate_tp_price(self, instrument: str, direction: str, tp_pips: int) -> str:
        """Calculate take profit price"""
        
        # Get current price
        current_price = self.get_current_price(instrument)
        
        if direction == 'BUY':
            tp_price = current_price + (tp_pips * 0.0001)
        else:
            tp_price = current_price - (tp_pips * 0.0001)
            
        return f"{tp_price:.5f}"
        
    def get_current_price(self, instrument: str) -> float:
        """Get current market price"""
        
        url = f"{self.oanda_url}/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'prices' in data and len(data['prices']) > 0:
                price_data = data['prices'][0]
                bid = float(price_data.get('bids', [{'price': '0'}])[0]['price'])
                ask = float(price_data.get('asks', [{'price': '0'}])[0]['price'])
                return (bid + ask) / 2
                
        return 0.0
        
    def get_account_balance(self) -> float:
        """Get account balance"""
        
        url = f"{self.oanda_url}/v3/accounts/{self.account_id}/summary"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            return float(data.get('account', {}).get('balance', 10000))
            
        return 10000  # Default
        
    def monitor_positions(self):
        """Monitor open positions and report to AI"""
        
        url = f"{self.oanda_url}/v3/accounts/{self.account_id}/openTrades"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            trades = data.get('trades', [])
            
            for trade in trades:
                trade_id = trade.get('id')
                instrument = trade.get('instrument')
                unrealized_pl = float(trade.get('unrealizedPL', 0))
                
                # Report to AI for learning
                self.redis_client.hset(
                    f"trinity:position:{trade_id}",
                    "current",
                    json.dumps({
                        'id': trade_id,
                        'instrument': instrument,
                        'unrealized_pl': unrealized_pl,
                        'timestamp': datetime.now().isoformat()
                    })
                )
                
    def close_position(self, instrument: str):
        """Close position for instrument"""
        
        url = f"{self.oanda_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"
        data = {"longUnits": "ALL", "shortUnits": "ALL"}
        
        response = requests.put(url, headers=self.headers, json=data)
        
        if response.status_code == 200:
            logging.info(f"✅ Closed position: {instrument}")
        else:
            logging.error(f"❌ Failed to close {instrument}: {response.text}")
            
    def modify_position(self, signal: Dict):
        """Modify existing position based on AI signal"""
        
        trade_id = signal.get('trade_id')
        new_tp = signal.get('new_take_profit')
        new_sl = signal.get('new_stop_loss')
        
        if not trade_id:
            return
            
        url = f"{self.oanda_url}/v3/accounts/{self.account_id}/trades/{trade_id}/orders"
        
        data = {}
        if new_tp:
            data['takeProfit'] = {"price": str(new_tp)}
        if new_sl:
            data['stopLoss'] = {"distance": str(new_sl * 0.0001)}
            
        response = requests.put(url, headers=self.headers, json=data)
        
        if response.status_code == 200:
            logging.info(f"✅ Modified trade {trade_id}")
        else:
            logging.error(f"❌ Failed to modify trade: {response.text}")
            
    def report_execution(self, order_id: str, signal: Dict):
        """Report trade execution back to AI"""
        
        execution_data = {
            'order_id': order_id,
            'signal': signal,
            'executed_at': datetime.now().isoformat(),
            'trinity_version': 'v4_ai'
        }
        
        # Store in Redis for AI to learn
        self.redis_client.lpush(
            "trinity:executions",
            json.dumps(execution_data)
        )
        
        # Publish confirmation
        self.redis_client.publish(
            "trinity:confirmations",
            json.dumps({
                'type': 'execution_confirmed',
                'order_id': order_id
            })
        )
        
    def report_stats(self):
        """Report trading statistics"""
        
        win_rate = (self.trades_won / max(1, self.trades_executed)) * 100
        
        stats = {
            'version': 'Trinity V4 AI',
            'trades_executed': self.trades_executed,
            'trades_won': self.trades_won,
            'trades_lost': self.trades_lost,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'status': 'AI_INTEGRATED',
            'timestamp': datetime.now().isoformat()
        }
        
        # Store stats
        self.redis_client.hset(
            "trinity:stats",
            "current",
            json.dumps(stats)
        )
        
        logging.info(f"📊 Stats: Trades={self.trades_executed}, Win Rate={win_rate:.1f}%, P&L={self.total_profit:.1f}")


def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 TRINITY V4 - ULTRATHINK AI INTEGRATED")
    print("Real AI-Powered Trading System")
    print("=" * 60)
    print()
    print("Connecting to ULTRATHINK Brain...")
    
    trinity = TrinityV4AI()
    
    try:
        trinity.start()
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down Trinity V4...")
        trinity.stop()
        
    print("✅ Trinity V4 stopped")


if __name__ == "__main__":
    main()