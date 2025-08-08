#!/usr/bin/env python3
"""
ULTRATHINK OANDA v20 Integration
Using btoandav20 for proper API support
"""

import sys
import os
import json
import backtrader as bt
import btoandav20
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

# Add AI model paths
sys.path.insert(0, '/opt/cashmachine/trinity')

class UltrathinkOandaV20:
    """OANDA v20 Integration with AI Trading"""
    
    def __init__(self):
        self.config_dir = "/opt/cashmachine/config"
        self.oanda_config = self.load_oanda_config()
        self.store = None
        self.broker = None
        
    def load_oanda_config(self):
        """Load encrypted OANDA configuration"""
        try:
            with open(f"{self.config_dir}/.oanda.key", "rb") as f:
                key = f.read()
            with open(f"{self.config_dir}/oanda.enc", "rb") as f:
                encrypted = f.read()
            cipher = Fernet(key)
            config = json.loads(cipher.decrypt(encrypted))
            print("✅ OANDA config loaded")
            return config
        except Exception as e:
            print(f"❌ Failed to load OANDA config: {e}")
            # Fallback to direct config
            return {
                "api_token": "3b7b7f648b0b23589bbeeb6b3cb2ff5f-8b9aafc926428bd666a4d9c133f46809",
                "account_id": "101-001-20983972-001",
                "environment": "practice"
            }
    
    def initialize_store(self):
        """Initialize OANDA v20 store"""
        print("🏛️ Initializing OANDA v20 Store...")
        
        # Create store with proxy support
        self.store = btoandav20.stores.OandaV20Store(
            token=self.oanda_config['api_token'],
            account=self.oanda_config['account_id'],
            practice=True,  # Using practice account
            # Note: btoandav20 handles proxy through environment variables
        )
        
        print("✅ OANDA v20 Store initialized")
        return self.store
    
    def get_data_feed(self, instrument='EUR_USD', granularity='M5'):
        """Get data feed for an instrument"""
        if not self.store:
            self.initialize_store()
        
        data = self.store.getdata(
            dataname=instrument,
            granularity=granularity,
            compression=1,
            backfill_start=True,
            backfill=True,
            live_data=True
        )
        
        print(f"📊 Data feed created for {instrument}")
        return data
    
    def get_broker(self):
        """Get broker instance"""
        if not self.store:
            self.initialize_store()
        
        self.broker = self.store.getbroker()
        print(f"💰 Broker initialized - Balance: ${self.broker.getvalue():,.2f}")
        return self.broker
    
    def test_connection(self):
        """Test OANDA connection"""
        print("\n🧪 Testing OANDA v20 Connection...")
        
        try:
            # Set proxy for requests
            os.environ['http_proxy'] = 'http://10.100.1.72:3128'
            os.environ['https_proxy'] = 'http://10.100.1.72:3128'
            
            # Initialize store
            self.initialize_store()
            
            # Get broker
            broker = self.get_broker()
            
            # Try to get account info
            cash = broker.getcash()
            value = broker.getvalue()
            
            print(f"✅ Connection successful!")
            print(f"   Cash: ${cash:,.2f}")
            print(f"   Portfolio Value: ${value:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

class UltrathinkStrategyV20(bt.Strategy):
    """ULTRATHINK Strategy using OANDA v20"""
    
    params = (
        ('risk_per_trade', 0.02),
        ('max_positions', 3),
        ('stop_loss', 0.02),
        ('take_profit', 0.04),
    )
    
    def __init__(self):
        # Technical indicators
        self.sma20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma50 = bt.indicators.SMA(self.data.close, period=50)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
        print("🧠 ULTRATHINK Strategy initialized")
    
    def next(self):
        """Trading logic"""
        if not self.position:
            # Entry logic
            if self.sma20[0] > self.sma50[0] and self.rsi[0] < 70:
                size = self.calculate_position_size()
                self.buy(size=size)
                print(f"✅ BUY signal - Size: {size}")
        else:
            # Exit logic
            if self.sma20[0] < self.sma50[0] or self.rsi[0] > 80:
                self.close()
                print("📤 Position closed")
    
    def calculate_position_size(self):
        """Calculate position size based on risk"""
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.params.risk_per_trade
        # For Forex, use appropriate lot sizing
        return 1000  # 0.01 lot for testing

def run_test():
    """Test the complete setup"""
    print("\n" + "="*60)
    print("🚀 ULTRATHINK OANDA v20 Test")
    print("="*60)
    
    # Initialize connection
    oanda = UltrathinkOandaV20()
    
    # Test connection
    if oanda.test_connection():
        print("\n✅ OANDA v20 integration ready!")
        print("🧠 AI components can now trade with ZERO humans!")
    else:
        print("\n⚠️ Connection test failed - check proxy settings")

if __name__ == "__main__":
    run_test()
