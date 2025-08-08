#!/usr/bin/env python3
"""
Fix Alpaca connection using the official library
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv('/opt/cashmachine/trinity/.env')

print("🔧 FIXING ALPACA CONNECTION WITH OFFICIAL LIBRARY")
print("=" * 60)

# Try importing the library
try:
    import alpaca_trade_api as tradeapi
    print("✅ alpaca_trade_api library loaded")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test both sets of credentials
configs = [
    {
        'key': os.environ.get('ALPACA_API_KEY', 'PKV1BCMHP5CRY8M5MW51'),
        'secret': os.environ.get('ALPACA_API_SECRET', 'gx1CjGcxlrUINFP9mxAKVxZ0p7TrfBMMzcdIPVYs'),
        'source': '.env'
    },
    {
        'key': 'PKCS3E7GZ5N9UDZEZ5BF',
        'secret': 'Y3BeJHStNv0CqonTONiPEfRkWlPdR4aokprHzmSP',
        'source': 'real_alpaca_config.py'
    }
]

for config in configs:
    print(f"\n📊 Testing: {config['source']}")
    print(f"Key: {config['key'][:15]}...")
    
    try:
        # Use the official library
        api = tradeapi.REST(
            config['key'],
            config['secret'],
            'https://paper-api.alpaca.markets',
            api_version='v2'
        )
        
        # Get account
        account = api.get_account()
        
        print(f"""
        ✅✅✅ CONNECTED SUCCESSFULLY! ✅✅✅
        
        💰 Account Details:
        Cash: ${account.cash}
        Buying Power: ${account.buying_power}
        Equity: ${account.equity}
        Pattern Day Trader: {account.pattern_day_trader}
        Trading Blocked: {account.trading_blocked}
        Account Number: {account.account_number}
        Status: {account.status}
        """)
        
        # Get recent orders
        orders = api.list_orders(limit=5)
        print(f"📈 Recent Orders: {len(orders)}")
        for order in orders[:3]:
            print(f"  - {order.side} {order.qty} {order.symbol} - {order.status}")
        
        # Get positions
        positions = api.list_positions()
        print(f"📊 Open Positions: {len(positions)}")
        
        # PLACE A TEST TRADE
        print("\n🚀 PLACING REAL PAPER TRADE...")
        
        try:
            order = api.submit_order(
                symbol='AAPL',
                qty=1,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            
            print(f"""
            ✅✅✅ REAL TRADE EXECUTED! ✅✅✅
            Order ID: {order.id}
            Symbol: {order.symbol}
            Qty: {order.qty}
            Side: {order.side}
            Status: {order.status}
            Created: {order.created_at}
            """)
            
            # Save working config
            with open('/opt/cashmachine/trinity/WORKING_ALPACA_CONFIG.txt', 'w') as f:
                f.write(f"API_KEY={config['key']}\n")
                f.write(f"API_SECRET={config['secret']}\n")
                f.write(f"BASE_URL=https://paper-api.alpaca.markets\n")
                f.write(f"SOURCE={config['source']}\n")
            
            print("✅ Working configuration saved!")
            
        except Exception as e:
            print(f"Trade failed: {e}")
        
        break  # Found working config
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        if "forbidden" in str(e).lower():
            print("   API key is rejected - may need to be regenerated")
        elif "401" in str(e):
            print("   Authentication failed - check credentials")
        else:
            print(f"   Error details: {str(e)[:200]}")

print("\n" + "=" * 60)