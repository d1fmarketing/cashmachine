#!/usr/bin/env python3
"""Test Alpaca API connection with different configurations"""
import os
import sys

# Method 1: Using environment variables (standard way)
def test_env_vars():
    print("Testing with environment variables...")
    os.environ['APCA_API_KEY_ID'] = 'PKV1BCMHP5CRY8M5MW51'
    os.environ['APCA_API_SECRET_KEY'] = 'gx1CjGcxlrUINFP9mxAKVxZ0p7TrfBMMzcdIPVYs'
    os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
    
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST()
        account = api.get_account()
        print(f"✅ SUCCESS! Balance: ${account.cash}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

# Method 2: Direct initialization
def test_direct():
    print("\nTesting with direct initialization...")
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            key_id='PKV1BCMHP5CRY8M5MW51',
            secret_key='gx1CjGcxlrUINFP9mxAKVxZ0p7TrfBMMzcdIPVYs',
            base_url='https://paper-api.alpaca.markets',
            api_version='v2'
        )
        account = api.get_account()
        print(f"✅ SUCCESS! Balance: ${account.cash}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

# Method 3: Try without /v2 in URL
def test_without_v2():
    print("\nTesting without /v2 in URL...")
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            'PKV1BCMHP5CRY8M5MW51',
            'gx1CjGcxlrUINFP9mxAKVxZ0p7TrfBMMzcdIPVYs',
            'https://paper-api.alpaca.markets'
        )
        account = api.get_account()
        print(f"✅ SUCCESS! Balance: ${account.cash}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ALPACA API CONNECTION TEST")
    print("=" * 50)
    
    # Try all methods
    results = []
    results.append(("Env vars", test_env_vars()))
    results.append(("Direct", test_direct()))
    results.append(("No /v2", test_without_v2()))
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    for method, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {method}: {'SUCCESS' if success else 'FAILED'}")