import alpaca_trade_api as tradeapi

# REAL ALPACA PAPER TRADING ACCOUNT
API_KEY = 'PKCS3E7GZ5N9UDZEZ5BF'
API_SECRET = 'Y3BeJHStNv0CqonTONiPEfRkWlPdR4aokprHzmSP'
BASE_URL = 'https://paper-api.alpaca.markets'

# Connect to REAL paper trading
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Test with REAL account
account = api.get_account()
print(f"✅ ALPACA REAL PAPER ACCOUNT CONNECTED!")
print(f"💰 Balance: ${account.cash}")
print(f"🔥 Buying Power: ${account.buying_power}")
print(f"📊 Portfolio Value: ${account.portfolio_value}")

# Place a REAL paper trade
order = api.submit_order(
    symbol='SPY',
    qty=1,
    side='buy',
    type='market',
    time_in_force='day'
)
print(f"✅ REAL ORDER PLACED: Buy 1 SPY")
print(f"📝 Order ID: {order.id}")
