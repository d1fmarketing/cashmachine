#!/usr/bin/env python3
"""
Fix for trinity_scalper_secure.py to use paper trading
This patch updates the trading connectors to attempt real paper trading
"""

import sys
import os

def fix_scalper():
    """Update trinity_scalper_secure.py to use real paper trading"""
    
    file_path = '/opt/cashmachine/trinity/trinity_scalper_secure.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace simulated OANDA connection
    content = content.replace(
        '            logger.info("✅ Connected to OANDA (simulated)")',
        '''            # Try real OANDA connection
            try:
                from oandapyV20 import API
                self.oanda_api = API(access_token=token, environment='practice')
                logger.info("✅ Connected to OANDA Practice API (REAL)")
            except Exception as e:
                logger.warning(f"⚠️ OANDA API error: {e}, using simulated mode")
                logger.info("✅ Connected to OANDA (simulated fallback)")'''
    )
    
    # Replace simulated Alpaca connection  
    content = content.replace(
        '            logger.info("✅ Connected to Alpaca (simulated)")',
        '''            # Try real Alpaca connection
            try:
                import alpaca_trade_api as tradeapi
                self.alpaca_api = tradeapi.REST(
                    key_id=key,
                    secret_key=self.creds.get('alpaca', 'secret'),
                    base_url=self.creds.get('alpaca', 'base_url'),
                    api_version='v2'
                )
                account = self.alpaca_api.get_account()
                logger.info(f"✅ Connected to Alpaca Paper API - Balance: ${account.cash}")
            except Exception as e:
                logger.warning(f"⚠️ Alpaca API error: {e}, using simulated mode")
                logger.info("✅ Connected to Alpaca (simulated fallback)")'''
    )
    
    # Update OANDA execute_trade to try real execution
    old_oanda_execute = '''    async def execute_trade(self, symbol: str, units: int, signal: str) -> Dict:
        """Execute forex trade"""
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
        
        # Simulated trading for demonstration
        logger.info(f"📊 OANDA: {signal} {units} units of {symbol}")'''
    
    new_oanda_execute = '''    async def execute_trade(self, symbol: str, units: int, signal: str) -> Dict:
        """Execute forex trade"""
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
        
        # Try real paper trading first
        try:
            if hasattr(self, 'oanda_api'):
                from oandapyV20.endpoints.orders import OrderCreate
                from oandapyV20.contrib.requests import MarketOrderRequest
                
                # Create market order
                if signal.lower() == 'buy':
                    units = abs(units)
                else:
                    units = -abs(units)
                
                order_data = MarketOrderRequest(
                    instrument=symbol.replace('/', '_'),
                    units=units
                )
                
                r = OrderCreate(self.creds.get('oanda', 'account'), data=order_data.data)
                response = self.oanda_api.request(r)
                
                logger.info(f"📊 OANDA PAPER: {signal} {abs(units)} units of {symbol} - ORDER PLACED!")
                return {
                    'status': 'success',
                    'trade_id': response.get('orderCreateTransaction', {}).get('id', 'unknown'),
                    'symbol': symbol,
                    'signal': signal,
                    'units': units,
                    'timestamp': datetime.now().isoformat(),
                    'real_trade': True
                }
        except Exception as e:
            logger.debug(f"OANDA paper trade failed: {e}, using simulation")
        
        # Fallback to simulated trading
        logger.info(f"📊 OANDA SIMULATED: {signal} {units} units of {symbol}")'''
    
    content = content.replace(old_oanda_execute, new_oanda_execute)
    
    # Update Alpaca execute_trade similarly
    old_alpaca_execute = '''    async def execute_trade(self, symbol: str, qty: float, signal: str) -> Dict:
        """Execute stock/crypto trade"""
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
        
        # Simulated trading for demonstration
        logger.info(f"📊 ALPACA: {signal} ${qty} of {symbol}")'''
    
    new_alpaca_execute = '''    async def execute_trade(self, symbol: str, qty: float, signal: str) -> Dict:
        """Execute stock/crypto trade"""
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
        
        # Try real paper trading first
        try:
            if hasattr(self, 'alpaca_api'):
                # Calculate shares from dollar amount
                quote = self.alpaca_api.get_latest_quote(symbol)
                price = quote.ask_price if signal.lower() == 'buy' else quote.bid_price
                shares = int(qty / price)
                
                if shares > 0:
                    order = self.alpaca_api.submit_order(
                        symbol=symbol,
                        qty=shares,
                        side=signal.lower(),
                        type='market',
                        time_in_force='gtc'
                    )
                    
                    logger.info(f"📊 ALPACA PAPER: {signal} {shares} shares of {symbol} @ ${price:.2f} - ORDER PLACED!")
                    return {
                        'status': 'success',
                        'trade_id': order.id,
                        'symbol': symbol,
                        'signal': signal,
                        'notional': qty,
                        'shares': shares,
                        'price': price,
                        'timestamp': datetime.now().isoformat(),
                        'real_trade': True
                    }
        except Exception as e:
            logger.debug(f"Alpaca paper trade failed: {e}, using simulation")
        
        # Fallback to simulated trading
        logger.info(f"📊 ALPACA SIMULATED: {signal} ${qty} of {symbol}")'''
    
    content = content.replace(old_alpaca_execute, new_alpaca_execute)
    
    # Add necessary imports at the top
    if 'from datetime import datetime' not in content:
        content = content.replace(
            'from datetime import datetime',
            'from datetime import datetime\nimport alpaca_trade_api\nfrom oandapyV20 import API'
        )
    
    # Write the fixed version
    backup_path = file_path + '.backup_before_paper'
    fixed_path = file_path + '.paper_fixed'
    
    # Save backup
    with open(file_path, 'r') as f:
        backup_content = f.read()
    with open(backup_path, 'w') as f:
        f.write(backup_content)
    
    # Write fixed version
    with open(fixed_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created fixed version at: {fixed_path}")
    print(f"✅ Backup saved at: {backup_path}")
    print("\nTo apply the fix, run:")
    print(f"  cp {fixed_path} {file_path}")
    print(f"  systemctl restart trinity-scalper")
    
    return fixed_path

if __name__ == "__main__":
    fix_scalper()