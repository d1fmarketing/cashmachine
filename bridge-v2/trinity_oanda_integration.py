#!/usr/bin/env python3
"""
Trinity System OANDA Integration Module
Advanced trading operations interface for CashMachine platform
Implements hybrid human-AI trading system with OANDA API
"""

import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

from oanda_auth_manager import OANDAAuthManager
from oanda_proxy_config import OANDAProxyConfig

class OrderType(Enum):
    """OANDA order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"
    LIMIT_IF_TOUCHED = "LIMIT_IF_TOUCHED"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP_LOSS = "TRAILING_STOP_LOSS"

class TimeInForce(Enum):
    """Time in force options"""
    FOK = "FOK"  # Fill or Kill
    IOC = "IOC"  # Immediate or Cancel
    GTC = "GTC"  # Good Till Cancelled
    GTD = "GTD"  # Good Till Date

@dataclass
class TradingSignal:
    """Trading signal from Trinity system"""
    instrument: str
    action: str  # BUY, SELL
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    timestamp: Optional[str] = None
    source: str = "TRINITY"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_position_size: float = 0.02  # 2% of account per trade
    max_daily_risk: float = 0.05     # 5% of account per day
    max_drawdown: float = 0.10       # 10% maximum drawdown
    correlation_limit: float = 0.7   # Maximum correlation between positions
    max_positions: int = 10          # Maximum concurrent positions
    stop_loss_pct: float = 0.02      # 2% stop loss
    take_profit_pct: float = 0.04    # 4% take profit (2:1 RR)
    
class TrinityOANDAIntegration:
    """Main integration class for Trinity system with OANDA"""
    
    def __init__(self, auth_manager: OANDAAuthManager = None, 
                 proxy_config: OANDAProxyConfig = None,
                 risk_params: RiskParameters = None):
        """
        Initialize Trinity OANDA integration
        
        Args:
            auth_manager: OANDA authentication manager
            proxy_config: Proxy configuration manager
            risk_params: Risk management parameters
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.auth_manager = auth_manager or OANDAAuthManager()
        self.proxy_config = proxy_config or OANDAProxyConfig()
        self.risk_params = risk_params or RiskParameters()
        
        # Trading state
        self.credentials = None
        self.session_data = None
        self.account_info = None
        self.positions = {}
        self.orders = {}
        self.instruments = {}
        
        # Performance tracking
        self.trade_history = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Initialize state
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize OANDA session and load account data"""
        try:
            # Load credentials
            self.credentials = self.auth_manager.load_credentials()
            self.logger.info(f"Loaded credentials for {self.credentials['environment']} environment")
            
            # Create session
            self.session_data = self.auth_manager.create_session(self.credentials)
            
            # Load account information
            self.account_info = self.auth_manager.get_account_info(self.credentials)
            
            # Update proxy configuration for blackbox if needed
            if self.credentials['environment'] == 'practice':
                self.proxy_config.configure_for_blackbox()
            
            self.logger.info("Trinity OANDA integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize session: {e}")
            raise
    
    async def get_market_data(self, instruments: List[str], 
                            count: int = 500, granularity: str = "M1") -> Dict[str, pd.DataFrame]:
        """
        Get historical market data for instruments
        
        Args:
            instruments: List of instrument names (e.g., ['EUR_USD', 'GBP_USD'])
            count: Number of candles to retrieve
            granularity: Candle granularity (S5, S10, S15, S30, M1, M5, M15, M30, H1, H4, D)
            
        Returns:
            Dictionary mapping instrument names to DataFrames
        """
        market_data = {}
        
        try:
            headers = self.auth_manager.get_auth_headers(self.credentials)
            endpoint = self.session_data['endpoint']
            
            for instrument in instruments:
                url = f"{endpoint}/v3/instruments/{instrument}/candles"
                params = {
                    'count': count,
                    'granularity': granularity,
                    'price': 'MBA'  # Mid, Bid, Ask
                }
                
                response = self.proxy_config.make_request(
                    'GET', url, headers, params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    candles = data.get('candles', [])
                    
                    # Convert to DataFrame
                    df_data = []
                    for candle in candles:
                        if candle['complete']:
                            df_data.append({
                                'time': pd.to_datetime(candle['time']),
                                'open': float(candle['mid']['o']),
                                'high': float(candle['mid']['h']),
                                'low': float(candle['mid']['l']),
                                'close': float(candle['mid']['c']),
                                'volume': int(candle['volume']),
                                'bid_open': float(candle['bid']['o']),
                                'bid_high': float(candle['bid']['h']),
                                'bid_low': float(candle['bid']['l']),
                                'bid_close': float(candle['bid']['c']),
                                'ask_open': float(candle['ask']['o']),
                                'ask_high': float(candle['ask']['h']),
                                'ask_low': float(candle['ask']['l']),
                                'ask_close': float(candle['ask']['c'])
                            })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('time', inplace=True)
                    market_data[instrument] = df
                    
                    self.logger.info(f"Retrieved {len(df)} candles for {instrument}")
                    
                else:
                    self.logger.error(f"Failed to get data for {instrument}: {response.status_code}")
                    
                # Rate limiting
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            raise
        
        return market_data
    
    async def get_current_prices(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get current bid/ask prices for instruments
        
        Args:
            instruments: List of instrument names
            
        Returns:
            Dictionary mapping instruments to price data
        """
        try:
            headers = self.auth_manager.get_auth_headers(self.credentials)
            endpoint = self.session_data['endpoint']
            
            instruments_str = ','.join(instruments)
            url = f"{endpoint}/v3/accounts/{self.credentials['account_id']}/pricing"
            params = {'instruments': instruments_str}
            
            response = self.proxy_config.make_request(
                'GET', url, headers, params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                prices = {}
                
                for price_data in data.get('prices', []):
                    instrument = price_data['instrument']
                    prices[instrument] = {
                        'bid': float(price_data['closeoutBid']),
                        'ask': float(price_data['closeoutAsk']),
                        'mid': (float(price_data['closeoutBid']) + float(price_data['closeoutAsk'])) / 2,
                        'spread': float(price_data['closeoutAsk']) - float(price_data['closeoutBid']),
                        'timestamp': price_data['time']
                    }
                
                return prices
            else:
                raise Exception(f"Failed to get prices: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error getting current prices: {e}")
            raise
    
    def calculate_position_size(self, instrument: str, signal: TradingSignal,
                              current_price: float) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            instrument: Trading instrument
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            Position size in units
        """
        try:
            # Get account balance
            account_balance = float(self.account_info['account']['balance'])
            
            # Calculate risk amount (percentage of balance)
            risk_amount = account_balance * self.risk_params.max_position_size
            
            # Calculate stop loss distance
            if signal.stop_loss:
                stop_distance = abs(current_price - signal.stop_loss)
            else:
                stop_distance = current_price * self.risk_params.stop_loss_pct
            
            # Calculate position size based on risk
            if stop_distance > 0:
                # For forex, need to consider pip value
                if '_' in instrument:  # Forex pair
                    # Simplified calculation - in practice, need proper pip value calculation
                    pip_value = 0.0001 if 'JPY' not in instrument else 0.01
                    position_size = risk_amount / (stop_distance / pip_value)
                else:
                    position_size = risk_amount / stop_distance
            else:
                position_size = 1000  # Default small position
            
            # Apply confidence adjustment
            position_size *= signal.confidence
            
            # Apply maximum position size limit
            max_size = account_balance * self.risk_params.max_position_size / current_price
            position_size = min(position_size, max_size)
            
            # Round to appropriate precision
            position_size = round(position_size, 0)
            
            self.logger.info(f"Calculated position size for {instrument}: {position_size} units")
            return max(1, position_size)  # Minimum 1 unit
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1000  # Default fallback
    
    async def execute_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Execute a trading signal from Trinity system
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            Execution result dictionary
        """
        try:
            self.logger.info(f"Executing signal: {signal.instrument} {signal.action} confidence={signal.confidence}")
            
            # Validate signal
            if signal.confidence < 0.5:
                return {"success": False, "error": "Signal confidence too low"}
            
            # Get current prices
            prices = await self.get_current_prices([signal.instrument])
            if signal.instrument not in prices:
                return {"success": False, "error": "Unable to get current price"}
            
            current_price = prices[signal.instrument]['mid']
            
            # Calculate position size
            position_size = self.calculate_position_size(signal.instrument, signal, current_price)
            
            # Prepare order
            order_data = {
                "order": {
                    "type": OrderType.MARKET.value,
                    "instrument": signal.instrument,
                    "units": str(int(position_size)) if signal.action == "BUY" else str(-int(position_size)),
                    "timeInForce": TimeInForce.FOK.value,
                    "positionFill": "DEFAULT"
                }
            }
            
            # Add stop loss if provided
            if signal.stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(signal.stop_loss),
                    "timeInForce": TimeInForce.GTC.value
                }
            
            # Add take profit if provided
            if signal.take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(signal.take_profit),
                    "timeInForce": TimeInForce.GTC.value
                }
            
            # Execute order
            headers = self.auth_manager.get_auth_headers(self.credentials)
            endpoint = self.session_data['endpoint']
            url = f"{endpoint}/v3/accounts/{self.credentials['account_id']}/orders"
            
            response = self.proxy_config.make_request(
                'POST', url, headers, json=order_data
            )
            
            if response.status_code == 201:
                result = response.json()
                
                # Record trade
                trade_record = {
                    "signal": asdict(signal),
                    "execution": result,
                    "timestamp": datetime.utcnow().isoformat(),
                    "position_size": position_size,
                    "current_price": current_price
                }
                self.trade_history.append(trade_record)
                
                self.logger.info(f"Signal executed successfully: {result.get('orderFillTransaction', {}).get('id', 'N/A')}")
                
                return {
                    "success": True,
                    "order_id": result.get('orderCreateTransaction', {}).get('id'),
                    "fill_id": result.get('orderFillTransaction', {}).get('id'),
                    "execution_price": result.get('orderFillTransaction', {}).get('price'),
                    "units_filled": result.get('orderFillTransaction', {}).get('units')
                }
            else:
                error_msg = f"Order execution failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error executing signal: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current open positions"""
        try:
            headers = self.auth_manager.get_auth_headers(self.credentials)
            endpoint = self.session_data['endpoint']
            url = f"{endpoint}/v3/accounts/{self.credentials['account_id']}/positions"
            
            response = self.proxy_config.make_request('GET', url, headers)
            
            if response.status_code == 200:
                data = response.json()
                positions = {}
                
                for position in data.get('positions', []):
                    if float(position['long']['units']) != 0 or float(position['short']['units']) != 0:
                        positions[position['instrument']] = {
                            'long_units': float(position['long']['units']),
                            'short_units': float(position['short']['units']),
                            'unrealized_pl': float(position['unrealizedPL']),
                            'margin_used': float(position['marginUsed'])
                        }
                
                self.positions = positions
                return positions
            else:
                raise Exception(f"Failed to get positions: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            raise
    
    async def close_position(self, instrument: str, units: str = "ALL") -> Dict[str, Any]:
        """
        Close a position
        
        Args:
            instrument: Instrument to close
            units: Units to close ("ALL" or specific number)
            
        Returns:
            Close result dictionary
        """
        try:
            headers = self.auth_manager.get_auth_headers(self.credentials)
            endpoint = self.session_data['endpoint']
            
            # Determine which side to close
            positions = await self.get_positions()
            if instrument not in positions:
                return {"success": False, "error": "No position found"}
            
            position = positions[instrument]
            
            if position['long_units'] > 0:
                url = f"{endpoint}/v3/accounts/{self.credentials['account_id']}/positions/{instrument}/close"
                data = {"longUnits": units}
            elif position['short_units'] < 0:
                url = f"{endpoint}/v3/accounts/{self.credentials['account_id']}/positions/{instrument}/close"
                data = {"shortUnits": units}
            else:
                return {"success": False, "error": "No units to close"}
            
            response = self.proxy_config.make_request('PUT', url, headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Position closed: {instrument}")
                return {"success": True, "result": result}
            else:
                error_msg = f"Failed to close position: {response.status_code}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Error closing position: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            account_balance = float(self.account_info['account']['balance'])
            nav = float(self.account_info['account']['NAV'])
            unrealized_pl = float(self.account_info['account']['unrealizedPL'])
            
            # Calculate metrics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history 
                               if trade.get('execution', {}).get('pl', 0) > 0)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "account_balance": account_balance,
                "nav": nav,
                "unrealized_pl": unrealized_pl,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "max_drawdown": self.max_drawdown,
                "daily_pnl": self.daily_pnl,
                "open_positions": len(self.positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}
    
    async def risk_check(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Perform risk checks before executing signal
        
        Args:
            signal: Trading signal to check
            
        Returns:
            Tuple of (is_approved, reason)
        """
        try:
            # Check daily risk limit
            if abs(self.daily_pnl) > self.risk_params.max_daily_risk:
                return False, "Daily risk limit exceeded"
            
            # Check maximum positions
            positions = await self.get_positions()
            if len(positions) >= self.risk_params.max_positions:
                return False, "Maximum positions limit reached"
            
            # Check drawdown
            account_balance = float(self.account_info['account']['balance'])
            if self.max_drawdown > self.risk_params.max_drawdown * account_balance:
                return False, "Maximum drawdown exceeded"
            
            # Check correlation (simplified)
            if signal.instrument in positions:
                return False, "Position already exists for this instrument"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            return False, f"Risk check error: {e}"
    
    async def emergency_stop(self):
        """Emergency stop - close all positions and cancel all orders"""
        try:
            self.logger.warning("EMERGENCY STOP ACTIVATED")
            
            # Get all positions
            positions = await self.get_positions()
            
            # Close all positions
            close_results = []
            for instrument in positions:
                result = await self.close_position(instrument)
                close_results.append(result)
            
            # Cancel all pending orders
            headers = self.auth_manager.get_auth_headers(self.credentials)
            endpoint = self.session_data['endpoint']
            
            # Get pending orders
            orders_url = f"{endpoint}/v3/accounts/{self.credentials['account_id']}/orders"
            response = self.proxy_config.make_request('GET', orders_url, headers)
            
            if response.status_code == 200:
                orders_data = response.json()
                for order in orders_data.get('orders', []):
                    cancel_url = f"{endpoint}/v3/accounts/{self.credentials['account_id']}/orders/{order['id']}/cancel"
                    self.proxy_config.make_request('PUT', cancel_url, headers)
            
            self.logger.warning("Emergency stop completed")
            return {"success": True, "positions_closed": len(close_results)}
            
        except Exception as e:
            self.logger.error(f"Error in emergency stop: {e}")
            return {"success": False, "error": str(e)}


async def main():
    """Example usage of Trinity OANDA Integration"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize integration
        trinity = TrinityOANDAIntegration()
        
        # Example signal
        signal = TradingSignal(
            instrument="EUR_USD",
            action="BUY",
            confidence=0.85,
            stop_loss=1.0500,
            take_profit=1.0600
        )
        
        # Execute signal
        result = await trinity.execute_signal(signal)
        print(f"Execution result: {result}")
        
        # Get performance metrics
        metrics = trinity.get_performance_metrics()
        print(f"Performance: {json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    asyncio.run(main())