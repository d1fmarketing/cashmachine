#!/usr/bin/env python3
"""
ULTRATHINK Alpha Vantage-Backtrader Bridge
Market data & economic indicators for AI trading
"""

import backtrader as bt
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.sectorperformance import SectorPerformances
from datetime import datetime, timedelta
import json
import os
from cryptography.fernet import Fernet
from typing import Dict, Optional

class AlphaVantageStore:
    """Alpha Vantage data store for Backtrader"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key or load from encrypted config"""
        
        if not api_key:
            # Load from encrypted config
            api_key = self.load_api_key()
        
        self.api_key = api_key
        
        # Set proxy
        os.environ['http_proxy'] = 'http://10.100.1.72:3128'
        os.environ['https_proxy'] = 'http://10.100.1.72:3128'
        
        # Initialize Alpha Vantage components
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.ti = TechIndicators(key=api_key, output_format='pandas')
        self.fd = FundamentalData(key=api_key, output_format='pandas')
        self.sp = SectorPerformances(key=api_key, output_format='pandas')
        
        print(f"✅ Alpha Vantage Store initialized")
    
    def load_api_key(self) -> str:
        """Load API key from encrypted config"""
        config_dir = "/opt/cashmachine/config"
        
        try:
            with open(f"{config_dir}/.alphavantage.key", "rb") as f:
                key = f.read()
            with open(f"{config_dir}/alphavantage.enc", "rb") as f:
                encrypted = f.read()
            cipher = Fernet(key)
            config = json.loads(cipher.decrypt(encrypted))
            return config['api_key']
        except:
            # Fallback to direct key
            return "4DCP9RES6PLJBO56"
    
    def get_daily_data(self, symbol: str, outputsize: str = 'full'):
        """Get daily time series data"""
        try:
            data, meta = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
            print(f"📊 Retrieved daily data for {symbol}")
            return data
        except Exception as e:
            print(f"❌ Error getting daily data: {e}")
            return None
    
    def get_intraday_data(self, symbol: str, interval: str = '5min'):
        """Get intraday time series data"""
        try:
            data, meta = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
            print(f"📊 Retrieved {interval} data for {symbol}")
            return data
        except Exception as e:
            print(f"❌ Error getting intraday data: {e}")
            return None
    
    def get_sma(self, symbol: str, interval: str = 'daily', time_period: int = 20):
        """Get Simple Moving Average"""
        try:
            data, meta = self.ti.get_sma(symbol=symbol, interval=interval, time_period=time_period)
            return data
        except Exception as e:
            print(f"❌ Error getting SMA: {e}")
            return None
    
    def get_rsi(self, symbol: str, interval: str = 'daily', time_period: int = 14):
        """Get Relative Strength Index"""
        try:
            data, meta = self.ti.get_rsi(symbol=symbol, interval=interval, time_period=time_period)
            return data
        except Exception as e:
            print(f"❌ Error getting RSI: {e}")
            return None
    
    def get_earnings(self, symbol: str):
        """Get earnings data"""
        try:
            data, meta = self.fd.get_earnings(symbol=symbol)
            return data
        except Exception as e:
            print(f"❌ Error getting earnings: {e}")
            return None
    
    def get_sector_performance(self):
        """Get sector performance data"""
        try:
            data, meta = self.sp.get_sector()
            return data
        except Exception as e:
            print(f"❌ Error getting sector performance: {e}")
            return None


class AlphaVantageData(bt.feeds.PandasData):
    """Alpha Vantage data feed for Backtrader"""
    
    params = (
        ('symbol', 'SPY'),
        ('interval', 'daily'),  # daily, 1min, 5min, 15min, 30min, 60min
        ('api_key', None),
    )
    
    def __init__(self):
        super(AlphaVantageData, self).__init__()
        
        # Create store
        self.store = AlphaVantageStore(api_key=self.params.api_key)
        
        # Get data based on interval
        if self.params.interval == 'daily':
            df = self.store.get_daily_data(self.params.symbol)
        else:
            df = self.store.get_intraday_data(self.params.symbol, self.params.interval)
        
        if df is not None:
            # Prepare dataframe for Backtrader
            df = self.prepare_dataframe(df)
            self.p.dataname = df
    
    def prepare_dataframe(self, df):
        """Prepare Alpha Vantage dataframe for Backtrader"""
        
        # Rename columns to match Backtrader expectations
        column_mapping = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume',
            '6. adjusted close': 'close'  # Use adjusted close if available
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Sort by date (ascending for Backtrader)
        df = df.sort_index()
        
        # Add any missing columns with default values
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        print(f"📊 Prepared {len(df)} bars of data")
        return df


class UltrathinkAlphaVantageStrategy(bt.Strategy):
    """
    ULTRATHINK Strategy using Alpha Vantage data
    Combines technical and fundamental analysis
    """
    
    params = (
        ('symbol', 'SPY'),
        ('use_fundamentals', True),
        ('use_sectors', True),
    )
    
    def __init__(self):
        # Initialize Alpha Vantage store
        self.av_store = AlphaVantageStore()
        
        # Technical indicators from Backtrader
        self.sma20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma50 = bt.indicators.SMA(self.data.close, period=50)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
        # Track additional data
        self.earnings_data = None
        self.sector_data = None
        
        if self.params.use_fundamentals:
            self.earnings_data = self.av_store.get_earnings(self.params.symbol)
        
        if self.params.use_sectors:
            self.sector_data = self.av_store.get_sector_performance()
        
        print("🧠 ULTRATHINK Alpha Vantage Strategy initialized")
    
    def next(self):
        """Trading logic with fundamental and technical analysis"""
        
        # Technical signals
        tech_bullish = self.sma20[0] > self.sma50[0] and self.rsi[0] < 70
        tech_bearish = self.sma20[0] < self.sma50[0] or self.rsi[0] > 80
        
        # Fundamental overlay (if available)
        fundamental_signal = self.check_fundamentals()
        
        # Sector analysis (if available)
        sector_signal = self.check_sectors()
        
        # Combined decision
        if not self.position:
            if tech_bullish and fundamental_signal != 'bearish':
                self.buy()
                print(f"✅ BUY signal at ${self.data.close[0]:.2f}")
        else:
            if tech_bearish or fundamental_signal == 'bearish':
                self.close()
                print(f"📤 CLOSE position at ${self.data.close[0]:.2f}")
    
    def check_fundamentals(self):
        """Analyze fundamental data"""
        if self.earnings_data is not None:
            # Simple earnings trend analysis
            # In real implementation, add more sophisticated analysis
            return 'neutral'
        return 'neutral'
    
    def check_sectors(self):
        """Analyze sector performance"""
        if self.sector_data is not None:
            # Check if tech sector is outperforming
            # In real implementation, add sector rotation logic
            return 'neutral'
        return 'neutral'


def test_alpha_vantage_integration():
    """Test Alpha Vantage integration with Backtrader"""
    
    print("\n" + "="*60)
    print("🧪 ULTRATHINK Alpha Vantage Integration Test")
    print("="*60)
    
    cerebro = bt.Cerebro()
    
    # Add Alpha Vantage data feed
    data = AlphaVantageData(
        symbol='SPY',
        interval='daily'
    )
    
    if data.p.dataname is not None:
        cerebro.adddata(data)
        
        # Add strategy
        cerebro.addstrategy(UltrathinkAlphaVantageStrategy, symbol='SPY')
        
        # Set broker parameters
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.001)
        
        print(f"\n💰 Initial Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        
        # Run backtest
        cerebro.run()
        
        print(f"💰 Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        print("\n✅ Alpha Vantage integration test complete!")
    else:
        print("❌ Failed to load data - check API connection")


if __name__ == "__main__":
    # Test the integration
    test_alpha_vantage_integration()