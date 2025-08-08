#!/usr/bin/env python3
"""
MICROSTRUCTURE GENIUS - Order Book Analysis Engine
This is what Renaissance Technologies actually does!
Implements:
- Level 2 order book imbalance detection
- Kyle's Lambda price impact prediction
- Volume-synchronized probability of informed trading (VPIN)
- Hidden liquidity detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class OrderBookLevel:
    """Single level in order book"""
    price: float
    size: float
    orders: int
    
@dataclass
class OrderBookSnapshot:
    """Full order book snapshot"""
    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points"""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0

class MicrostructureAnalyzer:
    """Advanced order book microstructure analysis"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.book_history = deque(maxlen=window_size)
        self.trade_history = deque(maxlen=window_size * 10)
        self.kyle_lambda = 0.0
        self.vpin = 0.0
        
    def calculate_order_flow_imbalance(self, book: OrderBookSnapshot, depth: int = 5) -> float:
        """
        Calculate order flow imbalance (OFI)
        Positive = Buy pressure, Negative = Sell pressure
        """
        bid_volume = sum(level.size for level in book.bids[:depth])
        ask_volume = sum(level.size for level in book.asks[:depth])
        
        if bid_volume + ask_volume == 0:
            return 0
        
        # Weighted by distance from mid
        bid_weighted = 0
        ask_weighted = 0
        mid = book.mid_price
        
        for i, level in enumerate(book.bids[:depth]):
            weight = 1 / (1 + i)  # Closer levels have more weight
            distance_weight = 1 / (1 + abs(mid - level.price) / mid)
            bid_weighted += level.size * weight * distance_weight
        
        for i, level in enumerate(book.asks[:depth]):
            weight = 1 / (1 + i)
            distance_weight = 1 / (1 + abs(level.price - mid) / mid)
            ask_weighted += level.size * weight * distance_weight
        
        # Normalize to [-1, 1]
        ofi = (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted)
        
        return ofi
    
    def calculate_kyle_lambda(self, trades: List[Dict]) -> float:
        """
        Calculate Kyle's Lambda - price impact per unit volume
        Higher lambda = less liquid, more impact
        """
        if len(trades) < 10:
            return self.kyle_lambda
        
        # Extract price changes and volumes
        price_changes = []
        volumes = []
        
        for i in range(1, len(trades)):
            price_change = abs(trades[i]['price'] - trades[i-1]['price'])
            volume = trades[i]['volume']
            
            price_changes.append(price_change)
            volumes.append(volume)
        
        if not volumes or not price_changes:
            return self.kyle_lambda
        
        # Linear regression: price_change = lambda * volume
        volumes_array = np.array(volumes)
        price_changes_array = np.array(price_changes)
        
        # Add small epsilon to avoid division by zero
        if np.std(volumes_array) > 0:
            lambda_estimate = np.cov(price_changes_array, volumes_array)[0, 1] / np.var(volumes_array)
            self.kyle_lambda = max(0, lambda_estimate)  # Lambda should be positive
        
        return self.kyle_lambda
    
    def calculate_vpin(self, trades: List[Dict], bucket_size: float = 1000) -> float:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN)
        Measures toxicity of order flow
        """
        if len(trades) < 20:
            return self.vpin
        
        # Create volume buckets
        buckets = []
        current_bucket = {'buy_volume': 0, 'sell_volume': 0, 'total_volume': 0}
        
        for trade in trades:
            # Classify as buy or sell based on tick rule
            if 'side' in trade:
                if trade['side'] == 'buy':
                    current_bucket['buy_volume'] += trade['volume']
                else:
                    current_bucket['sell_volume'] += trade['volume']
            else:
                # Use tick rule if side not available
                if len(buckets) > 0:
                    prev_price = buckets[-1].get('last_price', trade['price'])
                    if trade['price'] > prev_price:
                        current_bucket['buy_volume'] += trade['volume']
                    else:
                        current_bucket['sell_volume'] += trade['volume']
            
            current_bucket['total_volume'] += trade['volume']
            current_bucket['last_price'] = trade['price']
            
            # Check if bucket is full
            if current_bucket['total_volume'] >= bucket_size:
                buckets.append(current_bucket.copy())
                current_bucket = {'buy_volume': 0, 'sell_volume': 0, 'total_volume': 0}
        
        if len(buckets) < 5:
            return self.vpin
        
        # Calculate VPIN over last N buckets
        recent_buckets = buckets[-min(50, len(buckets)):]
        vpins = []
        
        for bucket in recent_buckets:
            if bucket['total_volume'] > 0:
                imbalance = abs(bucket['buy_volume'] - bucket['sell_volume'])
                vpin_bucket = imbalance / bucket['total_volume']
                vpins.append(vpin_bucket)
        
        if vpins:
            self.vpin = np.mean(vpins)
        
        return self.vpin
    
    def detect_hidden_liquidity(self, book: OrderBookSnapshot) -> Dict:
        """
        Detect hidden/iceberg orders by analyzing order book dynamics
        """
        hidden_indicators = {
            'likely_hidden_bid': False,
            'likely_hidden_ask': False,
            'hidden_size_estimate': 0
        }
        
        if len(self.book_history) < 10:
            self.book_history.append(book)
            return hidden_indicators
        
        # Check for persistent refilling at same price levels
        current_best_bid = book.bids[0].price if book.bids else 0
        current_best_ask = book.asks[0].price if book.asks else 0
        
        bid_refill_count = 0
        ask_refill_count = 0
        
        for hist_book in list(self.book_history)[-10:]:
            # Check if best bid/ask prices are stable but sizes keep refilling
            if hist_book.bids and abs(hist_book.bids[0].price - current_best_bid) < 0.01:
                if hist_book.bids[0].size > book.bids[0].size * 0.5:
                    bid_refill_count += 1
            
            if hist_book.asks and abs(hist_book.asks[0].price - current_best_ask) < 0.01:
                if hist_book.asks[0].size > book.asks[0].size * 0.5:
                    ask_refill_count += 1
        
        # If price level keeps refilling, likely hidden order
        if bid_refill_count > 7:
            hidden_indicators['likely_hidden_bid'] = True
            hidden_indicators['hidden_size_estimate'] += book.bids[0].size * 5
        
        if ask_refill_count > 7:
            hidden_indicators['likely_hidden_ask'] = True
            hidden_indicators['hidden_size_estimate'] += book.asks[0].size * 5
        
        self.book_history.append(book)
        return hidden_indicators
    
    def calculate_microprice(self, book: OrderBookSnapshot) -> float:
        """
        Calculate microprice - better estimate than mid price
        Weighted by order book imbalance
        """
        if not book.bids or not book.asks:
            return 0
        
        bid_price = book.bids[0].price
        ask_price = book.asks[0].price
        bid_size = book.bids[0].size
        ask_size = book.asks[0].size
        
        # Microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
        if bid_size + ask_size > 0:
            microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
        else:
            microprice = (bid_price + ask_price) / 2
        
        return microprice
    
    def predict_price_movement(self, book: OrderBookSnapshot, trades: List[Dict]) -> Dict:
        """
        Predict short-term price movement using microstructure signals
        """
        # Calculate all microstructure indicators
        ofi = self.calculate_order_flow_imbalance(book)
        kyle_lambda = self.calculate_kyle_lambda(trades)
        vpin = self.calculate_vpin(trades)
        hidden = self.detect_hidden_liquidity(book)
        microprice = self.calculate_microprice(book)
        
        # Combine signals
        signals = []
        
        # Order flow imbalance signal
        if ofi > 0.3:
            signals.append(('bullish', abs(ofi)))
        elif ofi < -0.3:
            signals.append(('bearish', abs(ofi)))
        
        # VPIN toxicity signal
        if vpin > 0.6:
            signals.append(('volatile', vpin))
        elif vpin < 0.3:
            signals.append(('calm', 1 - vpin))
        
        # Hidden liquidity signal
        if hidden['likely_hidden_bid']:
            signals.append(('support', 0.7))
        if hidden['likely_hidden_ask']:
            signals.append(('resistance', 0.7))
        
        # Kyle's lambda signal (inverse - low lambda is good)
        if kyle_lambda < 0.001:
            signals.append(('liquid', 0.8))
        elif kyle_lambda > 0.01:
            signals.append(('illiquid', 0.8))
        
        # Aggregate prediction
        bullish_score = sum(conf for sig, conf in signals if sig in ['bullish', 'support', 'liquid'])
        bearish_score = sum(conf for sig, conf in signals if sig in ['bearish', 'resistance', 'illiquid'])
        
        if bullish_score > bearish_score * 1.2:
            prediction = 'UP'
            confidence = min(bullish_score / (bullish_score + bearish_score), 0.9)
        elif bearish_score > bullish_score * 1.2:
            prediction = 'DOWN'
            confidence = min(bearish_score / (bullish_score + bearish_score), 0.9)
        else:
            prediction = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'ofi': ofi,
            'kyle_lambda': kyle_lambda,
            'vpin': vpin,
            'microprice': microprice,
            'hidden_bid': hidden['likely_hidden_bid'],
            'hidden_ask': hidden['likely_hidden_ask'],
            'signals': signals
        }

class HighFrequencyMicrostructure:
    """
    Ultra-fast microstructure for HFT-style execution
    This is the secret sauce!
    """
    
    def __init__(self):
        self.analyzer = MicrostructureAnalyzer()
        self.execution_alpha = 0
        
    async def calculate_execution_alpha(
        self,
        book: OrderBookSnapshot,
        target_size: float
    ) -> Dict:
        """
        Calculate expected alpha from smart execution
        """
        # Calculate immediate cost of market order
        market_cost = self.calculate_market_impact(book, target_size)
        
        # Calculate expected cost with smart execution
        smart_cost = self.calculate_smart_execution_cost(book, target_size)
        
        # Alpha is the difference
        execution_alpha = market_cost - smart_cost
        
        # Calculate optimal execution strategy
        if book.spread_bps < 5:
            strategy = "AGGRESSIVE"  # Take liquidity
        elif book.spread_bps > 20:
            strategy = "PASSIVE"  # Provide liquidity
        else:
            strategy = "MIXED"  # Combination
        
        return {
            'execution_alpha_bps': execution_alpha * 10000,
            'market_impact_bps': market_cost * 10000,
            'smart_cost_bps': smart_cost * 10000,
            'optimal_strategy': strategy,
            'recommended_slice_size': self.calculate_optimal_slice(target_size, book)
        }
    
    def calculate_market_impact(self, book: OrderBookSnapshot, size: float) -> float:
        """Calculate expected market impact of order"""
        cumulative_size = 0
        weighted_price = 0
        
        # Walk through book to calculate impact
        for level in book.asks:  # Assuming buy order
            level_size = min(level.size, size - cumulative_size)
            weighted_price += level.price * level_size
            cumulative_size += level_size
            
            if cumulative_size >= size:
                break
        
        if cumulative_size > 0:
            avg_price = weighted_price / cumulative_size
            impact = (avg_price - book.mid_price) / book.mid_price
            return impact
        
        return 0
    
    def calculate_smart_execution_cost(self, book: OrderBookSnapshot, size: float) -> float:
        """Calculate cost with smart execution"""
        # Assume we can get 30% at mid, 50% at quarter spread, 20% at half spread
        mid = book.mid_price
        quarter_spread = book.spread * 0.25
        half_spread = book.spread * 0.5
        
        cost = (
            0.3 * 0 +  # 30% at mid (no cost)
            0.5 * (quarter_spread / mid) +  # 50% at quarter spread
            0.2 * (half_spread / mid)  # 20% at half spread
        )
        
        return cost
    
    def calculate_optimal_slice(self, total_size: float, book: OrderBookSnapshot) -> float:
        """Calculate optimal slice size for iceberg orders"""
        # Base it on average level size in book
        avg_level_size = np.mean([level.size for level in book.bids[:5] + book.asks[:5]])
        
        # Optimal slice is 20-50% of average level
        optimal = avg_level_size * 0.35
        
        # But at least 1% and at most 10% of total
        optimal = max(total_size * 0.01, min(optimal, total_size * 0.1))
        
        return optimal

# ============================================================================
# REAL-TIME CONNECTOR
# ============================================================================

class RealTimeBookConnector:
    """Connect to real-time order book data"""
    
    def __init__(self, api_client):
        self.api = api_client
        self.analyzer = HighFrequencyMicrostructure()
        
    async def stream_and_analyze(self, symbol: str):
        """Stream order book and analyze in real-time"""
        logger.info(f"Starting microstructure analysis for {symbol}")
        
        while True:
            try:
                # Get Level 2 data (this would connect to real feed)
                book_data = await self.get_level2_snapshot(symbol)
                
                # Convert to our format
                book = self.parse_book_data(book_data)
                
                # Get recent trades
                trades = await self.get_recent_trades(symbol)
                
                # Analyze microstructure
                prediction = self.analyzer.analyzer.predict_price_movement(book, trades)
                
                # Calculate execution alpha
                execution = await self.analyzer.calculate_execution_alpha(book, 100)
                
                # Log insights
                if prediction['confidence'] > 0.7:
                    logger.info(f"🎯 HIGH CONFIDENCE: {prediction['prediction']} "
                              f"(conf: {prediction['confidence']:.2f}, "
                              f"OFI: {prediction['ofi']:.2f}, "
                              f"VPIN: {prediction['vpin']:.2f})")
                
                if execution['execution_alpha_bps'] > 5:
                    logger.info(f"💰 EXECUTION ALPHA: {execution['execution_alpha_bps']:.1f} bps "
                              f"using {execution['optimal_strategy']}")
                
                await asyncio.sleep(0.1)  # 100ms updates
                
            except Exception as e:
                logger.error(f"Microstructure error: {e}")
                await asyncio.sleep(1)
    
    async def get_level2_snapshot(self, symbol: str) -> Dict:
        """Get Level 2 order book snapshot"""
        # This would connect to real Level 2 feed
        # For now, return mock data
        return {
            'bids': [[100.00, 1000], [99.99, 2000], [99.98, 1500]],
            'asks': [[100.01, 1000], [100.02, 2000], [100.03, 1500]]
        }
    
    async def get_recent_trades(self, symbol: str) -> List[Dict]:
        """Get recent trades"""
        # This would connect to real trade feed
        # For now, return mock data
        return [
            {'price': 100.00, 'volume': 100, 'side': 'buy'},
            {'price': 100.01, 'volume': 200, 'side': 'sell'},
            {'price': 100.00, 'volume': 150, 'side': 'buy'}
        ]
    
    def parse_book_data(self, data: Dict) -> OrderBookSnapshot:
        """Parse raw book data into our format"""
        bids = [OrderBookLevel(price=p, size=s, orders=1) for p, s in data['bids']]
        asks = [OrderBookLevel(price=p, size=s, orders=1) for p, s in data['asks']]
        
        return OrderBookSnapshot(
            timestamp=asyncio.get_event_loop().time(),
            bids=bids,
            asks=asks
        )

# This is the module that would give ULTRATHINK the edge!