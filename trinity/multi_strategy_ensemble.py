#!/usr/bin/env python3
"""
MULTI-STRATEGY ENSEMBLE SYSTEM
Runs multiple trading strategies in parallel and combines their signals
Based on genius analysis recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ============================================================================
# BASE STRATEGY
# ============================================================================

class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2

@dataclass
class StrategyVote:
    strategy_name: str
    signal: Signal
    confidence: float
    reasoning: str
    expected_return: float
    risk: float
    timeframe: str  # "seconds", "minutes", "hours", "days"

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.performance_history = []
        self.current_position = None
        self.win_rate = 0.5
        self.sharpe_ratio = 0.0
        
    @abstractmethod
    async def analyze(self, market_data: pd.DataFrame, **kwargs) -> StrategyVote:
        """Analyze market and return vote"""
        pass
    
    def update_performance(self, profit: float):
        """Update strategy performance metrics"""
        self.performance_history.append(profit)
        if len(self.performance_history) > 20:
            wins = [p for p in self.performance_history if p > 0]
            self.win_rate = len(wins) / len(self.performance_history)
            
            returns = pd.Series(self.performance_history)
            if returns.std() > 0:
                self.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

class MomentumStrategy(BaseStrategy):
    """Momentum following strategy"""
    
    def __init__(self):
        super().__init__("Momentum")
        self.lookback = 20
        
    async def analyze(self, market_data: pd.DataFrame, **kwargs) -> StrategyVote:
        """Detect and follow momentum"""
        if len(market_data) < self.lookback:
            return StrategyVote(
                self.name, Signal.NEUTRAL, 0.0,
                "Insufficient data", 0.0, 0.0, "minutes"
            )
        
        # Calculate momentum indicators
        returns = market_data['close'].pct_change()
        momentum = returns.rolling(self.lookback).mean()
        current_momentum = momentum.iloc[-1]
        
        # Calculate RSI for confirmation
        rsi = self.calculate_rsi(market_data['close'])
        
        # Determine signal
        if current_momentum > 0.001 and rsi > 60:
            signal = Signal.BUY if rsi < 70 else Signal.STRONG_BUY
            confidence = min(abs(current_momentum) * 100, 0.9)
            reasoning = f"Positive momentum {current_momentum:.4f}, RSI {rsi:.1f}"
        elif current_momentum < -0.001 and rsi < 40:
            signal = Signal.SELL if rsi > 30 else Signal.STRONG_SELL
            confidence = min(abs(current_momentum) * 100, 0.9)
            reasoning = f"Negative momentum {current_momentum:.4f}, RSI {rsi:.1f}"
        else:
            signal = Signal.NEUTRAL
            confidence = 0.3
            reasoning = "No clear momentum"
        
        expected_return = current_momentum * self.lookback
        risk = returns.std() * np.sqrt(self.lookback)
        
        return StrategyVote(
            self.name, signal, confidence, reasoning,
            expected_return, risk, "minutes"
        )
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy"""
    
    def __init__(self):
        super().__init__("MeanReversion")
        self.lookback = 20
        self.z_threshold = 2.0
        
    async def analyze(self, market_data: pd.DataFrame, **kwargs) -> StrategyVote:
        """Detect mean reversion opportunities"""
        if len(market_data) < self.lookback:
            return StrategyVote(
                self.name, Signal.NEUTRAL, 0.0,
                "Insufficient data", 0.0, 0.0, "hours"
            )
        
        # Calculate Bollinger Bands
        close = market_data['close']
        sma = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        
        current_price = close.iloc[-1]
        current_sma = sma.iloc[-1]
        current_std = std.iloc[-1]
        
        # Calculate z-score
        z_score = (current_price - current_sma) / current_std if current_std > 0 else 0
        
        # Determine signal
        if z_score < -self.z_threshold:
            signal = Signal.STRONG_BUY
            confidence = min(abs(z_score) / 3, 0.9)
            reasoning = f"Oversold: z-score {z_score:.2f}"
        elif z_score < -1:
            signal = Signal.BUY
            confidence = min(abs(z_score) / 2, 0.7)
            reasoning = f"Below mean: z-score {z_score:.2f}"
        elif z_score > self.z_threshold:
            signal = Signal.STRONG_SELL
            confidence = min(abs(z_score) / 3, 0.9)
            reasoning = f"Overbought: z-score {z_score:.2f}"
        elif z_score > 1:
            signal = Signal.SELL
            confidence = min(abs(z_score) / 2, 0.7)
            reasoning = f"Above mean: z-score {z_score:.2f}"
        else:
            signal = Signal.NEUTRAL
            confidence = 0.3
            reasoning = f"Near mean: z-score {z_score:.2f}"
        
        # Expected return is reversion to mean
        expected_return = -(current_price - current_sma) / current_price
        risk = current_std / current_price
        
        return StrategyVote(
            self.name, signal, confidence, reasoning,
            expected_return, risk, "hours"
        )

class MarketMakingStrategy(BaseStrategy):
    """Market making strategy - profit from bid/ask spread"""
    
    def __init__(self):
        super().__init__("MarketMaking")
        
    async def analyze(self, market_data: pd.DataFrame, **kwargs) -> StrategyVote:
        """Analyze spread and liquidity"""
        # Get order book data if available
        spread = kwargs.get('spread', 0)
        bid_volume = kwargs.get('bid_volume', 0)
        ask_volume = kwargs.get('ask_volume', 0)
        volatility = market_data['close'].pct_change().std() if len(market_data) > 2 else 0
        
        # Market making is profitable when spread > volatility
        spread_bps = spread * 10000  # Convert to basis points
        vol_bps = volatility * 10000
        
        if spread_bps > vol_bps * 2 and bid_volume > 0 and ask_volume > 0:
            # Wide spread, good for market making
            signal = Signal.NEUTRAL  # We provide liquidity, not take direction
            confidence = min(spread_bps / (vol_bps + 1), 0.8)
            reasoning = f"Wide spread {spread_bps:.1f} bps vs vol {vol_bps:.1f} bps"
            expected_return = spread / 2  # Capture half spread on average
            risk = volatility
        else:
            signal = Signal.NEUTRAL
            confidence = 0.2
            reasoning = "Spread too tight for market making"
            expected_return = 0
            risk = volatility
        
        return StrategyVote(
            self.name, signal, confidence, reasoning,
            expected_return, risk, "seconds"
        )

class StatisticalArbitrageStrategy(BaseStrategy):
    """Statistical arbitrage - trade correlated pairs"""
    
    def __init__(self):
        super().__init__("StatArb")
        self.lookback = 30
        
    async def analyze(self, market_data: pd.DataFrame, **kwargs) -> StrategyVote:
        """Find statistical arbitrage opportunities"""
        # Need data for multiple symbols
        symbol2_data = kwargs.get('symbol2_data')
        
        if symbol2_data is None or len(market_data) < self.lookback:
            return StrategyVote(
                self.name, Signal.NEUTRAL, 0.0,
                "Insufficient pair data", 0.0, 0.0, "hours"
            )
        
        # Calculate correlation
        returns1 = market_data['close'].pct_change()
        returns2 = symbol2_data['close'].pct_change()
        correlation = returns1.corr(returns2)
        
        if abs(correlation) < 0.7:
            return StrategyVote(
                self.name, Signal.NEUTRAL, 0.2,
                f"Low correlation {correlation:.2f}", 0.0, 0.0, "hours"
            )
        
        # Calculate spread between pairs
        ratio = market_data['close'] / symbol2_data['close']
        ratio_mean = ratio.rolling(self.lookback).mean()
        ratio_std = ratio.rolling(self.lookback).std()
        
        current_ratio = ratio.iloc[-1]
        z_score = (current_ratio - ratio_mean.iloc[-1]) / ratio_std.iloc[-1]
        
        # Generate signals based on mean reversion of spread
        if z_score < -2:
            signal = Signal.BUY  # Buy symbol1, sell symbol2
            confidence = min(abs(z_score) / 3, 0.8)
            reasoning = f"Pair divergence: z-score {z_score:.2f}"
        elif z_score > 2:
            signal = Signal.SELL  # Sell symbol1, buy symbol2
            confidence = min(abs(z_score) / 3, 0.8)
            reasoning = f"Pair divergence: z-score {z_score:.2f}"
        else:
            signal = Signal.NEUTRAL
            confidence = 0.3
            reasoning = "Pairs in equilibrium"
        
        expected_return = -z_score * ratio_std.iloc[-1] / current_ratio
        risk = ratio_std.iloc[-1] / current_ratio
        
        return StrategyVote(
            self.name, signal, confidence, reasoning,
            expected_return, risk, "hours"
        )

class VolumeProfileStrategy(BaseStrategy):
    """Trade based on volume profile and accumulation/distribution"""
    
    def __init__(self):
        super().__init__("VolumeProfile")
        
    async def analyze(self, market_data: pd.DataFrame, **kwargs) -> StrategyVote:
        """Analyze volume patterns"""
        if len(market_data) < 20:
            return StrategyVote(
                self.name, Signal.NEUTRAL, 0.0,
                "Insufficient data", 0.0, 0.0, "minutes"
            )
        
        # Calculate On-Balance Volume (OBV)
        obv = self.calculate_obv(market_data)
        obv_ma = obv.rolling(10).mean()
        
        # Calculate Volume-Weighted Average Price (VWAP)
        vwap = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()
        
        current_price = market_data['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        current_obv = obv.iloc[-1]
        obv_trend = (current_obv - obv_ma.iloc[-1]) / abs(obv_ma.iloc[-1]) if obv_ma.iloc[-1] != 0 else 0
        
        # Signals based on volume analysis
        if current_price > current_vwap and obv_trend > 0.05:
            signal = Signal.BUY
            confidence = min(obv_trend * 10, 0.8)
            reasoning = f"Accumulation: OBV trend {obv_trend:.2%}, price above VWAP"
        elif current_price < current_vwap and obv_trend < -0.05:
            signal = Signal.SELL
            confidence = min(abs(obv_trend) * 10, 0.8)
            reasoning = f"Distribution: OBV trend {obv_trend:.2%}, price below VWAP"
        else:
            signal = Signal.NEUTRAL
            confidence = 0.3
            reasoning = "No clear volume signal"
        
        expected_return = obv_trend * 0.1
        risk = market_data['close'].pct_change().std()
        
        return StrategyVote(
            self.name, signal, confidence, reasoning,
            expected_return, risk, "minutes"
        )
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

class MachineLearningStrategy(BaseStrategy):
    """ML-based prediction strategy"""
    
    def __init__(self):
        super().__init__("MachineLearning")
        self.model_confidence = 0.65
        
    async def analyze(self, market_data: pd.DataFrame, **kwargs) -> StrategyVote:
        """Use ML model predictions"""
        # Get predictions from ULTRATHINK components
        asi_signal = kwargs.get('asi_signal', 'hold')
        hrm_signal = kwargs.get('hrm_signal', 'hold')
        mcts_signal = kwargs.get('mcts_signal', 'hold')
        
        # Convert to numerical scores
        signal_map = {'buy': 1, 'sell': -1, 'hold': 0}
        asi_score = signal_map.get(asi_signal.lower(), 0)
        hrm_score = signal_map.get(hrm_signal.lower(), 0)
        mcts_score = signal_map.get(mcts_signal.lower(), 0)
        
        # Weighted ensemble
        ml_score = (asi_score * 0.4 + hrm_score * 0.3 + mcts_score * 0.3)
        
        if ml_score > 0.5:
            signal = Signal.BUY if ml_score < 0.8 else Signal.STRONG_BUY
            confidence = min(abs(ml_score), 0.85)
            reasoning = f"ML ensemble: ASI={asi_signal}, HRM={hrm_signal}, MCTS={mcts_signal}"
        elif ml_score < -0.5:
            signal = Signal.SELL if ml_score > -0.8 else Signal.STRONG_SELL
            confidence = min(abs(ml_score), 0.85)
            reasoning = f"ML ensemble: ASI={asi_signal}, HRM={hrm_signal}, MCTS={mcts_signal}"
        else:
            signal = Signal.NEUTRAL
            confidence = 0.4
            reasoning = "ML models disagree"
        
        expected_return = ml_score * 0.01
        risk = 0.02  # Estimated based on historical ML performance
        
        return StrategyVote(
            self.name, signal, confidence, reasoning,
            expected_return, risk, "minutes"
        )

# ============================================================================
# ENSEMBLE ORCHESTRATOR
# ============================================================================

class StrategyEnsemble:
    """Orchestrates multiple strategies and combines their votes"""
    
    def __init__(self):
        self.strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            MarketMakingStrategy(),
            StatisticalArbitrageStrategy(),
            VolumeProfileStrategy(),
            MachineLearningStrategy()
        ]
        self.strategy_weights = {s.name: 1.0 for s in self.strategies}
        self.performance_window = 100
        
    async def get_ensemble_decision(
        self,
        market_data: pd.DataFrame,
        **kwargs
    ) -> Dict:
        """Get votes from all strategies and combine"""
        
        # Collect votes from all strategies in parallel
        votes = await asyncio.gather(*[
            strategy.analyze(market_data, **kwargs)
            for strategy in self.strategies
        ])
        
        # Update weights based on recent performance
        self.update_strategy_weights()
        
        # Combine votes
        final_decision = self.combine_votes(votes)
        
        return final_decision
    
    def combine_votes(self, votes: List[StrategyVote]) -> Dict:
        """Combine strategy votes into final decision"""
        
        # Weight votes by strategy performance and confidence
        weighted_signal = 0
        total_weight = 0
        
        vote_breakdown = []
        
        for vote in votes:
            weight = self.strategy_weights[vote.strategy_name] * vote.confidence
            weighted_signal += vote.signal.value * weight
            total_weight += weight
            
            vote_breakdown.append({
                'strategy': vote.strategy_name,
                'signal': vote.signal.name,
                'confidence': vote.confidence,
                'weight': weight,
                'reasoning': vote.reasoning
            })
        
        # Calculate final signal
        if total_weight > 0:
            final_score = weighted_signal / total_weight
        else:
            final_score = 0
        
        # Map to final decision
        if final_score > 1:
            final_signal = "STRONG_BUY"
            action = "buy"
        elif final_score > 0.3:
            final_signal = "BUY"
            action = "buy"
        elif final_score < -1:
            final_signal = "STRONG_SELL"
            action = "sell"
        elif final_score < -0.3:
            final_signal = "SELL"
            action = "sell"
        else:
            final_signal = "NEUTRAL"
            action = "hold"
        
        # Calculate ensemble confidence
        # Higher when strategies agree
        signal_variance = np.var([v.signal.value for v in votes])
        agreement_score = 1 / (1 + signal_variance)
        
        avg_confidence = np.mean([v.confidence for v in votes])
        ensemble_confidence = (agreement_score + avg_confidence) / 2
        
        # Calculate expected metrics
        expected_return = np.mean([v.expected_return for v in votes])
        risk = np.mean([v.risk for v in votes])
        
        return {
            'action': action,
            'signal': final_signal,
            'score': final_score,
            'confidence': ensemble_confidence,
            'expected_return': expected_return,
            'risk': risk,
            'vote_breakdown': vote_breakdown,
            'agreement_score': agreement_score,
            'strategies_bullish': sum(1 for v in votes if v.signal.value > 0),
            'strategies_bearish': sum(1 for v in votes if v.signal.value < 0),
            'strategies_neutral': sum(1 for v in votes if v.signal.value == 0)
        }
    
    def update_strategy_weights(self):
        """Update strategy weights based on performance"""
        for strategy in self.strategies:
            if len(strategy.performance_history) >= 20:
                # Increase weight for profitable strategies
                if strategy.win_rate > 0.55:
                    self.strategy_weights[strategy.name] = min(
                        self.strategy_weights[strategy.name] * 1.05, 2.0
                    )
                elif strategy.win_rate < 0.45:
                    self.strategy_weights[strategy.name] = max(
                        self.strategy_weights[strategy.name] * 0.95, 0.5
                    )
                
                # Bonus for high Sharpe ratio
                if strategy.sharpe_ratio > 1.5:
                    self.strategy_weights[strategy.name] *= 1.1
        
        # Normalize weights
        total = sum(self.strategy_weights.values())
        if total > 0:
            for name in self.strategy_weights:
                self.strategy_weights[name] /= total
    
    def update_performance(self, strategy_name: str, profit: float):
        """Update individual strategy performance"""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.update_performance(profit)
                break

# This ensemble system ensures we're not relying on a single strategy!