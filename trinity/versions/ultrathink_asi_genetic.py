#!/usr/bin/env python3
"""
ULTRATHINK ASI-Arch - Autonomous Strategy Intelligence
Genetic Algorithm for Self-Improving Trading Strategies
Real evolution, real fitness, real adaptation
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ASI_GENETIC')

@dataclass
class TradingStrategy:
    """Individual trading strategy genome"""
    # Technical indicator weights
    rsi_weight: float
    macd_weight: float
    bollinger_weight: float
    volume_weight: float
    momentum_weight: float
    
    # Thresholds
    buy_threshold: float
    sell_threshold: float
    stop_loss: float
    take_profit: float
    
    # Risk parameters
    position_size: float
    max_positions: int
    
    # Time parameters
    holding_period: int  # bars
    cooldown_period: int  # bars between trades
    
    # Meta parameters
    fitness: float = 0.0
    generation: int = 0
    trades_executed: int = 0
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate strategy parameters"""
        if random.random() < mutation_rate:
            self.rsi_weight *= np.random.uniform(0.8, 1.2)
        if random.random() < mutation_rate:
            self.macd_weight *= np.random.uniform(0.8, 1.2)
        if random.random() < mutation_rate:
            self.bollinger_weight *= np.random.uniform(0.8, 1.2)
        if random.random() < mutation_rate:
            self.volume_weight *= np.random.uniform(0.8, 1.2)
        if random.random() < mutation_rate:
            self.momentum_weight *= np.random.uniform(0.8, 1.2)
        
        if random.random() < mutation_rate:
            self.buy_threshold += np.random.uniform(-0.1, 0.1)
        if random.random() < mutation_rate:
            self.sell_threshold += np.random.uniform(-0.1, 0.1)
        if random.random() < mutation_rate:
            self.stop_loss *= np.random.uniform(0.9, 1.1)
        if random.random() < mutation_rate:
            self.take_profit *= np.random.uniform(0.9, 1.1)
        
        # Ensure valid ranges
        self.rsi_weight = max(0.0, min(2.0, self.rsi_weight))
        self.macd_weight = max(0.0, min(2.0, self.macd_weight))
        self.bollinger_weight = max(0.0, min(2.0, self.bollinger_weight))
        self.volume_weight = max(0.0, min(2.0, self.volume_weight))
        self.momentum_weight = max(0.0, min(2.0, self.momentum_weight))
        
        self.buy_threshold = max(-1.0, min(1.0, self.buy_threshold))
        self.sell_threshold = max(-1.0, min(1.0, self.sell_threshold))
        self.stop_loss = max(0.01, min(0.1, self.stop_loss))
        self.take_profit = max(0.02, min(0.2, self.take_profit))
        
        self.position_size = max(0.01, min(0.1, self.position_size))
        self.max_positions = max(1, min(10, self.max_positions))
        self.holding_period = max(1, min(100, self.holding_period))
        self.cooldown_period = max(0, min(10, self.cooldown_period))

class ASIGeneticEvolution:
    """Genetic algorithm for evolving trading strategies"""
    
    def __init__(self, population_size: int = 100):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_strategy = None
        self.hall_of_fame = []  # Best strategies ever
        
        # Evolution parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elitism_count = 5  # Keep top 5 strategies
        self.tournament_size = 5
        
        # Performance tracking
        self.fitness_history = []
        self.diversity_history = []
        
        # Initialize population
        self._initialize_population()
        
        logger.info(f"🧬 ASI Genetic Evolution initialized")
        logger.info(f"   Population size: {self.population_size}")
        logger.info(f"   Mutation rate: {self.mutation_rate}")
    
    def _initialize_population(self):
        """Create initial random population"""
        for i in range(self.population_size):
            strategy = TradingStrategy(
                rsi_weight=np.random.uniform(0.0, 1.5),
                macd_weight=np.random.uniform(0.0, 1.5),
                bollinger_weight=np.random.uniform(0.0, 1.5),
                volume_weight=np.random.uniform(0.0, 1.5),
                momentum_weight=np.random.uniform(0.0, 1.5),
                buy_threshold=np.random.uniform(-0.5, 0.5),
                sell_threshold=np.random.uniform(-0.5, 0.5),
                stop_loss=np.random.uniform(0.02, 0.05),
                take_profit=np.random.uniform(0.04, 0.1),
                position_size=np.random.uniform(0.02, 0.05),
                max_positions=np.random.randint(1, 6),
                holding_period=np.random.randint(5, 50),
                cooldown_period=np.random.randint(0, 5),
                generation=0
            )
            self.population.append(strategy)
    
    def calculate_fitness(self, strategy: TradingStrategy, market_data: Dict) -> float:
        """Calculate fitness based on backtesting results"""
        # Simulate trading with this strategy
        returns = []
        trades = 0
        wins = 0
        losses = 0
        
        prices = market_data.get('prices', [])
        if len(prices) < 50:
            return 0.0
        
        position = None
        cooldown = 0
        
        for i in range(50, len(prices)):
            # Calculate indicators
            price_slice = prices[i-50:i]
            current_price = prices[i]
            
            # Simple RSI
            gains = [max(0, price_slice[j] - price_slice[j-1]) for j in range(1, len(price_slice))]
            losses = [max(0, price_slice[j-1] - price_slice[j]) for j in range(1, len(price_slice))]
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
            rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10)))) if avg_loss > 0 else 50
            
            # Simple momentum
            momentum = (current_price - price_slice[-20]) / price_slice[-20] if len(price_slice) >= 20 else 0
            
            # Volume (simulated)
            volume_signal = np.random.uniform(-1, 1)
            
            # MACD (simplified)
            ema_12 = np.mean(price_slice[-12:]) if len(price_slice) >= 12 else current_price
            ema_26 = np.mean(price_slice[-26:]) if len(price_slice) >= 26 else current_price
            macd = (ema_12 - ema_26) / current_price
            
            # Bollinger (simplified)
            sma_20 = np.mean(price_slice[-20:]) if len(price_slice) >= 20 else current_price
            std_20 = np.std(price_slice[-20:]) if len(price_slice) >= 20 else 1
            bollinger = (current_price - sma_20) / (std_20 + 1e-10)
            
            # Calculate combined signal
            signal = (
                strategy.rsi_weight * (50 - rsi) / 50 +
                strategy.macd_weight * macd * 10 +
                strategy.bollinger_weight * bollinger +
                strategy.volume_weight * volume_signal +
                strategy.momentum_weight * momentum * 10
            )
            
            # Normalize signal
            total_weight = (strategy.rsi_weight + strategy.macd_weight + 
                          strategy.bollinger_weight + strategy.volume_weight + 
                          strategy.momentum_weight)
            if total_weight > 0:
                signal = signal / total_weight
            
            # Trading logic
            if cooldown > 0:
                cooldown -= 1
            
            if position is None and cooldown == 0:
                if signal > strategy.buy_threshold:
                    # Open long position
                    position = {
                        'type': 'long',
                        'entry': current_price,
                        'size': strategy.position_size,
                        'bars': 0
                    }
                    trades += 1
                elif signal < strategy.sell_threshold:
                    # Open short position
                    position = {
                        'type': 'short',
                        'entry': current_price,
                        'size': strategy.position_size,
                        'bars': 0
                    }
                    trades += 1
            
            elif position is not None:
                position['bars'] += 1
                pnl_pct = 0
                
                if position['type'] == 'long':
                    pnl_pct = (current_price - position['entry']) / position['entry']
                    
                    # Check exit conditions
                    if (pnl_pct <= -strategy.stop_loss or 
                        pnl_pct >= strategy.take_profit or
                        position['bars'] >= strategy.holding_period):
                        
                        returns.append(pnl_pct * position['size'])
                        if pnl_pct > 0:
                            wins += 1
                        else:
                            losses += 1
                        position = None
                        cooldown = strategy.cooldown_period
                
                else:  # short position
                    pnl_pct = (position['entry'] - current_price) / position['entry']
                    
                    if (pnl_pct <= -strategy.stop_loss or 
                        pnl_pct >= strategy.take_profit or
                        position['bars'] >= strategy.holding_period):
                        
                        returns.append(pnl_pct * position['size'])
                        if pnl_pct > 0:
                            wins += 1
                        else:
                            losses += 1
                        position = None
                        cooldown = strategy.cooldown_period
        
        # Calculate fitness metrics
        if len(returns) == 0:
            return 0.0
        
        total_return = sum(returns)
        avg_return = np.mean(returns)
        
        # Sharpe ratio (simplified)
        if len(returns) > 1:
            sharpe = avg_return / (np.std(returns) + 1e-10)
        else:
            sharpe = 0
        
        # Win rate
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Fitness combines multiple metrics
        fitness = (
            total_return * 100 +  # Total return weight
            sharpe * 10 +  # Sharpe ratio weight
            win_rate * 20 +  # Win rate weight
            trades * 0.1  # Activity bonus
        )
        
        # Penalty for too few trades
        if trades < 5:
            fitness *= 0.5
        
        strategy.fitness = fitness
        strategy.trades_executed = trades
        
        return fitness
    
    def selection(self) -> TradingStrategy:
        """Tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda s: s.fitness)
    
    def crossover(self, parent1: TradingStrategy, parent2: TradingStrategy) -> TradingStrategy:
        """Crossover two strategies to create offspring"""
        if random.random() > self.crossover_rate:
            return TradingStrategy(**asdict(parent1))
        
        # Uniform crossover
        child = TradingStrategy(
            rsi_weight=random.choice([parent1.rsi_weight, parent2.rsi_weight]),
            macd_weight=random.choice([parent1.macd_weight, parent2.macd_weight]),
            bollinger_weight=random.choice([parent1.bollinger_weight, parent2.bollinger_weight]),
            volume_weight=random.choice([parent1.volume_weight, parent2.volume_weight]),
            momentum_weight=random.choice([parent1.momentum_weight, parent2.momentum_weight]),
            buy_threshold=random.choice([parent1.buy_threshold, parent2.buy_threshold]),
            sell_threshold=random.choice([parent1.sell_threshold, parent2.sell_threshold]),
            stop_loss=random.choice([parent1.stop_loss, parent2.stop_loss]),
            take_profit=random.choice([parent1.take_profit, parent2.take_profit]),
            position_size=random.choice([parent1.position_size, parent2.position_size]),
            max_positions=random.choice([parent1.max_positions, parent2.max_positions]),
            holding_period=random.choice([parent1.holding_period, parent2.holding_period]),
            cooldown_period=random.choice([parent1.cooldown_period, parent2.cooldown_period]),
            generation=self.generation + 1
        )
        
        return child
    
    def evolve(self, market_data: Dict):
        """Run one generation of evolution"""
        logger.info(f"\n🧬 Generation {self.generation}")
        
        # Evaluate fitness for all strategies
        for strategy in self.population:
            self.calculate_fitness(strategy, market_data)
        
        # Sort by fitness
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        
        # Track best strategy
        self.best_strategy = self.population[0]
        self.hall_of_fame.append(TradingStrategy(**asdict(self.best_strategy)))
        
        # Log statistics
        fitnesses = [s.fitness for s in self.population]
        logger.info(f"   Best fitness: {max(fitnesses):.2f}")
        logger.info(f"   Avg fitness: {np.mean(fitnesses):.2f}")
        logger.info(f"   Diversity: {np.std(fitnesses):.2f}")
        
        # Store history
        self.fitness_history.append({
            'generation': self.generation,
            'best': max(fitnesses),
            'average': np.mean(fitnesses),
            'std': np.std(fitnesses)
        })
        
        # Create new population
        new_population = []
        
        # Elitism - keep best strategies
        for i in range(self.elitism_count):
            new_population.append(TradingStrategy(**asdict(self.population[i])))
        
        # Create rest of population through selection, crossover, mutation
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()
            
            child = self.crossover(parent1, parent2)
            child.mutate(self.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def get_best_strategy(self) -> TradingStrategy:
        """Return the current best strategy"""
        return self.best_strategy if self.best_strategy else self.population[0]
    
    def analyze(self, market_data: Dict) -> Dict:
        """Analyze market with best evolved strategy"""
        strategy = self.get_best_strategy()
        
        # Extract current indicators from market data
        prices = market_data.get('prices', [])
        if len(prices) < 50:
            return {'signal': 'hold', 'confidence': 0.0}
        
        current_price = prices[-1]
        
        # Calculate indicators (same as fitness function)
        gains = [max(0, prices[i] - prices[i-1]) for i in range(-49, 0)]
        losses = [max(0, prices[i-1] - prices[i]) for i in range(-49, 0)]
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))
        
        momentum = (current_price - prices[-20]) / prices[-20]
        
        ema_12 = np.mean(prices[-12:])
        ema_26 = np.mean(prices[-26:])
        macd = (ema_12 - ema_26) / current_price
        
        sma_20 = np.mean(prices[-20:])
        std_20 = np.std(prices[-20:])
        bollinger = (current_price - sma_20) / (std_20 + 1e-10)
        
        volume_signal = np.random.uniform(-1, 1)  # Placeholder
        
        # Calculate signal
        signal = (
            strategy.rsi_weight * (50 - rsi) / 50 +
            strategy.macd_weight * macd * 10 +
            strategy.bollinger_weight * bollinger +
            strategy.volume_weight * volume_signal +
            strategy.momentum_weight * momentum * 10
        )
        
        total_weight = (strategy.rsi_weight + strategy.macd_weight + 
                       strategy.bollinger_weight + strategy.volume_weight + 
                       strategy.momentum_weight)
        if total_weight > 0:
            signal = signal / total_weight
        
        # Determine action
        if signal > strategy.buy_threshold:
            action = 'buy'
            confidence = min(1.0, abs(signal - strategy.buy_threshold))
        elif signal < strategy.sell_threshold:
            action = 'sell'
            confidence = min(1.0, abs(signal - strategy.sell_threshold))
        else:
            action = 'hold'
            confidence = 1.0 - abs(signal)
        
        return {
            'signal': action,
            'confidence': confidence,
            'raw_signal': signal,
            'strategy_fitness': strategy.fitness,
            'generation': strategy.generation,
            'stop_loss': strategy.stop_loss,
            'take_profit': strategy.take_profit,
            'reason': f'ASI evolved strategy (Gen {strategy.generation}, Fitness: {strategy.fitness:.1f})'
        }

# Test the ASI system
if __name__ == "__main__":
    print("=" * 60)
    print("🧬 ULTRATHINK ASI - GENETIC STRATEGY EVOLUTION")
    print("✨ Self-Improving Trading Through Evolution")
    print("=" * 60)
    
    # Initialize ASI
    asi = ASIGeneticEvolution(population_size=50)
    
    # Generate synthetic market data for testing
    np.random.seed(42)
    prices = [100]
    for _ in range(500):
        change = np.random.randn() * 2
        prices.append(prices[-1] * (1 + change/100))
    
    market_data = {'prices': prices}
    
    # Run evolution for several generations
    print("\n🔬 Running evolution...")
    for gen in range(10):
        asi.evolve(market_data)
        
        if gen % 3 == 0:
            best = asi.get_best_strategy()
            print(f"\n📊 Best strategy stats:")
            print(f"   RSI weight: {best.rsi_weight:.2f}")
            print(f"   Momentum weight: {best.momentum_weight:.2f}")
            print(f"   Stop loss: {best.stop_loss:.2%}")
            print(f"   Take profit: {best.take_profit:.2%}")
    
    # Test final analysis
    print("\n🎯 Testing evolved strategy...")
    result = asi.analyze(market_data)
    print(f"   Signal: {result['signal'].upper()}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Strategy fitness: {result['strategy_fitness']:.1f}")
    
    print("\n✅ ASI GENETIC EVOLUTION READY!")