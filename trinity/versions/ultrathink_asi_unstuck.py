#!/usr/bin/env python3
"""
ULTRATHINK ASI UNSTUCK VERSION
Never gets stuck at 50% - guaranteed diversity
"""

import numpy as np
import random
import logging
from typing import Dict, Tuple
import time

logger = logging.getLogger(__name__)

class UnstuckGeneticStrategy:
    """ASI that NEVER gets stuck"""
    
    def __init__(self):
        # Sacred constants
        self.PI = 3.14159265359
        self.PHI = 1.618033988749
        self.SACRED_69 = 69
        
        # Dynamic parameters
        self.population_size = 30
        self.mutation_rate = 0.2  # Higher for more diversity
        self.crossover_rate = 0.7
        
        # Anti-stuck mechanisms
        self.stuck_iterations = 0
        self.last_decisions = []
        self.force_action_counter = 0
        self.diversity_boost = 1.0
        
        # Gene ranges - VERY WIDE for maximum diversity
        self.gene_ranges = {
            'rsi_buy': (15, 45),      # Wide range
            'rsi_sell': (55, 85),      # Wide range
            'ma_weight': (0.1, 3.0),
            'momentum_threshold': (0.0001, 0.1),
            'stop_loss': (0.001, 0.1),
            'take_profit': (0.005, 0.2),
            'sacred_bonus': (0.0, 0.5),
            'action_bias': (-0.3, 0.3),  # New: bias toward action
            'volatility_multiplier': (0.5, 2.0),
            'trend_strength': (0.1, 1.0)
        }
        
        self.population = []
        self.generation = 1
        self._initialize_ultra_diverse_population()
        
        logger.info(f"🚀 UnstuckASI initialized with ultra-diversity")
    
    def _initialize_ultra_diverse_population(self):
        """Create ULTRA diverse population with guaranteed variety"""
        self.population = []
        
        # Create 6 different strategy types
        strategies = ['aggressive', 'conservative', 'sacred', 'random', 'momentum', 'contrarian']
        
        for i in range(self.population_size):
            strategy = strategies[i % len(strategies)]
            individual = {}
            
            for gene, (min_val, max_val) in self.gene_ranges.items():
                if strategy == 'aggressive':
                    # Favor lower buy, higher sell thresholds
                    if 'buy' in gene:
                        val = random.uniform(min_val, min_val + (max_val - min_val) * 0.4)
                    elif 'sell' in gene:
                        val = random.uniform(max_val - (max_val - min_val) * 0.4, max_val)
                    else:
                        val = random.uniform(max_val - (max_val - min_val) * 0.3, max_val)
                
                elif strategy == 'conservative':
                    # Opposite of aggressive
                    if 'buy' in gene:
                        val = random.uniform(max_val - (max_val - min_val) * 0.4, max_val)
                    elif 'sell' in gene:
                        val = random.uniform(min_val, min_val + (max_val - min_val) * 0.4)
                    else:
                        val = random.uniform(min_val, min_val + (max_val - min_val) * 0.3)
                
                elif strategy == 'sacred':
                    # Use sacred numbers
                    if 'rsi_buy' in gene:
                        val = 31.4  # Pi * 10
                    elif 'rsi_sell' in gene:
                        val = self.SACRED_69
                    elif 'sacred' in gene:
                        val = max_val  # Max sacred bonus
                    else:
                        val = random.uniform(min_val, max_val) * self.PHI
                
                elif strategy == 'momentum':
                    # Focus on momentum
                    if 'momentum' in gene or 'trend' in gene:
                        val = random.uniform(max_val - (max_val - min_val) * 0.2, max_val)
                    else:
                        val = random.uniform(min_val, max_val)
                
                elif strategy == 'contrarian':
                    # Opposite of current trend
                    if 'action_bias' in gene:
                        val = random.choice([min_val, max_val])  # Extreme bias
                    else:
                        val = random.gauss((min_val + max_val) / 2, (max_val - min_val) / 4)
                
                else:  # random
                    val = random.uniform(min_val, max_val)
                
                # Ensure within bounds
                individual[gene] = max(min_val, min(max_val, val))
            
            individual['fitness'] = random.uniform(0, 0.1)  # Start with small random fitness
            individual['strategy_type'] = strategy
            self.population.append(individual)
    
    def get_decision(self, market_data: Dict) -> Tuple[str, float]:
        """Get trading decision - GUARANTEED not to get stuck"""
        
        # Extract market indicators
        rsi = market_data.get('rsi', 50)
        price_change = market_data.get('price_change', 0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        volatility = market_data.get('volatility', 0.01)
        
        # Check for stuck pattern
        if len(self.last_decisions) >= 5:
            recent = self.last_decisions[-5:]
            if all(d[0] == 'hold' for d in recent) or all(abs(d[1] - 0.5) < 0.01 for d in recent):
                self.stuck_iterations += 1
                logger.warning(f"⚠️ Stuck detected! Counter: {self.stuck_iterations}")
                
                # FORCE ACTION
                if self.stuck_iterations > 2:
                    forced_signal = 'buy' if random.random() > 0.5 else 'sell'
                    forced_confidence = random.uniform(0.3, 0.7)
                    logger.warning(f"🔥 FORCING ACTION: {forced_signal} @ {forced_confidence:.2%}")
                    self.last_decisions.append((forced_signal, forced_confidence))
                    self.stuck_iterations = 0
                    self._shake_up_population()
                    return forced_signal, forced_confidence
        
        # Get best individual
        best = max(self.population, key=lambda x: x['fitness'])
        
        # Calculate base signals
        buy_signal = rsi < best['rsi_buy']
        sell_signal = rsi > best['rsi_sell']
        momentum_buy = price_change > best['momentum_threshold']
        momentum_sell = price_change < -best['momentum_threshold']
        
        # Apply action bias to prevent holding
        action_bias = best['action_bias']
        
        # Sacred moment check
        current_second = int(time.time()) % 100
        sacred_boost = 0
        if current_second == 31:  # Pi moment
            sacred_boost = 0.2
            logger.info("✨ Pi moment boost!")
        elif current_second == 69:  # Sacred moment
            sacred_boost = 0.3
            logger.info("✨ Sacred 69 boost!")
        
        # Calculate confidence with multiple factors
        if buy_signal or momentum_buy:
            base_conf = 0.4 + abs(best['rsi_buy'] - rsi) / 100
            confidence = base_conf + action_bias + sacred_boost
            confidence *= best['volatility_multiplier'] * volatility + 1
            signal = 'buy'
        elif sell_signal or momentum_sell:
            base_conf = 0.4 + abs(rsi - best['rsi_sell']) / 100
            confidence = base_conf - action_bias + sacred_boost
            confidence *= best['volatility_multiplier'] * volatility + 1
            signal = 'sell'
        else:
            # Even in neutral, add some randomness
            random_factor = random.uniform(-0.1, 0.1)
            if random_factor + action_bias > 0.05:
                signal = 'buy'
                confidence = 0.25 + abs(random_factor) + sacred_boost
            elif random_factor + action_bias < -0.05:
                signal = 'sell'
                confidence = 0.25 + abs(random_factor) + sacred_boost
            else:
                signal = 'hold'
                confidence = 0.2 + sacred_boost  # Low confidence for hold
        
        # Apply sacred bonus
        if abs(rsi - self.SACRED_69) < 5:
            confidence += best['sacred_bonus'] * 0.1
        if abs(price_change * 100 - self.PI) < 1:
            confidence += best['sacred_bonus'] * 0.1
        
        # Ensure confidence is reasonable
        confidence = max(0.1, min(1.0, confidence))
        
        # Force diversity if too many holds
        if signal == 'hold' and random.random() < 0.2:  # 20% chance to override hold
            signal = 'buy' if rsi < 50 else 'sell'
            confidence = random.uniform(0.25, 0.4)
            logger.info(f"🎲 Random override: {signal}")
        
        # Track decisions
        self.last_decisions.append((signal, confidence))
        if len(self.last_decisions) > 10:
            self.last_decisions.pop(0)
        
        return signal, confidence
    
    def _shake_up_population(self):
        """Dramatically shake up population when stuck"""
        logger.warning("🌪️ Shaking up population!")
        
        # Replace bottom 50% with new random individuals
        self.population.sort(key=lambda x: x['fitness'])
        
        for i in range(self.population_size // 2):
            for gene, (min_val, max_val) in self.gene_ranges.items():
                # Completely random new values
                self.population[i][gene] = random.uniform(min_val, max_val)
            self.population[i]['fitness'] = random.uniform(0, 0.2)
        
        # Increase mutation rate temporarily
        self.mutation_rate = min(0.5, self.mutation_rate * 1.5)
        
        # Reset stuck counter
        self.stuck_iterations = 0
    
    def evolve(self, performance_metrics):
        """Evolve population with anti-stuck mechanisms"""
        
        # Update fitness scores
        for i, ind in enumerate(self.population):
            if i < len(performance_metrics):
                # Add randomness to fitness to prevent convergence
                ind['fitness'] = performance_metrics[i] + random.uniform(-0.05, 0.05)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Check if population is converging (bad!)
        fitness_variance = np.var([ind['fitness'] for ind in self.population])
        if fitness_variance < 0.01:
            logger.warning("⚠️ Population converging - forcing diversity!")
            self._shake_up_population()
        
        # Elitism with diversity requirement
        elite_size = max(2, self.population_size // 10)
        new_population = self.population[:elite_size].copy()
        
        # Add some completely random individuals for diversity
        random_count = max(2, self.population_size // 10)
        for _ in range(random_count):
            individual = {}
            for gene, (min_val, max_val) in self.gene_ranges.items():
                individual[gene] = random.uniform(min_val, max_val)
            individual['fitness'] = 0.0
            individual['strategy_type'] = 'random'
            new_population.append(individual)
        
        # Crossover and mutation for the rest
        while len(new_population) < self.population_size:
            # Select parents from different parts of population for diversity
            parent1 = random.choice(self.population[:self.population_size // 2])
            parent2 = random.choice(self.population[self.population_size // 4:])
            
            child = {}
            for gene in self.gene_ranges:
                # Crossover
                if random.random() < self.crossover_rate:
                    child[gene] = parent1[gene]
                else:
                    child[gene] = parent2[gene]
                
                # Mutation with variable rate
                mutation_chance = self.mutation_rate * (1 + self.stuck_iterations * 0.1)
                if random.random() < mutation_chance:
                    min_val, max_val = self.gene_ranges[gene]
                    # Stronger mutation
                    mutation = random.uniform(-0.3, 0.3) * (max_val - min_val)
                    child[gene] = max(min_val, min(max_val, child[gene] + mutation))
            
            child['fitness'] = 0.0
            child['strategy_type'] = 'evolved'
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Gradually reduce mutation rate unless stuck
        if self.stuck_iterations == 0:
            self.mutation_rate = max(0.1, self.mutation_rate * 0.98)
        
        logger.info(f"🧬 Generation {self.generation} | Diversity: {fitness_variance:.4f}")