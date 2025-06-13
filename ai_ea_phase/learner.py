import os
import random
import pickle
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Set, Optional
from collections import defaultdict, Counter
import numpy as np

MODEL_PATH = "models/strategy_model.pkl"
MIN_STRATEGIES_FOR_TRAINING = 5
MAX_STRATEGY_RULES = 6
MIN_STRATEGY_RULES = 3

class StrategyLearner:
    def __init__(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self.available_rules = [
            "Close > SMA20", "Close < SMA20", "RSI < 30", "RSI > 70", 
            "Volume > 1.5x MA", "MACD crossover", "MACD crossunder",
            "Price > Weekly High", "Price < Weekly Low", "ATR > 1.5x MA",
            "Bollinger Band Breakout", "Fibonacci Retracement 61.8%",
            "Support Bounce", "Resistance Break", "Order Block Activation",
            "Fair Value Gap", "Market Structure Shift", "Liquidity Sweep",
            "Candlestick Engulfing", "Candlestick Pinbar"
        ]
        self.rule_categories = self._categorize_rules()
        self.model_version = "1.2"

    def _categorize_rules(self) -> Dict[str, List[str]]:
        """Organize rules by their technical analysis category"""
        return {
            "trend": ["Close > SMA20", "Close < SMA20", "Price > Weekly High", "Price < Weekly Low"],
            "momentum": ["RSI < 30", "RSI > 70", "MACD crossover", "MACD crossunder"],
            "volatility": ["ATR > 1.5x MA", "Bollinger Band Breakout"],
            "volume": ["Volume > 1.5x MA"],
            "pattern": [
                "Support Bounce", "Resistance Break", "Fibonacci Retracement 61.8%",
                "Candlestick Engulfing", "Candlestick Pinbar"
            ],
            "advanced": [
                "Order Block Activation", "Fair Value Gap", 
                "Market Structure Shift", "Liquidity Sweep"
            ]
        }

    def _save_model(self, model_data: Dict) -> bool:
        """Save the trained model with versioning and checksum"""
        try:
            model_data["metadata"] = {
                "version": self.model_version,
                "created": datetime.now().isoformat(),
                "strategies_count": len(model_data.get("top_strategies", [])),
                "checksum": hashlib.md5(str(model_data).encode()).hexdigest()
            }
            
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}", exc_info=True)
            return False

    def _load_model(self) -> Optional[Dict]:
        """Load the model with validation checks"""
        if not os.path.exists(MODEL_PATH):
            return None
            
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
                
            # Validate model structure
            if not isinstance(model, dict) or "top_strategies" not in model:
                logging.warning("Invalid model structure found, rebuilding...")
                return None
                
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}", exc_info=True)
            return None

    def learn_and_generate(
        self, 
        all_strategies: List[Dict], 
        results: List[Dict], 
        num: int = 5,
        min_win_rate: float = 55.0,
        min_profit: float = 100.0
    ) -> List[Dict]:
        """
        Learn from strategy performance and generate new strategies
        Args:
            all_strategies: List of all available strategies
            results: Backtesting results for each strategy
            num: Number of new strategies to generate
            min_win_rate: Minimum win rate to consider a strategy successful
            min_profit: Minimum profit (in pips) to consider a strategy successful
        Returns:
            List of newly generated strategies
        """
        if len(all_strategies) < MIN_STRATEGIES_FOR_TRAINING:
            logging.warning(f"Insufficient strategies ({len(all_strategies)}) for training")
            return []

        # Create performance dictionary for quick lookup
        perf_dict = {r["name"]: r for r in results if "name" in r}
        
        # Filter successful strategies with enhanced criteria
        successful_strats = [
            s for s in all_strategies 
            if (perf_dict.get(s["name"], {}).get("win_rate", 0)) > min_win_rate and
               (perf_dict.get(s["name"], {}).get("profit", 0)) > min_profit and
               len(s.get("rules", [])) >= MIN_STRATEGY_RULES
        ]
        
        # If no successful strategies, use top 30% of all strategies
        if not successful_strats:
            logging.info("No strategies met success criteria, using top performers")
            scored_strats = sorted(
                all_strategies,
                key=lambda x: perf_dict.get(x["name"], {}).get("score", 0),
                reverse=True
            )
            successful_strats = scored_strats[:max(3, len(scored_strats) // 3)]
        
        # Train or load model with enhanced data
        model_data = self._load_model()
        if not model_data:
            logging.info("Training new strategy model...")
            model_data = self.train_model(successful_strats)
        else:
            logging.info("Updating existing model with new data...")
            model_data = self.update_model(model_data, successful_strats)
        
        # Generate new strategies with diversity enforcement
        new_strategies = self.generate_strategies(
            model_data, 
            successful_strats, 
            num,
            diversity_factor=0.7
        )
        
        return new_strategies

    def train_model(self, strategies: List[Dict]) -> Dict:
        """Train a new model with enhanced strategy analysis"""
        if not strategies:
            return {"top_strategies": [], "rule_stats": {}}
        
        # Analyze rule frequency and performance
        rule_stats = self._analyze_rules(strategies)
        
        model_data = {
            "top_strategies": sorted(
                strategies,
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:20],  # Keep top 20 strategies
            "rule_stats": rule_stats,
            "category_balance": self._calculate_category_balance(strategies),
            "last_trained": datetime.now().isoformat()
        }
        
        self._save_model(model_data)
        return model_data

    def update_model(self, existing_model: Dict, new_strategies: List[Dict]) -> Dict:
        """Update existing model with new strategies while maintaining quality"""
        if not new_strategies:
            return existing_model
            
        # Combine and deduplicate strategies
        all_strategies = existing_model["top_strategies"] + new_strategies
        unique_strategies = {self._strategy_hash(s): s for s in all_strategies}.values()
        
        # Retrain with updated strategies
        updated_model = self.train_model(list(unique_strategies))
        return updated_model

    def generate_strategies(
        self, 
        model_data: Dict, 
        base_strategies: List[Dict], 
        num: int = 5,
        diversity_factor: float = 0.5
    ) -> List[Dict]:
        """
        Generate new strategies with enhanced genetic algorithm approach
        Args:
            diversity_factor: 0-1 value controlling how much to prioritize diverse rules
        """
        new_strategies = []
        top_strategies = model_data.get("top_strategies", [])
        rule_stats = model_data.get("rule_stats", {})
        
        if not top_strategies:
            top_strategies = base_strategies
            
        for i in range(num):
            # Select parents with weighted probability based on performance
            parent_weights = [
                s.get("score", 1) for s in top_strategies
                if len(s.get("rules", [])) >= MIN_STRATEGY_RULES
            ]
            
            if not parent_weights or len(parent_weights) < 2:
                parents = random.sample(base_strategies, min(2, len(base_strategies)))
            else:
                parents = random.choices(
                    [s for s in top_strategies if len(s.get("rules", [])) >= MIN_STRATEGY_RULES],
                    weights=parent_weights,
                    k=random.randint(2, 3)
                )
            
            # Combine rules with diversity enhancement
            combined_rules = self._combine_rules(
                parents, 
                rule_stats,
                diversity_factor
            )
            
            # Create new strategy with balanced categories
            new_strategy = self._create_strategy(combined_rules)
            new_strategies.append(new_strategy)
        
        return new_strategies

    def _combine_rules(
        self, 
        parents: List[Dict], 
        rule_stats: Dict,
        diversity_factor: float
    ) -> List[str]:
        """Combine rules from parent strategies with diversity optimization"""
        # Collect all rules from parents
        all_rules = [rule for parent in parents for rule in parent.get("rules", [])]
        
        # Calculate rule scores (frequency vs performance)
        rule_scores = {}
        for rule in set(all_rules):
            freq = all_rules.count(rule) / len(all_rules)
            perf = rule_stats.get(rule, {}).get("avg_score", 1)
            # Balance between performance and diversity
            rule_scores[rule] = (perf * (1 - diversity_factor)) + ((1 - freq) * diversity_factor)
        
        # Select top scoring rules, ensuring category diversity
        selected_rules = set()
        category_counts = defaultdict(int)
        
        # Sort rules by score and select while maintaining category balance
        for rule in sorted(rule_scores, key=lambda x: rule_scores[x], reverse=True):
            rule_category = next(
                (cat for cat, rules in self.rule_categories.items() if rule in rules),
                "other"
            )
            
            # Limit rules per category
            if category_counts[rule_category] < 2 or len(selected_rules) < MIN_STRATEGY_RULES:
                selected_rules.add(rule)
                category_counts[rule_category] += 1
                
            if len(selected_rules) >= MAX_STRATEGY_RULES:
                break
        
        # Ensure minimum rule count
        while len(selected_rules) < MIN_STRATEGY_RULES:
            new_rule = random.choice([
                r for r in self.available_rules 
                if r not in selected_rules
            ])
            selected_rules.add(new_rule)
        
        return list(selected_rules)

    def _create_strategy(self, rules: List[str]) -> Dict:
        """Create a new strategy with balanced configuration"""
        # Determine direction based on rule types
        bullish_rules = {"Close > SMA20", "RSI < 30", "MACD crossover", "Support Bounce"}
        bearish_rules = {"Close < SMA20", "RSI > 70", "MACD crossunder", "Resistance Break"}
        
        bull_count = sum(1 for r in rules if r in bullish_rules)
        bear_count = sum(1 for r in rules if r in bearish_rules)
        
        direction = "buy" if bull_count > bear_count else (
            "sell" if bear_count > bull_count else 
            random.choice(["buy", "sell"])
        )
        
        return {
            "name": f"AI_{direction}_Strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "rules": rules,
            "direction": direction,
            "version": self.model_version,
            "created": datetime.now().isoformat(),
            "active": True,
            "score": 0  # Will be updated after evaluation
        }

    def _analyze_rules(self, strategies: List[Dict]) -> Dict:
        """Analyze rule performance across all strategies"""
        rule_stats = defaultdict(lambda: {"count": 0, "total_score": 0, "strategies": []})
        
        for strategy in strategies:
            score = strategy.get("score", 1)
            for rule in strategy.get("rules", []):
                rule_stats[rule]["count"] += 1
                rule_stats[rule]["total_score"] += score
                rule_stats[rule]["strategies"].append(strategy["name"])
        
        # Calculate average scores
        for rule in rule_stats:
            rule_stats[rule]["avg_score"] = (
                rule_stats[rule]["total_score"] / rule_stats[rule]["count"]
                if rule_stats[rule]["count"] > 0 else 1
            )
        
        return dict(rule_stats)

    def _calculate_category_balance(self, strategies: List[Dict]) -> Dict:
        """Calculate the distribution of rule categories across strategies"""
        category_counts = defaultdict(int)
        total_rules = 0
        
        for strategy in strategies:
            for rule in strategy.get("rules", []):
                category = next(
                    (cat for cat, rules in self.rule_categories.items() if rule in rules),
                    "other"
                )
                category_counts[category] += 1
                total_rules += 1
        
        return {
            cat: count / total_rules if total_rules > 0 else 0
            for cat, count in category_counts.items()
        }

    def _strategy_hash(self, strategy: Dict) -> str:
        """Create a unique hash for a strategy based on its rules"""
        rule_str = "|".join(sorted(strategy.get("rules", [])))
        return hashlib.md5(rule_str.encode()).hexdigest()