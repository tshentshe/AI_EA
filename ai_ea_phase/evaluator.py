import os
import logging
import json
from typing import Dict
from Backtester import Backtester

PERFORMANCE_FILE = "data/performance.json"

class StrategyEvaluator:
    def __init__(self):
        os.makedirs(os.path.dirname(PERFORMANCE_FILE), exist_ok=True)
        self.backtester = Backtester()
        
    def evaluate_strategy(self, strategy: Dict, market_data, symbol: str) -> Dict:
        """
        Evaluate strategy using proper backtesting
        Returns: {
            "name": strategy name,
            "win_rate": calculated win rate,
            "profit": estimated profit,
            "direction": strategy direction,
            "symbol": symbol,
            "score": performance score
        }
        """
        result = self.backtester.test_strategy(strategy, market_data, symbol)
        return {
            "name": strategy["name"],
            "win_rate": result["win_rate"],
            "profit": result["profit"],
            "direction": strategy["direction"],
            "symbol": symbol,
            "score": result["score"]
        }
    
    def save_performance(self, performance_data: Dict):
        try:
            with open(PERFORMANCE_FILE, "w") as f:
                json.dump(performance_data, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save performance: {e}")