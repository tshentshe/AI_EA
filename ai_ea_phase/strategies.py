import json
import os
import logging
from datetime import datetime
from typing import List, Dict

class StrategyManager:
    def __init__(self, path: str = "data/strategies.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({"user": [], "generated": []}, f)
    
    def load_strategies(self) -> Dict[str, List[Dict]]:
        try:
            with open(self.path, "r") as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Failed to load strategies: {str(e)}")
            return {"user": [], "generated": []}
    
    def save_strategies(self, strategies: dict):
        try:
            with open(self.path, "w") as file:
                json.dump(strategies, file, indent=2)
        except Exception as e:
            logging.error(f"Failed to save strategies: {str(e)}")

    def get_all_strategies(self) -> List[Dict]:
        data = self.load_strategies()
        return data["user"] + data["generated"]

    def get_active_strategies(self) -> List[Dict]:
        data = self.load_strategies()
        return [s for s in data["user"] + data["generated"] if s.get("active", True)]

    def add_user_strategy(self, strategy: Dict):
        data = self.load_strategies()
        strategy["version"] = 1
        strategy["created"] = datetime.now().isoformat()
        strategy["active"] = True
        data["user"].append(strategy)
        self.save_strategies(data)

    def add_generated_strategies(self, new_strategies: List[Dict]):
        data = self.load_strategies()
        data["generated"].extend(new_strategies)
        self.save_strategies(data)
    
    def refresh_strategies(self, results: List[Dict]):
        """Refresh strategy pool based on performance results"""
        data = self.load_strategies()
        
        # Deactivate underperforming strategies
        for strategy in data["generated"]:
            for result in results:
                if result["name"] == strategy["name"]:
                    if result["win_rate"] < 40 or result["profit"] < 0:
                        strategy["active"] = False
                    break
        
        # Activate top performers
        top_performers = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
        for strategy in data["generated"]:
            for top in top_performers:
                if strategy["name"] == top["name"]:
                    strategy["active"] = True
                    break
        
        self.save_strategies(data)
    
    def log_performance(self, results: List[Dict]):
        log_dir = "logs/performance"
        os.makedirs(log_dir, exist_ok=True)
        
        filename = f"performance_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(log_dir, filename)
        
        try:
            with open(filepath, "a") as f:
                for result in results:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "strategy": result["name"],
                        "symbol": result.get("symbol", "N/A"),
                        "win_rate": result["win_rate"],
                        "profit": result["profit"],
                        "direction": result["direction"]
                    }
                    f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logging.error(f"Failed to log performance: {str(e)}")