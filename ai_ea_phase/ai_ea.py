import time
import logging
from datetime import datetime 
from strategies import StrategyManager
from learner import StrategyLearner
from evaluator import StrategyEvaluator
from data_fetcher import MT5DataFetcher
from executor import MT5Executor
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Configuration
MT5_LOGIN = 210194168
MT5_PASSWORD = "7$Hepang"
MT5_SERVER = "Exness-MT5Trial9"
SYMBOLS = ["BTCUSDm", "EURJPYm", "US30m", "USTECm"]
LOT_SIZE = 0.02
TIMEFRAME = mt5.TIMEFRAME_H2
BARS = 550
SLEEP_INTERVAL = 5 * 60
MAX_CONNECTION_ATTEMPTS = 10
CONNECTION_RETRY_DELAY = 5
MAX_POSITIONS_PER_SYMBOL = 2
MIN_WIN_RATE = 45.0
TRAILING_START = 50 # Pips profit to activate trailing stop
TRAILING_STEP = 20  # Pips for trailing step
BREAKEVEN_AT = 30   # Pips profit to move to breakeven

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_ea.log"),
        logging.StreamHandler()
    ]
)

class ReversalDetector:
    def __init__(self):
        self.patterns = {
            'pin_bar': self._detect_pin_bar,
            'engulfing': self._detect_engulfing,
            'rsi_divergence': self._detect_rsi_divergence,
            'macd_cross': self._detect_macd_cross
        }
    
    def detect(self, df, symbol):
        """
        Detect potential reversal signals in the given DataFrame.
        Returns: (bool, str) - reversal detected and direction ('bullish' or 'bearish')
        """
        signals = []
        
        # Check for RSI divergence
        rsi_div = self._detect_rsi_divergence(df)
        if rsi_div:
            signals.append(rsi_div)
        
        # Check for MACD cross
        macd_cross = self._detect_macd_cross(df)
        if macd_cross:
            signals.append(macd_cross)
        
        # Check candlestick patterns
        pin_bar = self._detect_pin_bar(df)
        if pin_bar:
            signals.append(pin_bar)
        
        engulfing = self._detect_engulfing(df)
        if engulfing:
            signals.append(engulfing)
        
        # Analyze signals
        if not signals:
            return False, None
        
        # Count bullish and bearish signals
        bullish_count = sum(1 for s in signals if s == 'bullish')
        bearish_count = sum(1 for s in signals if s == 'bearish')
        
        # Require at least 2 signals in the same direction
        if bullish_count >= 2:
            logging.info(f"Bullish reversal detected on {symbol}")
            return True, 'bullish'
        elif bearish_count >= 2:
            logging.info(f"Bearish reversal detected on {symbol}")
            return True, 'bearish'
        
        return False, None

    def _detect_rsi_divergence(self, df, period=14, lookback=20):
        """Detect bullish/bearish RSI divergence"""
        if 'rsi' not in df.columns or len(df) < lookback + period:
            return None
            
        prices = df['close'].tail(lookback).values
        rsis = df['rsi'].tail(lookback).values
        
        # Find peaks and troughs
        price_peaks = np.argsort(prices)[-2:]
        price_troughs = np.argsort(prices)[:2]
        rsi_peaks = np.argsort(rsis)[-2:]
        rsi_troughs = np.argsort(rsis)[:2]
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if price_peaks[-1] > price_peaks[-2] and rsi_peaks[-1] < rsi_peaks[-2]:
                return 'bearish'
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if price_troughs[0] < price_troughs[1] and rsi_troughs[0] > rsi_troughs[1]:
                return 'bullish'
        
        return None

    def _detect_macd_cross(self, df):
        """Detect MACD crossover/crossunder"""
        if 'macd_line' not in df.columns or 'macd_signal' not in df.columns or len(df) < 3:
            return None
            
        macd_line = df['macd_line'].tail(2).values
        macd_signal = df['macd_signal'].tail(2).values
        
        # Bullish cross: MACD crosses above signal line
        if macd_line[-2] < macd_signal[-2] and macd_line[-1] > macd_signal[-1]:
            return 'bullish'
        
        # Bearish cross: MACD crosses below signal line
        if macd_line[-2] > macd_signal[-2] and macd_line[-1] < macd_signal[-1]:
            return 'bearish'
        
        return None

    def _detect_pin_bar(self, df):
        """Detect bullish/bearish pin bar patterns"""
        if len(df) < 1:
            return None
            
        candle = df.iloc[-1]
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return None
            
        body_ratio = body_size / total_range
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        # Bullish pin bar: long lower wick
        if body_ratio < 0.3 and lower_wick > 2 * upper_wick and lower_wick > 2 * body_size:
            return 'bullish'
        
        # Bearish pin bar: long upper wick
        if body_ratio < 0.3 and upper_wick > 2 * lower_wick and upper_wick > 2 * body_size:
            return 'bearish'
        
        return None

    def _detect_engulfing(self, df):
        """Detect bullish/bearish engulfing patterns"""
        if len(df) < 2:
            return None
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish engulfing: current green candle completely engulfs previous red candle
        if (current['close'] > current['open'] and 
            prev['close'] < prev['open'] and
            current['open'] < prev['close'] and 
            current['close'] > prev['open']):
            return 'bullish'
        
        # Bearish engulfing: current red candle completely engulfs previous green candle
        if (current['close'] < current['open'] and 
            prev['close'] > prev['open'] and
            current['open'] > prev['close'] and 
            current['close'] < prev['open']):
            return 'bearish'
        
        return None

def initialize_mt5():
    """Initialize MT5 connection with retries"""
    attempts = 0
    while attempts < MAX_CONNECTION_ATTEMPTS:
        try:
            if not mt5.initialize():
                raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
            
            if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
                raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")
            
            logging.info("Successfully connected to MT5")
            return True
        except Exception as e:
            attempts += 1
            logging.error(f"Connection attempt {attempts} failed: {str(e)}")
            time.sleep(CONNECTION_RETRY_DELAY)
    
    logging.critical("Failed to connect to MT5 after multiple attempts")
    return False

def run_cycle(cycle_count):
    logging.info(f"Cycle #{cycle_count} start: {datetime.now()}")
    
    if not initialize_mt5():
        return

    try:
        # Initialize components
        strategy_manager = StrategyManager()
        strategy_learner = StrategyLearner()
        strategy_evaluator = StrategyEvaluator()
        fetcher = MT5DataFetcher()
        executor = MT5Executor(
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER
        )
        detector = ReversalDetector()

        # Load strategies
        all_strategies = strategy_manager.get_active_strategies()
        logging.info(f"Loaded {len(all_strategies)} active strategies")
        
        # Process each symbol
        for symbol in SYMBOLS:
            logging.info(f"Processing symbol: {symbol}")
            
            # Fetch fresh market data
            try:
                market_data = fetcher.get_candles(symbol, TIMEFRAME, BARS)
                if market_data is None or len(market_data) < 100:
                    logging.error(f"Insufficient data for {symbol} ({len(market_data) if market_data else 0} bars)")
                    continue
            except Exception as e:
                logging.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue

            # Detect potential reversals
            reversal_detected, reversal_direction = detector.detect(market_data, symbol)
            
            # Manage existing positions
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                positions = []
                
            for position in positions:
                try:
                    # Get current price
                    tick = mt5.symbol_info_tick(symbol)
                    current_price = tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid
                    
                    # Calculate profit in pips
                    pip_multiplier = executor.get_pip_multiplier(symbol)
                    profit = (current_price - position.price_open) * (1 if position.type == mt5.ORDER_TYPE_BUY else -1)
                    profit_pips = profit / pip_multiplier
                    
                    # Trailing stop logic
                    if profit_pips > TRAILING_START:
                        new_sl = position.price_open + (TRAILING_STEP * pip_multiplier * 
                                                       (1 if position.type == mt5.ORDER_TYPE_BUY else -1))
                        
                        # Ensure SL is better than current
                        if ((position.type == mt5.ORDER_TYPE_BUY and new_sl > position.sl) or
                            (position.type == mt5.ORDER_TYPE_SELL and new_sl < position.sl)):
                            executor.update_sl(position.ticket, new_sl)
                            logging.info(f"Updated trailing SL for {symbol} to {new_sl:.5f}")
                    
                    # Breakeven logic
                    elif profit_pips > BREAKEVEN_AT:
                        if ((position.type == mt5.ORDER_TYPE_BUY and position.sl < position.price_open) or
                            (position.type == mt5.ORDER_TYPE_SELL and position.sl > position.price_open)):
                            executor.update_sl(position.ticket, position.price_open)
                            logging.info(f"Moved to breakeven for {symbol}")
                    
                    # Close profitable trades before reversal turns them to loss
                    if reversal_detected and profit_pips > 0:
                        logging.info(f"Closing profitable trade before reversal: {symbol} {position.ticket}")
                        executor.close_position(position.ticket)
                        
                        # Open hedge position (optional)
                        current_positions = mt5.positions_get(symbol=symbol) or []
                        if len(current_positions) < MAX_POSITIONS_PER_SYMBOL:
                            hedge_direction = "sell" if position.type == mt5.ORDER_TYPE_BUY else "buy"
                            hedge_tp = position.tp - (position.tp - position.price_open) * 0.5
                            
                            executor.place_order(
                                symbol=symbol,
                                lot=position.volume,
                                order_type=hedge_direction,
                                tp=hedge_tp
                            )
                            logging.info(f"Opened hedge position for {symbol}")
                
                except Exception as e:
                    logging.error(f"Error managing position {position.ticket}: {str(e)}")

            # Evaluate strategies
            results = []
            for strategy in all_strategies:
                try:
                    result = strategy_evaluator.evaluate_strategy(strategy, market_data, symbol)
                    result["strategy"] = strategy
                    results.append(result)
                    logging.info(f"Evaluated {strategy['name']} on {symbol}: "
                                f"Win Rate={result['win_rate']:.2f}%, Profit={result['profit']:.2f} pips")
                except Exception as e:
                    logging.error(f"Evaluation failed for {strategy['name']} on {symbol}: {str(e)}")

            # Find best strategy
            if results:
                # Sort strategies by score and select top 3
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                top_strategies = results[:3]
                
                for best_strategy in top_strategies:
                    # Execute trade if meets criteria
                    if best_strategy["win_rate"] > MIN_WIN_RATE and best_strategy["profit"] > 0:
                        try:
                            positions = mt5.positions_get(symbol=symbol) or []
                            positions_count = len(positions)
                            
                            if positions_count < MAX_POSITIONS_PER_SYMBOL:
                                executor.place_order(
                                    symbol=symbol,
                                    lot=LOT_SIZE,
                                    order_type=best_strategy['strategy']['direction']
                                )
                                logging.info(f"Executed {best_strategy['strategy']['direction']} order using {best_strategy['strategy']['name']}")
                                break  # Only execute one trade per symbol
                            else:
                                logging.info(f"Max positions ({MAX_POSITIONS_PER_SYMBOL}) reached for {symbol}, skipping")
                        except Exception as e:
                            logging.error(f"Execution failed for {symbol}: {str(e)}")
                    else:
                        logging.info(f"No trade: Strategy criteria not met for {symbol} with {best_strategy['strategy']['name']}")
            else:
                logging.warning(f"No valid strategies for {symbol}")

        # Log performance and generate new strategies
        try:
            strategy_manager.log_performance(results)
            new_strategies = strategy_learner.learn_and_generate(all_strategies, results)
            strategy_manager.add_generated_strategies(new_strategies)
            logging.info(f"Added {len(new_strategies)} new strategies")
            
            # Periodically refresh strategy pool
            if cycle_count % 5 == 0:
                strategy_manager.refresh_strategies(results)
                logging.info("Refreshed strategy pool based on performance")
                
        except Exception as e:
            logging.error(f"Learning/generation failed: {str(e)}")

    except Exception as e:
        logging.error(f"Unhandled error in cycle: {str(e)}", exc_info=True)
    finally:
        mt5.shutdown()
        logging.info("Disconnected from MT5")

    logging.info(f"Cycle #{cycle_count} complete")

def main():
    logging.info("AI EA running in live loop mode")
    cycle_count = 0
    while True:
        cycle_count += 1
        run_cycle(cycle_count)
        logging.info(f"Sleeping {SLEEP_INTERVAL//60} minutes")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()