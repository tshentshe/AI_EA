import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class MT5DataFetcher:
    def __init__(self):
        self.connected = False
        self.connect()
        
    def connect(self):
        if not mt5.initialize():
            raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
        self.connected = True
        logging.info("Connected to MT5 for data fetching")
    
    def shutdown(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logging.info("Disconnected from MT5")

    def get_candles(self, symbol: str, timeframe, bars=500):
        if not self.connected:
            self.connect()
            
        # Calculate required time range
        time_now = datetime.utcnow()
        time_from = time_now - timedelta(days=31)  # 31 days back as default
        
        # Get rates
        rates = mt5.copy_rates_from(symbol, timeframe, time_now, bars)
        
        if rates is None or len(rates) == 0:
            # Try with larger time range
            rates = mt5.copy_rates_from(symbol, timeframe, time_from, bars)
            if rates is None or len(rates) == 0:
                logging.error(f"No data returned for {symbol} {timeframe}")
                return None
                
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Calculate essential technical indicators
        df['sma20'] = df['close'].rolling(20).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['atr'] = self.calculate_atr(df, 14)
        df['macd'] = self.calculate_macd(df['close'])
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['volume_ma'] = df['real_volume'].rolling(20).mean()
        
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def calculate_macd(self, close, fast=12, slow=26):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow