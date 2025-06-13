# Revised executor.py
import MetaTrader5 as mt5
from datetime import datetime, time as dt_time
import os
import json
import logging
import time

LOG_PATH = "data/trade_log.json"
DAILY_MAX_TRADES = 200000
DAILY_MAX_LOSS = -9999999
MIN_EQUITY = 1500
MAX_CONNECTION_ATTEMPTS = 15
CONNECTION_RETRY_DELAY = 5

class MT5Executor:
    def __init__(self, login, password, server):
        self.login = login
        self.password = password
        self.server = server
        self.today = datetime.now().date()
        self._ensure_log()
        self.connected = False
        self.connect()

    def connect(self):
        attempts = 0
        while attempts < MAX_CONNECTION_ATTEMPTS and not self.connected:
            try:
                if not mt5.initialize():
                    raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
                if not mt5.login(self.login, self.password, self.server):
                    raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")
                self.connected = True
                logging.info("Connected to MT5")
            except Exception as e:
                attempts += 1
                logging.error(f"Connection attempt {attempts} failed: {e}")
                time.sleep(CONNECTION_RETRY_DELAY)
        if not self.connected:
            logging.critical("Failed to connect to MT5")

    def close(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logging.info("Disconnected from MT5")

    def _ensure_log(self):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w") as f:
                json.dump({}, f)

    def _load_log(self):
        try:
            with open(LOG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load trade log: {e}")
            return {}

    def _save_log(self, log):
        try:
            with open(LOG_PATH, "w") as f:
                json.dump(log, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save trade log: {e}")

    def get_position(self, symbol: str):
        if not self.connected:
            self.connect()
        positions = mt5.positions_get(symbol=symbol)
        return positions[0] if positions else None

    def _get_today_pnl(self):
        if not self.connected:
            return 0.0
        start_time = datetime.combine(self.today, dt_time.min)
        end_time = datetime.now()
        deals = mt5.history_deals_get(start_time, end_time)
        if deals is None:
            return 0.0
        return sum(deal.profit for deal in deals if deal.entry in (mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT))

    def _check_risk(self):
        log = self._load_log()
        today_str = str(self.today)
        today_log = log.get(today_str, {"trades": 0, "pnl": 0.0})
        today_log["pnl"] = self._get_today_pnl()

        if today_log["trades"] >= DAILY_MAX_TRADES:
            logging.warning("Max trades reached")
            return False
        if today_log["pnl"] <= DAILY_MAX_LOSS:
            logging.warning("Max daily loss reached")
            return False

        log[today_str] = today_log
        self._save_log(log)
        return True

    def get_pip_multiplier(self, symbol: str) -> float:
        """Return pip multiplier for SL/TP based on symbol type"""
        if symbol.startswith("XAUUSDm"):      # Gold (e.g., XAUUSD)
            return 0.0007
        elif symbol.startswith("XAG"):    # Silver
            return 0.001
        elif symbol.endswith("USD") or "/" in symbol:  # Forex
            return 0.0001
        elif symbol.startswith("US30m") or symbol.startswith("USTECm") or symbol.endswith("Index"):  # Indices
            return 0.005
        elif "BTCUSDm" in symbol or "ETH" in symbol:  # Crypto
            return 0.04  # Adjust as needed
        else:
            return 0.00001  # Default for unknown (use symbol_info.point as fallback)

    # ADDED MISSING METHODS BELOW
    def update_sl(self, ticket: int, new_sl: float):
        """Update stop loss for an existing position"""
        if not self.connected:
            self.connect()
            if not self.connected:
                raise ConnectionError("Not connected to MT5")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logging.error(f"Position with ticket {ticket} not found")
            return False

        position = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": position.tp,
            "symbol": position.symbol,
            "deviation": 20,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to update SL for {ticket}: {result.comment}")
            return False
        logging.info(f"Updated SL for {ticket} to {new_sl}")
        return True

    def close_position(self, ticket: int):
        """Close an existing position by ticket"""
        if not self.connected:
            self.connect()
            if not self.connected:
                raise ConnectionError("Not connected to MT5")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logging.error(f"Position with ticket {ticket} not found")
            return False

        position = position[0]
        symbol = position.symbol
        volume = position.volume
        order_type = (
            mt5.ORDER_TYPE_BUY 
            if position.type == mt5.ORDER_TYPE_SELL 
            else mt5.ORDER_TYPE_SELL
        )

        price = (
            mt5.symbol_info_tick(symbol).ask 
            if order_type == mt5.ORDER_TYPE_SELL 
            else mt5.symbol_info_tick(symbol).bid
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 10042025,
            "comment": "Closed by AI_EA",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to close position {ticket}: {result.comment}")
            return False
        logging.info(f"Closed position {ticket} successfully")
        return True
    # END OF ADDED METHODS

    def place_order(self, symbol: str, lot: float, order_type: str,
                    sl_pips: int = 10000, tp_pips: int = 40000):

        if not self.connected:
            self.connect()
            if not self.connected:
                raise ConnectionError("Not connected to MT5")

        account_info = mt5.account_info()
        if account_info is None or account_info.equity < MIN_EQUITY:
            raise RuntimeError("Account equity too low or info unavailable")

        if not self._check_risk():
            raise RuntimeError("Risk limits exceeded")

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Failed to select {symbol}")

        digits = symbol_info.digits
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to get tick for {symbol}")

        pip_value = self.get_pip_multiplier(symbol)
        sl_pips_value = sl_pips * pip_value
        tp_pips_value = tp_pips * pip_value

        if order_type.lower() == "buy":
            price = round(tick.ask, digits)
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            sl = round(price - sl_pips_value, digits) if sl_pips > 0 else 0.0
            tp = round(price + tp_pips_value, digits) if tp_pips > 0 else 0.0
        elif order_type.lower() == "sell":
            price = round(tick.bid, digits)
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            sl = round(price + sl_pips_value, digits) if sl_pips > 0 else 0.0
            tp = round(price - tp_pips_value, digits) if tp_pips > 0 else 0.0
        else:
            raise ValueError(f"Invalid order type: {order_type}")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type_mt5,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 10042025,
            "comment": f"AI_EA_{order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            if result and result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                logging.warning("Requote - retrying order...")
                time.sleep(1)
                return self.place_order(symbol, lot, order_type, sl_pips, tp_pips)
            error = result.comment if result else "No response from order_send"
            raise RuntimeError(f"Order failed: {error}")

        logging.info(f"Order success: {symbol} {order_type} {lot} lots - "
                     f"SL: {sl:.5f} | TP: {tp:.5f} | Ticket: {result.order}")

        # Update log
        log = self._load_log()
        today_str = str(self.today)
        today_log = log.get(today_str, {"trades": 0, "pnl": 0.0})
        today_log["trades"] += 1
        log[today_str] = today_log
        self._save_log(log)

        return result