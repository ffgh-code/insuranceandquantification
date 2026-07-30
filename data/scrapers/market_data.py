"""Market data downloader supporting Chinese (akshare) and US (yfinance) markets.

Auto-detects market by ticker format:
- sh/sz prefix -> akshare (Chinese markets)
- Otherwise -> yfinance (US markets)
- Falls back to synthetic data when API unavailable
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta
from typing import Literal, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CSI 300 + popular A-shares
DEFAULT_TICKERS = ["sh000300", "sh000016", "sh000688", "sz000001", "sh600519"]

# Map common names to akshare symbols
TICKER_ALIAS = {
    "000300": "sh000300", "hs300": "sh000300", "csi300": "sh000300",
    "000016": "sh000016", "sz50": "sh000016", "ss50": "sh000016",
    "000688": "sh000688", "kcb50": "sh000688",
    "600519": "sh600519", "kweichow": "sh600519", "maotai": "sh600519",
    "000001": "sz000001", "pingan": "sz000001",
    "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM",
}


class MarketDataFetcher:
    """Fetch market data from Chinese (akshare) or US (yfinance) sources."""

    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        # Resolve aliases
        raw = tickers or DEFAULT_TICKERS
        self.tickers = [TICKER_ALIAS.get(t.lower(), t) for t in raw]
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.start_date = start_date or (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    @staticmethod
    def _is_chinese(ticker: str) -> bool:
        return ticker.startswith("sh") or ticker.startswith("sz")

    def download_prices(self) -> dict[str, pd.DataFrame]:
        """Download OHLCV data. Auto-selects akshare or yfinance per ticker."""
        all_data = {}
        china_tickers = [t for t in self.tickers if self._is_chinese(t)]
        us_tickers = [t for t in self.tickers if not self._is_chinese(t)]

        if china_tickers:
            try:
                cn = self._download_china(china_tickers)
                all_data.update(cn)
            except Exception as e:
                logger.warning("China data failed: %s. Using synthetic.", e)

        if us_tickers:
            try:
                us = self._download_us(us_tickers)
                all_data.update(us)
            except Exception as e:
                logger.warning("US data failed: %s. Using synthetic.", e)

        # Fill missing with synthetic
        for t in self.tickers:
            if t not in all_data:
                synth = self._generate_synthetic_data([t])
                if synth:
                    all_data[t] = synth[t]

        return all_data

    def _download_china(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Download Chinese market data via akshare."""
        import akshare as ak
        result = {}
        for ticker in tickers:
            # Extract the numeric code
            code = ticker[2:]  # remove 'sh' or 'sz' prefix
            prefix = ticker[:2]
            symbol = f"{prefix}{code}"
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is not None and not df.empty:
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                # Filter by date range
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                if not df.empty:
                    result[ticker] = df
                    logger.info("Fetched %s: %d rows", ticker, len(df))
            time.sleep(0.3)
        return result

    def _download_us(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Download US market data via yfinance."""
        import yfinance as yf
        result = {}
        for ticker in tickers:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
            if not hist.empty:
                hist.columns = [c.lower() for c in hist.columns]
                result[ticker] = hist
                logger.info("Fetched %s: %d rows", ticker, len(hist))
            time.sleep(0.5)
        return result

    def _generate_synthetic_data(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Generate realistic synthetic OHLCV as fallback."""
        np.random.seed(42)
        n_days = 756
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="B")
        sigma_base = 0.20 / np.sqrt(252)
        result = {}
        for idx, ticker in enumerate(tickers):
            price = 100.0 + idx * 20
            prices_list = [price]
            vol = sigma_base * (1 + 0.5 * np.sin(idx))
            for i in range(1, n_days):
                vol *= np.exp(0.1 * (np.log(sigma_base) - np.log(vol)) + 0.2 * np.random.randn())
                vol = max(vol, sigma_base * 0.3)
                ret = 0.08 / 252 + vol * np.random.randn()
                price *= np.exp(ret)
                prices_list.append(price)
            prices_arr = np.array(prices_list)
            daily_vol = sigma_base * (0.5 + 0.5 * np.random.rand(n_days))
            opens = prices_arr * np.exp(0.1 * daily_vol * np.random.randn(n_days))
            df = pd.DataFrame({
                "open": opens,
                "high": np.maximum(opens, prices_arr) * np.exp(1.5 * daily_vol),
                "low": np.minimum(opens, prices_arr) * np.exp(-1.2 * daily_vol),
                "close": prices_arr,
                "volume": np.random.lognormal(15, 0.8, n_days),
            }, index=dates)
            df["high"] = df[["open", "close", "high"]].max(axis=1)
            df["low"] = df[["open", "close", "low"]].min(axis=1)
            result[ticker] = df
        return result

    @staticmethod
    def compute_returns(prices: pd.DataFrame, method: Literal["simple", "log"] = "log") -> pd.Series:
        if method == "log":
            return np.log(prices["close"] / prices["close"].shift(1))
        return prices["close"].pct_change()

    @staticmethod
    def compute_realized_volatility(returns: pd.Series, window: int = 21,
                                    annualize: bool = True, trading_days: int = 242) -> pd.Series:
        """Note: trading_days=242 for Chinese markets (vs 252 for US)."""
        vol = returns.rolling(window=window).std()
        if annualize:
            vol *= np.sqrt(trading_days)
        return vol
