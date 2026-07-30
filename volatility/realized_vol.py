"""Realized volatility estimation methods.

Provides multiple approaches for estimating realized volatility from
OHLCV data: close-to-close, Parkinson, Garman-Klass, Rogers-Satchell,
and Yang-Zhang estimators.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RealizedVolatility:
    """Realized volatility estimation from OHLCV data.

    Supports multiple estimators with increasing robustness to market
    microstructure noise and overnight jumps.
    """

    # Annual trading days convention
    TRADING_DAYS = 252

    @staticmethod
    def close_to_close(
        prices: pd.DataFrame,
        window: int = 21,
        annualize: bool = True,
    ) -> pd.Series:
        """Standard close-to-close realized volatility.

        The simplest estimator, using the standard deviation of log returns.

        Args:
            prices: DataFrame with 'close' column.
            window: Rolling window size in trading days.
            annualize: Whether to annualize.

        Returns:
            Series of annualized realized volatility estimates.
        """
        log_returns = np.log(prices["close"] / prices["close"].shift(1))
        vol = log_returns.rolling(window=window).std()
        if annualize:
            vol *= np.sqrt(RealizedVolatility.TRADING_DAYS)
        return vol

    @staticmethod
    def parkinson(prices: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
        """Parkinson's high-low volatility estimator.

        Uses daily high and low prices. More efficient than close-to-close
        as it captures intraday price range.

        Reference: Parkinson, M. (1980). The extreme value method for
        estimating the variance of the rate of return.
        """
        high = prices["high"]
        low = prices["low"]
        # Parkinson variance: (1 / (4 * log(2))) * (log(H/L))^2
        parkinson_var = (1.0 / (4.0 * np.log(2.0))) * (np.log(high / low)) ** 2
        vol = np.sqrt(parkinson_var.rolling(window=window).mean())
        if annualize:
            vol *= np.sqrt(RealizedVolatility.TRADING_DAYS)
        return vol

    @staticmethod
    def garman_klass(prices: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
        """Garman-Klass OHLC volatility estimator.

        Uses open, high, low, and close prices for greater efficiency.

        Reference: Garman, M. B. & Klass, M. J. (1980). On the estimation
        of security price volatilities from historical data.
        """
        high = prices["high"]
        low = prices["low"]
        opn = prices["open"]
        close = prices["close"]

        term1 = 0.5 * (np.log(high / low)) ** 2
        term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(close / opn)) ** 2
        gk_var = term1 - term2
        vol = np.sqrt(gk_var.rolling(window=window).mean())
        if annualize:
            vol *= np.sqrt(RealizedVolatility.TRADING_DAYS)
        return vol

    @staticmethod
    def yang_zhang(prices: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.Series:
        """Yang-Zhang volatility estimator.

        The most robust estimator that is independent of drift and
        handles both opening jumps and intraday volatility.

        Reference: Yang, D. & Zhang, Q. (2000). Drift-independent
        volatility estimation based on high, low, open, and close prices.
        """
        high = prices["high"]
        low = prices["low"]
        opn = prices["open"]
        close = prices["close"]

        # Overnight volatility (close-to-open)
        log_co = np.log(opn / close.shift(1))
        overnight_var = log_co.rolling(window=window).var()

        # Open-to-close volatility
        log_oc = np.log(close / opn)
        open_close_var = log_oc.rolling(window=window).var()

        # Rogers-Satchell component (high/low range)
        log_ho = np.log(high / opn)
        log_lo = np.log(low / opn)
        log_co_daily = np.log(close / opn)
        rs_var = (log_ho * log_co_daily + log_lo * log_co_daily).rolling(window=window).mean()

        # Yang-Zhang weighting
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
        vol = np.sqrt(yz_var)
        if annualize:
            vol *= np.sqrt(RealizedVolatility.TRADING_DAYS)
        return vol

    @staticmethod
    def compute_all(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """Compute all volatility estimators and return as a DataFrame.

        Args:
            prices: DataFrame with OHLC columns.
            window: Rolling window size.

        Returns:
            DataFrame with columns: close_to_close, parkinson,
            garman_klass, yang_zhang.
        """
        return pd.DataFrame(
            {
                "close_to_close": RealizedVolatility.close_to_close(prices, window),
                "parkinson": RealizedVolatility.parkinson(prices, window),
                "garman_klass": RealizedVolatility.garman_klass(prices, window),
                "yang_zhang": RealizedVolatility.yang_zhang(prices, window),
            }
        )

    @staticmethod
    def compute_forward_volatility(
        prices: pd.DataFrame, forward_window: int = 21
    ) -> pd.Series:
        """Compute forward (ex-post) realized volatility for prediction targets.

        Used as the target variable when training volatility prediction models.

        Args:
            prices: DataFrame with 'close' column.
            forward_window: Forward-looking window in trading days.

        Returns:
            Series of forward realized volatility (annualized).
        """
        log_returns = np.log(prices["close"] / prices["close"].shift(1))
        # Shift backward: today's target is vol over the next `forward_window` days
        fwd_vol = (
            log_returns.shift(-forward_window)
            .rolling(window=forward_window)
            .std()
            * np.sqrt(RealizedVolatility.TRADING_DAYS)
        )
        return fwd_vol
