"""Enhanced backtest engine with rolling window, market regime, and cross-sectional factors."""

from __future__ import annotations
import logging
from typing import Optional, Literal
from datetime import timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RollingWindowBacktest:
    """Rolling window backtest with 7:3 train/test split.

    Wraps the existing BacktestEngine with rolling window logic:
    - Window: 240 trading days
    - Refit: monthly roll forward
    - Split: 7:3 train/test (168 days train, 72 days test per window)
    """

    def __init__(self, window_size: int = 240, step_size: int = 20,
                 train_ratio: float = 0.7, initial_capital: float = 1_000_000):
        self.window_size = window_size
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.initial_capital = initial_capital
        self.results: list[dict] = []

    def run(self, signals: pd.Series, prices: pd.Series,
            model_refit_fn=None) -> pd.DataFrame:
        """Run rolling window backtest.

        Args:
            signals: Full trading signal series.
            prices: Full price series.
            model_refit_fn: Optional function to refit model each window.

        Returns:
            DataFrame with per-window performance.
        """
        from .backtest import BacktestEngine
        aligned = pd.concat([signals.rename("signal"),
                             prices.rename("price")], axis=1).dropna()

        total_len = len(aligned)
        if total_len < self.window_size:
            logger.warning("Insufficient data for rolling window")
            return pd.DataFrame()

        n_windows = (total_len - self.window_size) // self.step_size
        if n_windows < 1:
            n_windows = 1

        window_results = []
        all_equity_curves = []

        for w in range(n_windows):
            start = w * self.step_size
            train_end = start + int(self.window_size * self.train_ratio)
            window_end = start + self.window_size

            if window_end > total_len:
                break

            # Train/val split
            train = aligned.iloc[start:train_end]
            test = aligned.iloc[train_end:window_end]

            if len(test) < 20:
                break

            # Refit model if callback provided
            if model_refit_fn is not None:
                try:
                    model_refit_fn(train)
                except Exception as e:
                    logger.warning("Refit failed at window %d: %s", w, e)

            # Backtest on test set
            engine = BacktestEngine(initial_capital=self.initial_capital)
            metrics = engine.run(test["signal"], test["price"])

            window_results.append({
                "window": w,
                "train_start": aligned.index[start],
                "train_end": aligned.index[train_end],
                "test_start": aligned.index[train_end],
                "test_end": aligned.index[window_end - 1],
                "sharpe": metrics.sharpe_ratio if metrics else 0,
                "return_pct": metrics.total_return_pct if metrics else 0,
                "max_dd": metrics.max_drawdown if metrics else 0,
                "win_rate": metrics.win_rate if metrics else 0,
                "n_trades": metrics.total_trades if metrics else 0,
            })
            if metrics is not None and not metrics.equity_curve.empty:
                all_equity_curves.append(metrics.equity_curve)

        self.results = window_results
        result_df = pd.DataFrame(window_results)

        # Summary stats
        result_df.attrs["avg_sharpe"] = result_df["sharpe"].mean()
        result_df.attrs["std_sharpe"] = result_df["sharpe"].std()
        result_df.attrs["avg_return"] = result_df["return_pct"].mean()
        result_df.attrs["out_of_sample_sharpe"] = result_df["sharpe"].mean()
        return result_df

    @staticmethod
    def monthly_returns_distribution(equity_curve: pd.Series) -> pd.DataFrame:
        """Compute monthly return distribution for histogram output."""
        daily_returns = equity_curve.pct_change().dropna()
        monthly = daily_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1)
        return pd.DataFrame({
            "month": monthly.index,
            "monthly_return": monthly.values,
        }).dropna()

    @staticmethod
    def plot_monthly_returns(monthly_df: pd.DataFrame):
        """Placeholder for monthly return histogram (called from dashboard)."""
        pass


class CrossSectionalFactor:
    """Cross-sectional multi-factor selection framework for CSI 300 constituents."""

    def __init__(self, n_stocks: int = 50):
        self.n_stocks = n_stocks
        self.factor_data: Optional[pd.DataFrame] = None

    def load_constituents(self) -> list[str]:
        """Get CSI 300 constituent stock codes via akshare."""
        try:
            import akshare as ak
            df = ak.index_stock_cons(symbol="000300")
            return df["品种代码"].tolist()
        except Exception as e:
            logger.warning("Failed to load constituents: %s", e)
            return [f"{i:06d}" for i in range(600000, 600300)]

    def compute_factors(self, price_data: dict[str, pd.DataFrame],
                        sentiment: Optional[pd.Series] = None) -> pd.DataFrame:
        """Compute cross-sectional factors for each stock.

        Factors: volatility, momentum, sentiment, volume.
        Returns IC and IR evaluation.
        """
        records = []
        for code, df in price_data.items():
            if df.empty or "close" not in df.columns:
                continue
            ret = df["close"].pct_change()
            vol_20d = ret.rolling(20).std() * np.sqrt(242)
            mom_60d = df["close"].pct_change(60)
            mom_20d = df["close"].pct_change(20)
            avg_vol = df["volume"].rolling(20).mean()

            records.append({
                "stock": code,
                "vol_20d": float(vol_20d.iloc[-1]) if not vol_20d.empty else 0,
                "mom_60d": float(mom_60d.iloc[-1]) if not mom_60d.empty else 0,
                "mom_20d": float(mom_20d.iloc[-1]) if not mom_20d.empty else 0,
                "avg_volume": float(avg_vol.iloc[-1]) if not avg_vol.empty else 0,
                "close": float(df["close"].iloc[-1]) if not df.empty else 0,
            })

        factor_df = pd.DataFrame(records)
        if sentiment is not None:
            factor_df["sentiment"] = float(sentiment.iloc[-1]) if not sentiment.empty else 0

        self.factor_data = factor_df
        return factor_df

    @staticmethod
    def rank_ic(factor: pd.Series, forward_return: pd.Series) -> dict:
        """Information Coefficient: Spearman rank correlation between factor and forward return."""
        from scipy.stats import spearmanr
        valid = pd.concat([factor, forward_return], axis=1).dropna()
        if len(valid) < 10:
            return {"ic": 0, "ir": 0}
        ic, pval = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        # IR = mean(IC) / std(IC) over time - here use cross-sectional approximation
        ir = ic / (1 - ic**2)**0.5 * np.sqrt(len(valid)) if abs(ic) < 1 else 0
        return {"ic": round(float(ic), 4), "ir": round(float(ir), 4),
                "p_value": round(float(pval), 4)}

    def select_top_stocks(self, factor_df: pd.DataFrame,
                          factor_col: str = "mom_60d",
                          n_select: int = 20) -> list[str]:
        """Select top n stocks by a given factor."""
        sorted_df = factor_df.sort_values(factor_col, ascending=False)
        return sorted_df.head(n_select)["stock"].tolist()


class MarketRegimeClassifier:
    """Classify market into bull/bear/range-bound regimes."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def classify(self, price_series: pd.Series) -> pd.Series:
        """Classify each period into market regime.

        Returns:
            Series with values: 'bull', 'bear', 'range'
        """
        ret = price_series.pct_change(self.lookback)
        vol = price_series.pct_change().rolling(self.lookback).std() * np.sqrt(242)

        regime = pd.Series("range", index=price_series.index)

        # Bull: return > 8% annualized and vol moderate
        bull_mask = (ret > 0.08) & (vol < 0.35)
        regime[bull_mask] = "bull"

        # Bear: return < -5%
        bear_mask = ret < -0.05
        regime[bear_mask] = "bear"

        return regime

    def partition_results(self, signals: pd.Series, prices: pd.Series,
                          metrics_fn) -> dict:
        """Run backtest separately for each regime and return per-regime metrics."""
        from .backtest import BacktestEngine
        regime = self.classify(prices)

        results = {}
        for r in ["bull", "bear", "range"]:
            mask = regime == r
            if mask.sum() < 20:
                results[r] = {"error": "insufficient data"}
                continue
            engine = BacktestEngine()
            metrics = engine.run(signals[mask], prices[mask])
            if metrics:
                results[r] = {
                    "sharpe": metrics.sharpe_ratio,
                    "return": metrics.total_return_pct,
                    "max_dd": metrics.max_drawdown,
                    "win_rate": metrics.win_rate,
                    "n_days": int(mask.sum()),
                }
        return results
