"""Backtesting engine for volatility-based trading strategies.

Provides a clean framework for strategy backtesting with proper
handling of transaction costs, risk metrics, and performance reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade record."""

    date: pd.Timestamp
    direction: str  # "long" or "short"
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    pnl: Optional[float] = None
    return_pct: Optional[float] = None


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio performance metrics."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_return_pct: float = 0.0
    daily_returns: pd.Series = field(default_factory=pd.Series)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: list = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a display DataFrame."""
        data = {
            "Metric": [
                "Total Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Max Drawdown",
                "Win Rate",
                "Total Trades",
                "Profitable Trades",
                "Avg Win",
                "Avg Loss",
                "Profit Factor",
            ],
            "Value": [
                f"{self.total_return_pct:.2f}%",
                f"{self.annualized_return:.2%}",
                f"{self.annualized_volatility:.2%}",
                f"{self.sharpe_ratio:.2f}",
                f"{self.sortino_ratio:.2f}",
                f"{self.calmar_ratio:.2f}",
                f"{self.max_drawdown:.2%}",
                f"{self.win_rate:.1%}",
                str(self.total_trades),
                str(self.profitable_trades),
                f"{self.avg_win:.2%}",
                f"{self.avg_loss:.2%}",
                f"{self.profit_factor:.2f}",
            ],
        }
        return pd.DataFrame(data)


class BacktestEngine:
    """Backtesting engine for financial trading strategies.

    Handles position sizing, trade execution, transaction costs,
    and performance computation.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.trades: list[Trade] = []
        self.metrics: Optional[PortfolioMetrics] = None

    def run(
        self,
        signals: pd.Series,
        prices: pd.Series,
        position_size_pct: float = 0.25,
    ) -> PortfolioMetrics:
        """Run backtest on a signal series.

        Args:
            signals: Trading signals (-1 to +1, no units).
                Positive = long, Negative = short, 0 = flat.
            prices: Asset price series aligned with signals.
            position_size_pct: Fraction of capital allocated per signal.

        Returns:
            PortfolioMetrics with full performance breakdown.
        """
        self.trades = []
        aligned = pd.concat(
            [signals.rename("signal"), prices.rename("price")], axis=1
        ).dropna()

        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        current_trade: Optional[Trade] = None
        equity = [capital]
        dates = [aligned.index[0]]

        for date, row in aligned.iterrows():
            signal = row["signal"]
            price = row["price"]

            # Determine target position
            if abs(signal) > 0.1:
                target_position = np.sign(signal) * position_size_pct * capital / price
            else:
                target_position = 0.0

            # Close existing trade if signal changes direction
            if current_trade is not None and (
                (current_trade.direction == "long" and signal <= 0)
                or (current_trade.direction == "short" and signal >= 0)
            ):
                # Close trade
                exit_cost = (
                    abs(position) * price * self.transaction_cost
                    + abs(position) * price * self.slippage
                )
                trade_pnl = position * (price - entry_price) - exit_cost
                capital += position * price + trade_pnl
                current_trade.exit_price = price
                current_trade.exit_date = date
                current_trade.pnl = trade_pnl
                current_trade.return_pct = trade_pnl / (abs(position) * entry_price + 1e-8)
                self.trades.append(current_trade)
                current_trade = None
                position = 0.0

            # Open new trade
            if abs(signal) > 0.1 and current_trade is None:
                direction = "long" if signal > 0 else "short"
                entry_cost = (
                    abs(target_position) * price * self.transaction_cost
                    + abs(target_position) * price * self.slippage
                )
                capital -= entry_cost
                position = target_position
                entry_price = price
                current_trade = Trade(
                    date=date,
                    direction=direction,
                    size=abs(position),
                    entry_price=price,
                )
            elif current_trade is not None:
                # Mark-to-market P&L for equity curve
                unrealized_pnl = position * (price - entry_price)
                current_equity = capital + position * price
                equity.append(current_equity)
                dates.append(date)
                continue

            current_equity = capital + (position * price if abs(position) > 0 else 0)
            equity.append(current_equity)
            dates.append(date)

        # Close any remaining position at last price
        if current_trade is not None:
            last_price = aligned.iloc[-1]["price"]
            close_cost = (
                abs(position) * last_price * self.transaction_cost
                + abs(position) * last_price * self.slippage
            )
            trade_pnl = position * (last_price - entry_price) - close_cost
            capital += position * last_price + trade_pnl
            current_trade.exit_price = last_price
            current_trade.exit_date = aligned.index[-1]
            current_trade.pnl = trade_pnl
            current_trade.return_pct = trade_pnl / (abs(position) * entry_price + 1e-8)
            self.trades.append(current_trade)

        # Compute metrics
        equity_curve = pd.Series(equity, index=dates)
        daily_returns = equity_curve.pct_change().dropna()

        if len(self.trades) > 0:
            trades_df = pd.DataFrame(
                [
                    {
                        "direction": t.direction,
                        "return": t.return_pct or 0.0,
                        "pnl": t.pnl or 0.0,
                    }
                    for t in self.trades
                ]
            )
            wins = trades_df[trades_df["return"] > 0]
            losses = trades_df[trades_df["return"] < 0]
            win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0
            avg_win = wins["return"].mean() if len(wins) > 0 else 0.0
            avg_loss = losses["return"].mean() if len(losses) > 0 else 0.0
            profit_factor = (
                wins["return"].sum() / abs(losses["return"].sum())
                if len(losses) > 0 and losses["return"].sum() != 0
                else float("inf") if len(wins) > 0 else 0.0
            )
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0

        total_return_pct = (capital - self.initial_capital) / self.initial_capital * 100
        trading_days = len(daily_returns)
        ann_factor = 252 / max(trading_days, 1)

        annualized_return = (
            (1 + daily_returns.mean()) ** 252 - 1 if trading_days > 0 else 0.0
        )
        annualized_vol = (
            daily_returns.std() * np.sqrt(252) if trading_days > 0 else 0.0
        )
        sharpe = (
            (daily_returns.mean() / daily_returns.std() * np.sqrt(252))
            if daily_returns.std() > 0
            else 0.0
        )

        # Sortino (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino = (
            (daily_returns.mean() * 252) / downside_std if downside_std > 0 else 0.0
        )

        # Max drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min()
        calmar = -annualized_return / max_dd if max_dd < 0 else 0.0

        self.metrics = PortfolioMetrics(
            total_return=capital - self.initial_capital,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(self.trades),
            profitable_trades=len([t for t in self.trades if t.pnl and t.pnl > 0]),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_return_pct=total_return_pct,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            trades=self.trades,
        )
        return self.metrics

    def plot_results(self):
        """Generate equity curve and drawdown plots."""
        import matplotlib.pyplot as plt

        if self.metrics is None:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Equity curve
        axes[0].plot(self.metrics.equity_curve.index, self.metrics.equity_curve, "b-", lw=1.5)
        axes[0].axhline(y=self.initial_capital, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title("Equity Curve")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        cummax = self.metrics.equity_curve.cummax()
        drawdown = (self.metrics.equity_curve - cummax) / cummax
        axes[1].fill_between(drawdown.index, 0, drawdown * 100, color="red", alpha=0.3)
        axes[1].set_title("Drawdown (%)")
        axes[1].set_ylabel("Drawdown %")
        axes[1].grid(True, alpha=0.3)

        # Daily returns
        axes[2].bar(
            self.metrics.daily_returns.index,
            self.metrics.daily_returns * 100,
            color="steelblue",
            alpha=0.6,
            width=1,
        )
        axes[2].set_title("Daily Returns (%)")
        axes[2].set_ylabel("Return %")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
