"""backtest 包：回测引擎与对照指标（复用 strategy 模块）。"""
from .control_group_metrics import ControlGroupMetrics
from strategy.rolling_backtest import RollingWindowBacktest, MarketRegimeClassifier
from strategy.cross_sectional import CSI300Constituents, CrossSectionalFactors

__all__ = ["ControlGroupMetrics", "RollingWindowBacktest",
           "MarketRegimeClassifier", "CSI300Constituents", "CrossSectionalFactors"]
