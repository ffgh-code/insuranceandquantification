"""High-frequency market data and intraday realized volatility.

双分支设计:
  Branch A(本地环境):正常调用akshare获取沪深300 5min数据，自动保存为parquet缓存
  Branch B(离线/云端测试):检测缓存文件不存在时，生成模拟Demo数据保证整条代码不报错、可演示流程

【本地运行时删除模拟数据分支，自动读取真实akshare行情】
"""

from __future__ import annotations
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HighFreqData:
    """5分钟高频数据 + 日内已实现波动率计算。

    完整实现包含:
    - 5分钟对数收益率序列计算
    - 每日日内已实现波动率 RV_t = sum(r_{t,i}^2)
    - 年化处理(242个交易日 * 48个5分钟区间)
    - 与日度波动率对比可视化数据输出
    """

    # 参数常量
    TRADING_DAYS = 242  # A股年交易日数
    FIVE_MIN_PER_DAY = 48  # 每日5分钟K线数量(4小时 / 5分钟)
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

    def __init__(self, symbol: str = "000300", n_days: int = 756):
        self.symbol = symbol
        self.n_days = n_days
        self._minute_data: Optional[pd.DataFrame] = None
        self._intraday_rv: Optional[pd.Series] = None
        self._daily_counts: Optional[pd.Series] = None
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    #  Branch A:本地环境，正常调用akshare
    # ═══════════════════════════════════════════════════════════════

    def _fetch_from_akshare(self) -> Optional[pd.DataFrame]:
        """Branch A: 通过akshare获取沪深300真实5分钟数据。

        【本地运行时删除模拟数据分支，自动读取真实akshare行情】
        使用东方财富接口获取5分钟K线，包含OHLCV。
        """
        import akshare as ak
        try:
            # 获取沪深300指数5分钟数据
            # period="5" 表示5分钟线
            df = ak.index_zh_a_hist_min_em(symbol=self.symbol, period="5")
            if df is not None and not df.empty:
                # 统一列名
                df.columns = [c.lower() for c in df.columns]
                # 解析时间:东方财富返回格式 "2024-01-15 09:35"
                df["datetime"] = pd.to_datetime(df["time"])
                df["date"] = df["datetime"].dt.date
                logger.info("Branch A: 成功获取 %d 条5分钟数据", len(df))
                return df
        except Exception as e:
            logger.warning("Branch A akshare失败: %s，切换Branch B", e)
        return None

    # ═══════════════════════════════════════════════════════════════
    #  Branch B:离线/云端测试——生成模拟数据
    # ═══════════════════════════════════════════════════════════════

    def _generate_synthetic_minute(self) -> pd.DataFrame:
        """Branch B: 生成模拟5分钟数据用于演示。

        模拟逻辑:
        - 每日48根5分钟线，基于GBM生成
        - 日内波动率U型模式:开盘/收盘高，午盘低
        - 日间波动率符合随机波动率过程，模拟真实市场特征
        """
        np.random.seed(42)
        n_minutes_total = self.n_days * self.FIVE_MIN_PER_DAY
        records = []
        base_price = 4500.0

        # 日间波动率过程(随机波动率)
        daily_vol = np.random.lognormal(-1.6, 0.3, self.n_days)

        for day in range(self.n_days):
            current_date = pd.Timestamp.now().normalize() - timedelta(days=self.n_days - day)

            # 过滤周末(模拟交易日历)
            if current_date.weekday() >= 5:
                continue

            day_vol = daily_vol[day] * 0.002  # 当日波动率缩放

            # 日内U型波动率曲线:开盘和收盘波动较大，午盘较小
            intraday_pattern = np.array([
                1.3, 1.2, 1.1, 1.0, 0.9,  # 09:30-09:55
                0.8, 0.7, 0.7, 0.7, 0.7,  # 10:00-10:45
                0.6, 0.5, 0.5, 0.6, 0.7,  # 11:00-11:30 午休
                0.8, 0.9, 1.0, 1.1, 1.2,  # 13:00-13:45
                1.3, 1.3, 1.2, 1.1, 1.0,  # 14:00-14:45
                0.9, 0.8, 0.7, 0.6, 0.5,  # 14:50-15:00
            ])
            # 补齐48根线
            pattern = np.resize(intraday_pattern, self.FIVE_MIN_PER_DAY)

            # 模拟当日每分钟价格
            day_opens = []
            day_highs = []
            day_lows = []
            day_closes = []
            day_volumes = []
            t = 0

            for i in range(self.FIVE_MIN_PER_DAY):
                # 时间戳
                hour = 9 + (30 + i * 5) // 60
                minute = (30 + i * 5) % 60
                if hour >= 15:
                    break

                # U型波动率缩放
                vol_scale = pattern[t] if t < len(pattern) else 1.0

                # 生成OHLC
                if t == 0:
                    open_p = base_price * (1 + np.random.randn() * 0.002)
                else:
                    open_p = day_closes[-1] * (1 + np.random.randn() * 0.0005)

                ret = np.random.randn() * day_vol * vol_scale
                close_p = open_p * (1 + ret)
                high_p = max(open_p, close_p) * (1 + abs(np.random.randn()) * 0.5 * day_vol * vol_scale)
                low_p = min(open_p, close_p) * (1 - abs(np.random.randn()) * 0.5 * day_vol * vol_scale)
                vol_val = int(np.random.lognormal(13, 0.5))

                # 记录
                dt = current_date + timedelta(hours=hour, minutes=minute)
                records.append({
                    "datetime": dt,
                    "date": current_date.date(),
                    "open": round(open_p, 2),
                    "high": round(high_p, 2),
                    "low": round(low_p, 2),
                    "close": round(close_p, 2),
                    "volume": vol_val,
                })
                day_closes.append(close_p)
                t += 1

            # 日末更新base_price(用于下一天的开盘)
            if day_closes:
                base_price = day_closes[-1] * (1 + np.random.randn() * 0.002)

        df = pd.DataFrame(records)
        logger.info("Branch B: 生成 %d 条模拟5分钟数据", len(df))
        return df

    # ═══════════════════════════════════════════════════════════════
    #  缓存机制
    # ═══════════════════════════════════════════════════════════════

    def _cache_path(self) -> str:
        return os.path.join(self.CACHE_DIR, f"hf_{self.symbol}.parquet")

    def _load_cache(self) -> Optional[pd.DataFrame]:
        path = self._cache_path()
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                logger.info("从缓存加载 %d 条5分钟数据", len(df))
                return df
            except Exception:
                pass
        return None

    def _save_cache(self, df: pd.DataFrame):
        try:
            df.to_parquet(self._cache_path(), index=False)
            logger.info("5分钟数据已缓存至 %s", self._cache_path())
        except Exception as e:
            logger.warning("缓存写入失败: %s", e)

    # ═══════════════════════════════════════════════════════════════
    #  主入口:双分支自动选择
    # ═══════════════════════════════════════════════════════════════

    def fetch_minute_data(self, force_refetch: bool = False) -> pd.DataFrame:
        """主入口:自动选择数据来源。

        优先顺序:
        1. 缓存文件(parquet)
        2. Branch A(akshare在线获取)
        3. Branch B(模拟数据)
        """
        if self._minute_data is not None and not force_refetch:
            return self._minute_data

        # 尝试从缓存加载
        if not force_refetch:
            cached = self._load_cache()
            if cached is not None:
                self._minute_data = cached
                return cached

        # Branch A:akshare
        df = self._fetch_from_akshare()
        if df is not None:
            self._save_cache(df)
            self._minute_data = df
            return df

        # Branch B:模拟数据
        logger.info("Branch B: 生成模拟高频数据(仅演示用途)")
        df = self._generate_synthetic_minute()
        self._minute_data = df
        return df

    # ═══════════════════════════════════════════════════════════════
    #  日内已实现波动率 RV 计算(核心函数)
    # ═══════════════════════════════════════════════════════════════

    def compute_intraday_rv(self, minute_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """计算每日日内已实现波动率 RV_t。

        数学定义:
          RV_t = sqrt( sum_{i=1}^{N_t} r_{t,i}^2 * N_yearly / N_t )
        其中:
          r_{t,i} = log(P_{t,i} / P_{t,i-1})  第t日第i个5分钟对数收益率
          N_yearly = 242 * 48 = 11,616(年化5分钟区间数)

        Returns:
            Series indexed by date, annualized intraday RV.
        """
        data = minute_data if minute_data is not None else self.fetch_minute_data()
        if data is None or data.empty:
            return pd.Series(dtype=float)

        df = data.copy()
        # 按日期排序
        df = df.sort_values("datetime")

        # 计算5分钟对数收益率
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

        # 标记每个交易日内的5分钟区间序号
        df["date"] = pd.to_datetime(df["date"])

        # 日内RV:每日各5分钟收益率的平方和，再年化
        def _daily_rv(group):
            # 去掉第一个NaN(shift引入)
            returns = group["log_ret"].iloc[1:]
            if len(returns) < 5:  # 数据太少，不可靠
                return np.nan
            rv_daily = np.nansum(returns ** 2)
            # 年化:年化5分钟区间数 / 当日实际5分钟区间数
            n_obs = len(returns.dropna())
            if n_obs < 1:
                return np.nan
            scale_factor = (self.TRADING_DAYS * self.FIVE_MIN_PER_DAY) / n_obs
            return np.sqrt(rv_daily * scale_factor)

        daily_rv = df.groupby("date").apply(_daily_rv, include_groups=False)
        daily_rv = daily_rv.sort_index().dropna()
        # 每日有效5分钟K线数量(用于质量控制)
        daily_counts = df.groupby("date").apply(
            lambda g: len(g["log_ret"].iloc[1:].dropna()), include_groups=False)

        self._intraday_rv = daily_rv
        self._daily_counts = daily_counts
        return daily_rv

    # ═══════════════════════════════════════════════════════════════
    #  与日度波动率对比(绘图数据准备)
    # ═══════════════════════════════════════════════════════════════

    def get_rv_comparison(self, daily_vol_df: pd.DataFrame) -> pd.DataFrame:
        """合并日内RV与日度波动率估计量，用于对比可视化。

        返回DataFrame包含:
        - close_to_close(日度简单波动率)
        - parkinson(日度Parkinson极差估计)
        - garman_klass(日度GK估计)
        - yang_zhang(日度YZ估计，最鲁棒)
        - intraday_rv(日内已实现波动率，含48个5分钟区间信息)

        日内RV理论上应包含最丰富的信息，可作为"准真实值"对比日度估计量。
        """
        if self._intraday_rv is None:
            self.compute_intraday_rv()

        combined = daily_vol_df.copy()
        combined["intraday_rv"] = self._intraday_rv.reindex(combined.index)
        return combined

    @staticmethod
    def prepare_rv_chart_data(rv_comparison: pd.DataFrame,
                               plot_columns: list[str] = None) -> pd.DataFrame:
        """准备RV对比图表数据，支持选择性输出列。

        默认输出所有波动率估计量，便于前端Plotly/Matplotlib绘图。
        """
        if plot_columns is None:
            plot_columns = ["close_to_close", "parkinson",
                            "garman_klass", "yang_zhang", "intraday_rv"]
        available = [c for c in plot_columns if c in rv_comparison.columns]
        return rv_comparison[available].dropna(how="all").reset_index()
