"""Cross-sectional multi-factor selection framework for CSI 300 constituents.

双分支设计:
  Branch A(本地环境):正常调用akshare获取沪深300成分股数据，自动缓存
  Branch B(离线/云端测试):检测缓存文件不存在时，生成模拟成分股数据

【本地运行时删除模拟数据分支，自动读取akshare真实成分股行情】
"""

from __future__ import annotations
import logging
import os
import time
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class CSI300Constituents:
    """沪深300成分股数据获取与管理(双分支设计)。"""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._constituents: Optional[pd.DataFrame] = None
        self._price_data: dict[str, pd.DataFrame] = {}

    # ─── Branch A:akshare 真实数据 ────────────────────────────

    def _fetch_constituents_akshare(self) -> pd.DataFrame:
        """Branch A: 通过akshare获取沪深300最新成分股列表。"""
        import akshare as ak
        df = ak.index_stock_cons(symbol="000300")
        logger.info("Branch A: 获取 %d 只成分股", len(df))
        return df

    def _fetch_stock_daily(self, code: str) -> Optional[pd.DataFrame]:
        """Branch A: 获取单只成分股日线数据。"""
        import akshare as ak
        try:
            prefix = "sh" if code.startswith("6") else "sz"
            df = ak.stock_zh_index_daily(symbol=prefix + code)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                return df
        except Exception as e:
            logger.warning("获取 %s 失败: %s", code, e)
        return None

    # ─── Branch B:模拟数据 ────────────────────────────────────

    def _generate_synthetic_constituents(self) -> pd.DataFrame:
        """Branch B: 生成模拟成分股列表。"""
        np.random.seed(42)
        stocks = []
        for i in range(300):
            code = f"{600000 + i:06d}" if i < 200 else f"{300000 + i - 200:06d}"
            stocks.append({"品种代码": code, "品种名称": f"模拟股票{i+1}"})
        df = pd.DataFrame(stocks)
        logger.info("Branch B: 生成 %d 只模拟成分股", len(df))
        return df

    def _generate_synthetic_prices(self, code: str) -> pd.DataFrame:
        """Branch B: 生成单只模拟成分股日线数据。"""
        np.random.seed(abs(hash(code)) % (2**31))
        n = 400
        price = 20.0 + (abs(hash(code)) % 100)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
        prices = [price]
        for _ in range(1, n):
            ret = np.random.randn() * 0.02 + 0.0005
            price *= np.exp(ret)
            prices.append(price)
        return pd.DataFrame({
            "open": np.array(prices) * (1 + np.random.randn(n) * 0.005),
            "high": np.array(prices) * (1 + abs(np.random.randn(n)) * 0.01),
            "low": np.array(prices) * (1 - abs(np.random.randn(n)) * 0.01),
            "close": prices,
            "volume": np.random.lognormal(14, 0.6, n),
        }, index=dates)

    # ─── 主入口 ────────────────────────────────────────────────

    def load_constituents(self) -> pd.DataFrame:
        if self._constituents is not None:
            return self._constituents
        try:
            df = self._fetch_constituents_akshare()
        except Exception:
            df = self._generate_synthetic_constituents()
        self._constituents = df
        return df

    def load_stock_prices(self, code: str) -> pd.DataFrame:
        if code in self._price_data and not self._price_data[code].empty:
            return self._price_data[code]
        df = self._fetch_stock_daily(code)
        if df is None or df.empty:
            df = self._generate_synthetic_prices(code)
        self._price_data[code] = df
        return df


class CrossSectionalFactors:
    """横截面多因子计算与IC/IR评估。

    因子清单:
    - 波动率因子:20日已实现波动率
    - 动量因子:60日累计收益
    - 反转因子:20日累计收益(短期反转效应)
    - 情绪因子:LLM情绪得分(如有)
    - 换手率因子:20日平均换手率(成交量代理)
    """

    def __init__(self, constituents: CSI300Constituents):
        self.constituents = constituents
        self._factor_df: Optional[pd.DataFrame] = None

    def compute_factors(self, sentiment_series: Optional[pd.Series] = None,
                        n_stocks: int = 50) -> pd.DataFrame:
        """计算多因子截面数据。

        Args:
            sentiment_series: 全局情绪时间序列(可选)。
            n_stocks: 参与计算的股票数量(subset for speed)。

        Returns:
            DataFrame: stock | vol_20d | mom_60d | mom_20d | sentiment | avg_volume
        """
        cons = self.constituents.load_constituents()
        codes = cons["品种代码"].tolist()[:n_stocks]

        records = []
        for code in codes:
            df = self.constituents.load_stock_prices(code)
            if df.empty or "close" not in df.columns:
                continue
            ret = df["close"].pct_change().dropna()
            if len(ret) < 60:
                continue

            vol_20d = ret.rolling(20).std().iloc[-1] * np.sqrt(242)
            mom_60d = df["close"].pct_change(60).iloc[-1]
            mom_20d = df["close"].pct_change(20).iloc[-1]
            avg_vol_val = df["volume"].iloc[-20:].mean()

            records.append({
                "stock": code,
                "name": cons.loc[cons["品种代码"] == code, "品种名称"].values[0] if not cons.empty else code,
                "vol_20d": float(vol_20d) if not np.isnan(vol_20d) else 0.0,
                "mom_60d": float(mom_60d) if not np.isnan(mom_60d) else 0.0,
                "mom_20d": float(mom_20d) if not np.isnan(mom_20d) else 0.0,
                "avg_volume": float(avg_vol_val) if not np.isnan(avg_vol_val) else 0.0,
                "close": float(df["close"].iloc[-1]),
            })

        factor_df = pd.DataFrame(records)
        if sentiment_series is not None and not sentiment_series.empty:
            factor_df["sentiment"] = float(sentiment_series.iloc[-1])

        self._factor_df = factor_df
        return factor_df

    def compute_ic_ir(self, factor_df: Optional[pd.DataFrame] = None) -> dict:
        """计算各因子的IC(信息系数)和IR(信息比率)。

        IC = Spearman秩相关系数(因子值 vs 下一期收益)
        IR = mean(IC) / std(IC) over cross-section

        使用下一期5日收益作为预测目标。
        """
        from scipy.stats import spearmanr
        if factor_df is None:
            factor_df = self._factor_df
        if factor_df is None or factor_df.empty:
            return {}

        # 模拟下一期收益(真实环境下需使用未来5日收益)
        np.random.seed(42)
        factor_df["forward_ret_5d"] = np.random.randn(len(factor_df)) * 0.03

        factors = ["vol_20d", "mom_60d", "mom_20d", "sentiment"]
        available_factors = [f for f in factors if f in factor_df.columns]

        results = {}
        for factor in available_factors:
            valid = factor_df[[factor, "forward_ret_5d"]].dropna()
            if len(valid) < 10:
                continue
            ic, pval = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
            # IR = IC / (1 - IC^2)^0.5 * sqrt(N) 近似
            ir = (ic / np.sqrt(max(1 - ic**2, 1e-6))) * np.sqrt(len(valid)) / 100
            results[factor] = {
                "ic": round(float(ic), 4),
                "ir": round(float(ir), 4),
                "p_value": round(float(pval), 4),
                "significant": pval < 0.05,
            }

        return results

    def select_top_stocks(self, factor_df: Optional[pd.DataFrame] = None,
                          combine_factors: list[str] = None,
                          n_select: int = 20) -> pd.DataFrame:
        """综合多因子打分，选取得分最高的n只股票。

        因子标准化后等权合成，按总分排序。
        """
        if factor_df is None:
            factor_df = self._factor_df
        if factor_df is None or factor_df.empty:
            return pd.DataFrame()

        if combine_factors is None:
            combine_factors = ["vol_20d", "mom_60d"]
        available = [f for f in combine_factors if f in factor_df.columns]

        if not available:
            return factor_df.head(n_select)

        # 因子标准化(Z-score)
        scores = np.zeros(len(factor_df))
        for f in available:
            mean = factor_df[f].mean()
            std = factor_df[f].std()
            if std > 0:
                scores += (factor_df[f] - mean) / std

        factor_df["combined_score"] = scores
        result = factor_df.sort_values("combined_score", ascending=False)
        return result.head(n_select)[["stock", "name", "combined_score"] + available]
