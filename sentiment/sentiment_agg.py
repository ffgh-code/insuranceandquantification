"""Upgraded sentiment analysis: weighted aggregation, Granger causality, topic GARCH-X.

Extends the base LLM/VADER sentiment module with:
- Source-weighted aggregation (regulatory > sector news > market flash)
- Positive/negative separate time series
- Granger causality tests between sentiment and volatility
- 4 refined topic categories for GARCH-X exogenous variables
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Source weights for weighted aggregation
SOURCE_WEIGHTS = {
    "regulatory": 1.0,   # 监管公告
    "sector": 0.7,       # 行业头条
    "market": 0.4,       # 市场快讯
}

# Topic mapping: refined into 4 categories
TOPIC_CATEGORIES = {
    "monetary": ["rates", "fed", "rate", "interest rate", "fomc", "central bank", "monetary", "liquidity", "降准", "利率", "央行", "货币"],
    "industrial": ["earnings", "revenue", "profit", "tech", "ai", "chip", "energy", "新能源", "半导体", "人工智能", "产业"],
    "macro": ["economy", "gdp", "employment", "job", "consumer", "inflation", "trade", "gdp", "消费", "就业", "通胀", "贸易"],
    "geopolitical": ["geopolitics", "tension", "war", "sanction", "tariff", "地缘", "制裁", "关税", "局势"],
}


class SentimentAggregator:
    """Sentiment aggregation with source weighting and topic refinement."""

    def __init__(self):
        self._pos_series: Optional[pd.Series] = None
        self._neg_series: Optional[pd.Series] = None

    @staticmethod
    def classify_source(headline: str) -> str:
        """Classify headline source type."""
        h = headline.lower()
        if any(w in h for w in ["announce", "公告", "发布", "notice", "regulation", "规则", "监管"]):
            return "regulatory"
        elif any(w in h for w in ["sector", "industry", "行业", "板块", "龙头", "企业"]):
            return "sector"
        else:
            return "market"

    @staticmethod
    def refine_topic(topic: str) -> str:
        """Map detailed topic to 4 refined categories."""
        for cat, keywords in TOPIC_CATEGORIES.items():
            if topic.lower() in keywords or any(kw == topic.lower() for kw in keywords):
                return cat
        return "macro"  # default

    def aggregate_weighted(self, df: pd.DataFrame,
                           headline_col: str = "headline",
                           score_col: str = "llm_score",
                           source_col: str = None) -> pd.DataFrame:
        """Weighted aggregation by source type.

        Args:
            df: DataFrame with headlines and scores.
            headline_col: Column name for headline text.
            score_col: Column name for sentiment score.
            source_col: Optional pre-classified source column.
        """
        daily = df.copy()
        if "date" not in daily.columns:
            daily["date"] = pd.Timestamp.now().date()
        daily["date"] = pd.to_datetime(daily["date"]).dt.date

        # Classify source
        if source_col is None:
            daily["source_type"] = daily[headline_col].apply(self.classify_source)
        else:
            daily["source_type"] = daily[source_col]

        daily["weight"] = daily["source_type"].map(SOURCE_WEIGHTS).fillna(0.5)

        # Weighted score
        daily["weighted_score"] = daily[score_col] * daily["weight"]

        # Split positive/negative
        daily["pos_score"] = daily["weighted_score"].clip(lower=0)
        daily["neg_score"] = daily["weighted_score"].clip(upper=0).abs()

        # Aggregate daily
        agg = daily.groupby("date").agg(
            pos_sentiment=("pos_score", "mean"),
            neg_sentiment=("neg_score", "mean"),
            headline_count=("headline", "count"),
        ).reset_index()
        agg["date"] = pd.to_datetime(agg["date"])
        agg["net_sentiment"] = agg["pos_sentiment"] - agg["neg_sentiment"]

        self._pos_series = agg.set_index("date")["pos_sentiment"]
        self._neg_series = agg.set_index("date")["neg_sentiment"]
        return agg

    def get_series(self, sentiment_type: str = "net") -> pd.Series:
        """Get aggregated sentiment time series."""
        if sentiment_type == "pos":
            return self._pos_series or pd.Series(dtype=float)
        elif sentiment_type == "neg":
            return self._neg_series or pd.Series(dtype=float)
        else:
            pos = self._pos_series or pd.Series(0, index=pd.DatetimeIndex([]))
            neg = self._neg_series or pd.Series(0, index=pd.DatetimeIndex([]))
            return pos - neg

    @staticmethod
    def granger_causality(sentiment: pd.Series, target: pd.Series,
                          max_lag: int = 5) -> dict:
        """Granger causality test: does sentiment Granger-cause volatility?

        Returns F-statistic and p-value for each lag direction.
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        aligned = pd.concat([sentiment, target], axis=1).dropna()
        if len(aligned) < max_lag + 10:
            return {"error": "insufficient data"}

        # Test: sentiment -> target
        try:
            result_s2t = grangercausalitytests(aligned, maxlag=max_lag, verbose=False)
            s2t_stats = {lag: {
                "f_stat": round(result_s2t[lag][0]["ssr_ftest"][0], 4),
                "p_value": round(result_s2t[lag][0]["ssr_ftest"][1], 4),
            } for lag in range(1, max_lag + 1)}

            # Test: target -> sentiment (reverse)
            result_t2s = grangercausalitytests(
                aligned[[aligned.columns[1], aligned.columns[0]]], maxlag=max_lag, verbose=False)
            t2s_stats = {lag: {
                "f_stat": round(result_t2s[lag][0]["ssr_ftest"][0], 4),
                "p_value": round(result_t2s[lag][0]["ssr_ftest"][1], 4),
            } for lag in range(1, max_lag + 1)}

            return {
                "sentiment_to_target": s2t_stats,
                "target_to_sentiment": t2s_stats,
                "best_lag_s2t": min(s2t_stats, key=lambda l: s2t_stats[l]["p_value"]),
                "best_lag_t2s": min(t2s_stats, key=lambda l: t2s_stats[l]["p_value"]),
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def topic_sentiment_for_garchx(df: pd.DataFrame,
                                   topic_category: str,
                                   score_col: str = "llm_score",
                                   topic_col: str = "llm_topic") -> pd.Series:
        """Extract sentiment series for a single refined topic category.

        For use as exogenous variable in GARCH-X models.
        """
        refined = df[topic_col].apply(
            lambda t: SentimentAggregator.refine_topic(t))
        topic_df = df[refined == topic_category].copy()
        if topic_df.empty:
            return pd.Series(dtype=float)
        topic_df["date"] = pd.to_datetime(topic_df.get("published_dt", pd.Timestamp.now()))
        return topic_df.groupby("date")[score_col].mean()
