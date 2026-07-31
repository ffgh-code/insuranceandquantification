"""Local sentiment fallback: CSV cache + optional Qwen local model.

双方案设计：
  方案A（轻量化）：读取本地 sentiment_cache/local_sentiment.csv 兜底，
                    API 作为可选分支，不依赖外网即可完整复现。
  方案B（进阶版）：接入 Qwen 开源轻量化模型（qwen2.5-0.5b-instruct），
                    完全脱离外网接口，本地推理。

PS 迭代描述：
  "在模型迭代中，我将情绪分析引擎从单一云端 API 依赖重构为本地可复现的
   双通道架构：存量情绪 CSV 缓存保证离线可复现，可选接入 Qwen 轻量模型
   实现无外网环境下的端到端推理。"
"""

from __future__ import annotations
import logging
import os
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "sentiment_cache", "local_sentiment.csv"
)


class LocalSentimentEngine:
    """本地情绪引擎：CSV 兜底 + Qwen 本地模型。"""

    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path or CACHE_PATH
        self._cache_df: Optional[pd.DataFrame] = None
        self._qwen_model = None
        self._qwen_tokenizer = None

    # ─── 方案A：CSV 存量数据兜底 ───────────────────────────────

    def load_cache(self) -> pd.DataFrame:
        """读取本地情绪 CSV 缓存。"""
        if self._cache_df is not None:
            return self._cache_df
        if os.path.exists(self.cache_path):
            self._cache_df = pd.read_csv(self.cache_path)
            logger.info("Loaded %d cached sentiment records", len(self._cache_df))
        else:
            logger.warning("Cache file not found: %s", self.cache_path)
            self._cache_df = pd.DataFrame(
                columns=["headline", "date", "score", "direction", "topic", "source_weight"]
            )
        return self._cache_df

    def lookup(self, headline: str) -> Optional[dict]:
        """从 CSV 缓存中匹配头条情绪。"""
        df = self.load_cache()
        if df.empty:
            return None
        match = df[df["headline"] == headline]
        if match.empty:
            # 模糊匹配：包含关键词
            for _, row in df.iterrows():
                if str(row["headline"])[:8] in headline:
                    return row.to_dict()
            return None
        return match.iloc[0].to_dict()

    def get_daily_series(self) -> pd.Series:
        """按日期聚合的本地情绪序列（用于 GARCH-X）。"""
        df = self.load_cache()
        if df.empty:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date")["score"].mean()

    # ─── 方案B：Qwen 本地模型 ──────────────────────────────────

    def load_qwen(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """加载 Qwen 轻量模型（需 transformers + torch）。

        安装：pip install transformers torch accelerate
        首次运行自动下载模型（约1GB），之后可离线推理。
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading Qwen model: %s", model_name)
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            return True
        except Exception as e:
            logger.warning("Qwen load failed: %s", e)
            return False

    def qwen_analyze(self, headline: str) -> dict:
        """使用 Qwen 本地模型分析单条新闻情绪。"""
        if self._qwen_model is None:
            if not self.load_qwen():
                return {"score": 0.0, "direction": "neutral",
                        "topic": "macro", "confidence": 0.0}

        prompt = (
            "请判断以下财经新闻对A股市场的情绪影响，"
            "输出JSON格式：{\"score\": -1到1, \"direction\": \"bullish/bearish/neutral\", "
            "\"topic\": \"monetary/industrial/macro/geopolitical\"}\n新闻："
            + headline
        )
        try:
            import json
            import torch
            inputs = self._qwen_tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self._qwen_model.generate(
                    **inputs, max_new_tokens=80, do_sample=False
                )
            text = self._qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取 JSON 部分
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                return result
        except Exception as e:
            logger.warning("Qwen inference failed: %s", e)
        return {"score": 0.0, "direction": "neutral", "topic": "macro", "confidence": 0.0}

    def analyze_headlines_local(self, headlines: list[str], use_qwen: bool = False) -> pd.DataFrame:
        """批量分析头条：CSV 优先，未命中时 Qwen 或规则回退。"""
        records = []
        for h in headlines:
            cached = self.lookup(h)
            if cached is not None:
                records.append({
                    "headline": h,
                    "llm_score": float(cached["score"]),
                    "llm_direction": cached["direction"],
                    "llm_topic": cached["topic"],
                    "source": "csv_cache",
                })
            elif use_qwen:
                result = self.qwen_analyze(h)
                records.append({
                    "headline": h,
                    "llm_score": float(result.get("score", 0)),
                    "llm_direction": result.get("direction", "neutral"),
                    "llm_topic": result.get("topic", "macro"),
                    "source": "qwen_local",
                })
            else:
                # 规则回退（内置金融词典）
                from sentiment.llm_sentiment import LLMSentimentAnalyzer
                analyzer = LLMSentimentAnalyzer()
                result = analyzer._fallback_analysis(h)
                records.append({
                    "headline": h,
                    "llm_score": result["score"],
                    "llm_direction": result["direction"],
                    "llm_topic": result["topic"],
                    "source": "rule_fallback",
                })
        return pd.DataFrame(records)
