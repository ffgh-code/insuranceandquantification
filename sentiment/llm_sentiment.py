"""LLM-based sentiment analysis for financial news.

Uses OpenAI-compatible APIs to extract nuanced sentiment signals
from financial text, including directional sentiment, magnitude,
and topic classification.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SENTIMENT_SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. Analyze the sentiment of each news"
    " headline for its impact on financial markets.\\n\\n"
    "For each headline, return a JSON object with these fields:\\n"
    '- "score": float from -1.0 (extremely negative) to +1.0 (extremely positive), 0 = neutral\\n'
    '- "magnitude": float from 0.0 to 1.0 indicating the strength of the sentiment\\n'
    '- "direction": "bullish", "bearish", or "neutral" for market impact\\n'
    '- "topic": one of ["rates", "earnings", "economy", "geopolitics", "energy", "tech", "regulation", "consumer", "housing", "employment", "trade", "other"]\\n'
    '- "confidence": float from 0.0 to 1.0\\n\\n'
    "Respond with valid JSON only, no additional text."
)


class LLMSentimentAnalyzer:
    """Analyze financial news sentiment using LLMs.

    Supports OpenAI API and local models via OpenAI-compatible endpoints.
    Falls back to rule-based heuristics if API is unavailable, ensuring
    the pipeline runs without an API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self._client = None

    def _init_client(self):
        """Initialize OpenAI client (lazy)."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._client = OpenAI(**kwargs)
        except ImportError:
            logger.warning("openai package not installed. LLM sentiment will use fallback.")
            self._client = None

    def analyze_text(self, text: str) -> dict:
        """Analyze sentiment of a single financial text using LLM.

        Args:
            text: Financial news headline or text.

        Returns:
            Dict with keys: score, magnitude, direction, topic, confidence.
            If the API call fails, returns a neutral fallback.
        """
        result = self._call_llm(text)
        if result is None:
            return LLMSentimentAnalyzer._fallback_analysis(text)
        return result

    def _call_llm(self, text: str) -> Optional[dict]:
        """Call LLM API for sentiment analysis."""
        self._init_client()
        if self._client is None or not self.api_key:
            return None

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Analyze this financial headline: {text}",
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
            )
            content = response.choices[0].message.content
            return self._parse_json_response(content)
        except Exception as e:
            logger.warning("LLM API call failed: %s. Using fallback.", e)
            return None

    @staticmethod
    def _parse_json_response(content: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling formatting variations."""
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _fallback_analysis(text: str) -> dict:
        """Rule-based fallback when API is unavailable.

        Uses financial keyword lexicon for basic sentiment detection.
        This ensures the pipeline runs without any API key.
        """
        text_lower = text.lower()

        positive_words = {
            "surge": 0.4, "soar": 0.4, "rally": 0.35, "jump": 0.3, "rise": 0.2,
            "gain": 0.25, "up": 0.15, "bullish": 0.4, "outperform": 0.35,
            "beat": 0.3, "growth": 0.3, "expansion": 0.3, "recovery": 0.3,
            "boost": 0.25, "positive": 0.25, "strong": 0.2, "record": 0.35,
            "profit": 0.25, "upgrade": 0.3, "optimistic": 0.3, "momentum": 0.25,
            "rebound": 0.25, "breakthrough": 0.35, "exceed": 0.3, "improve": 0.2,
            "stable": 0.15, "confidence": 0.25, "opportunity": 0.2
        }
        negative_words = {
            "plunge": 0.4, "crash": 0.45, "slump": 0.35, "drop": 0.2, "fall": 0.2,
            "decline": 0.2, "loss": 0.3, "bearish": 0.4, "underperform": 0.35,
            "miss": 0.3, "recession": 0.4, "contraction": 0.3, "downgrade": 0.3,
            "negative": 0.25, "weak": 0.2, "slowdown": 0.3, "crisis": 0.45,
            "default": 0.4, "volatile": 0.15, "uncertainty": 0.2, "risk": 0.15,
            "layoff": 0.35, "cut": 0.15, "losses": 0.3, "debt": 0.2,
            "inflation": 0.2, "tariff": 0.25, "tension": 0.25, "pressure": 0.15,
            "worst": 0.35, "fear": 0.3
        }

        score = 0.0
        matched_words = []

        for word, weight in positive_words.items():
            if word in text_lower:
                score += weight
                matched_words.append(word)

        for word, weight in negative_words.items():
            if word in text_lower:
                score -= weight
                matched_words.append(word)

        score = max(-1.0, min(1.0, score))
        magnitude = min(1.0, len(matched_words) * 0.15 + abs(score) * 0.5)

        if score > 0.15:
            direction = "bullish"
        elif score < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "score": round(score, 3),
            "magnitude": round(magnitude, 3),
            "direction": direction,
            "topic": LLMSentimentAnalyzer._classify_topic(text_lower),
            "confidence": round(min(1.0, len(matched_words) * 0.1 + 0.2), 3),
        }

    @staticmethod
    def _classify_topic(text: str) -> str:
        """Classify headline into a financial topic."""
        topics = {
            "rates": ["fed", "rate", "interest rate", "fomc", "central bank", "monetary"],
            "earnings": ["earnings", "revenue", "profit", "quarterly", "fiscal", "eps"],
            "economy": ["gdp", "economy", "economic", "inflation", "consumer price", "pmi"],
            "geopolitics": ["tension", "war", "sanction", "trade", "tariff", "geopolitical"],
            "energy": ["oil", "gas", "energy", "crude", "petroleum", "renewable"],
            "tech": ["tech", "ai", "artificial intelligence", "chip", "software", "cloud", "data"],
            "regulation": ["regulation", "regulatory", "sec", "compliance", "policy"],
            "consumer": ["consumer", "retail", "spending", "demand", "sales"],
            "housing": ["housing", "mortgage", "real estate", "home", "property"],
            "employment": ["job", "employment", "unemployment", "payroll", "labor"],
        }
        text_lower = text.lower()
        best_topic = "other"
        best_matches = 0
        for topic, keywords in topics.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > best_matches:
                best_matches = matches
                best_topic = topic
        return best_topic

    def analyze_headlines(
        self, headlines: list[str], use_api: bool = True
    ) -> pd.DataFrame:
        """Analyze sentiment for a list of headlines.

        Args:
            headlines: List of headline strings.
            use_api: If True, tries the LLM API first; falls back to rule-based.

        Returns:
            DataFrame with columns: headline, llm_score, llm_magnitude,
            llm_direction, llm_topic, llm_confidence.
        """
        records = []
        for headline in headlines:
            if use_api:
                result = self.analyze_text(headline)
            else:
                result = LLMSentimentAnalyzer._fallback_analysis(headline)
            records.append(
                {
                    "headline": headline,
                    "llm_score": result["score"],
                    "llm_magnitude": result["magnitude"],
                    "llm_direction": result["direction"],
                    "llm_topic": result["topic"],
                    "llm_confidence": result["confidence"],
                }
            )
        return pd.DataFrame(records)

    def aggregate_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate LLM sentiment scores to daily frequency.

        Args:
            df: DataFrame with llm_score and datetime index or date column.

        Returns:
            Daily aggregated sentiment DataFrame.
        """
        daily = df.copy()
        if "published_dt" in daily.columns:
            daily["date"] = pd.to_datetime(daily["published_dt"]).dt.date
        elif "date" not in daily.columns:
            daily["date"] = pd.Timestamp.now().date()
        else:
            daily["date"] = pd.to_datetime(daily["date"]).dt.date

        agg = daily.groupby("date").agg(
            avg_llm_score=("llm_score", "mean"),
            avg_magnitude=("llm_magnitude", "mean"),
            avg_confidence=("llm_confidence", "mean"),
            headline_count=("headline", "count"),
            bullish_ratio=("llm_direction", lambda x: (x == "bullish").mean()),
            bearish_ratio=("llm_direction", lambda x: (x == "bearish").mean()),
            neutral_ratio=("llm_direction", lambda x: (x == "neutral").mean()),
        ).reset_index()
        agg["date"] = pd.to_datetime(agg["date"])
        return agg

