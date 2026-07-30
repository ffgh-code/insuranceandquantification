"""Traditional lexicon-based sentiment analysis for financial text.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) and
TextBlob as baselines for comparison with LLM-based approaches.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TraditionalSentimentAnalyzer:
    """Lexicon-based financial sentiment analysis.

    Provides VADER and TextBlob scores as baselines. VADER is particularly
    suitable for social media and news headlines; TextBlob provides
    a complementary polarity score.
    """

    def __init__(self):
        self._vader = None
        self._textblob = None
        self._nltk_downloaded = False

    def _ensure_vader(self):
        """Lazy import VADER to avoid heavy dependencies at module load."""
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                logger.error(
                    "vaderSentiment not installed. Run: pip install vaderSentiment"
                )
                raise

    def _ensure_textblob(self):
        """Lazy import TextBlob, downloading NLTK corpora if needed."""
        if self._textblob is None:
            try:
                from textblob import TextBlob

                self._textblob = TextBlob
                if not self._nltk_downloaded:
                    self._download_nltk_data()
            except ImportError:
                logger.error("textblob not installed. Run: pip install textblob")
                raise

    @staticmethod
    def _download_nltk_data():
        """Download required NLTK corpora for TextBlob."""
        import nltk

        for resource in ["punkt_tab", "averaged_perceptron_tagger_eng", "brown"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

    def vader_score(self, text: str) -> dict:
        """Get VADER sentiment scores for a single text.

        Args:
            text: Input text string.

        Returns:
            Dict with keys: 'neg', 'neu', 'pos', 'compound' (-1 to +1).
        """
        self._ensure_vader()
        return self._vader.polarity_scores(text)

    def textblob_score(self, text: str) -> dict:
        """Get TextBlob sentiment scores for a single text.

        Args:
            text: Input text string.

        Returns:
            Dict with keys: 'polarity' (-1 to +1), 'subjectivity' (0 to 1).
        """
        self._ensure_textblob()
        blob = self._textblob(text)
        return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

    def analyze_headlines(self, headlines: list[str]) -> pd.DataFrame:
        """Analyze a list of headlines with both VADER and TextBlob.

        Args:
            headlines: List of headline strings.

        Returns:
            DataFrame with columns: headline, vader_compound, vader_pos,
            vader_neg, vader_neu, textblob_polarity, textblob_subjectivity.
        """
        records = []
        for h in headlines:
            vader = self.vader_score(h)
            tb = self.textblob_score(h)
            records.append(
                {
                    "headline": h,
                    "vader_compound": vader["compound"],
                    "vader_pos": vader["pos"],
                    "vader_neg": vader["neg"],
                    "vader_neu": vader["neu"],
                    "textblob_polarity": tb["polarity"],
                    "textblob_subjectivity": tb["subjectivity"],
                }
            )
        return pd.DataFrame(records)

    def aggregate_daily_sentiment(
        self, df: pd.DataFrame, date_col: str = "published_dt"
    ) -> pd.DataFrame:
        """Aggregate sentiment scores to daily frequency.

        Args:
            df: DataFrame with sentiment scores and date column.
            date_col: Column name for date.

        Returns:
            Daily aggregated sentiment DataFrame with mean compound score.
        """
        daily = df.copy()
        daily["date"] = pd.to_datetime(daily[date_col]).dt.date
        agg = daily.groupby("date").agg(
            avg_vader_compound=("vader_compound", "mean"),
            avg_textblob_polarity=("textblob_polarity", "mean"),
            headline_count=("headline", "count"),
            pos_ratio=("vader_compound", lambda x: (x > 0.05).mean()),
            neg_ratio=("vader_compound", lambda x: (x < -0.05).mean()),
        ).reset_index()
        agg["date"] = pd.to_datetime(agg["date"])
        return agg

    @staticmethod
    def sentiment_to_factor(sentiment_series: pd.Series, window: int = 5) -> pd.Series:
        """Convert raw sentiment scores to a smoothed factor signal.

        Args:
            sentiment_series: Daily sentiment scores.
            window: Rolling window for smoothing.

        Returns:
            Z-score normalized sentiment factor.
        """
        smoothed = sentiment_series.rolling(window=window, min_periods=1).mean()
        factor = (smoothed - smoothed.expanding().mean()) / smoothed.expanding().std()
        return factor.fillna(0)
