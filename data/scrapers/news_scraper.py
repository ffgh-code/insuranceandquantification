"""Financial news scraper for sentiment analysis.

Aggregates headlines from RSS feeds and news APIs to build a
corpus for sentiment-based volatility analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Financial news RSS feeds
FINANCIAL_RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
}


class NewsScraper:
    """Scrape financial news headlines for sentiment analysis."""

    def __init__(self, cache_days: int = 30):
        self.cache_days = cache_days

    def fetch_rss_headlines(
        self, feed_url: str, max_items: int = 50
    ) -> list[dict]:
        """Fetch headlines from an RSS feed.

        Args:
            feed_url: RSS feed URL.
            max_items: Maximum number of items to fetch.

        Returns:
            List of dicts with 'title', 'link', 'published', 'source' keys.
        """
        headlines = []
        try:
            # Try feedparser first (better RSS support)
            import feedparser

            feed = feedparser.parse(feed_url)
            source = feed_url.split("//")[1].split(".")[0] if "//" in feed_url else "unknown"
            for entry in feed.entries[:max_items]:
                headlines.append(
                    {
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": source,
                    }
                )
        except (ImportError, Exception):
            # Fallback: manual RSS XML parsing
            logger.info("Falling back to manual RSS parsing for %s", feed_url)
            headlines = self._parse_rss_manually(feed_url, max_items)
        return headlines

    def _parse_rss_manually(self, url: str, max_items: int) -> list[dict]:
        """Manually parse RSS XML using requests + BeautifulSoup."""
        headlines = []
        try:
            resp = requests.get(url, timeout=10, headers=self._headers())
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml-xml")
            source = url.split("//")[1].split(".")[0] if "//" in url else "unknown"
            items = soup.find_all("item")[:max_items]
            for item in items:
                title = item.find("title")
                link = item.find("link")
                pub_date = item.find("pubDate")
                if title:
                    headlines.append(
                        {
                            "title": title.text.strip(),
                            "link": link.text.strip() if link else "",
                            "published": pub_date.text.strip() if pub_date else "",
                            "source": source,
                        }
                    )
        except Exception as e:
            logger.warning("Failed to parse RSS feed %s: %s", url, e)
        return headlines

    def fetch_all_headlines(self, max_per_feed: int = 30) -> pd.DataFrame:
        """Fetch headlines from all configured financial news sources.

        Args:
            max_per_feed: Max headlines per source.

        Returns:
            DataFrame with columns: title, link, published, source, fetched_at.
        """
        all_headlines = []
        for name, url in FINANCIAL_RSS_FEEDS.items():
            logger.info("Fetching headlines from %s (%s)", name, url)
            headlines = self.fetch_rss_headlines(url, max_items=max_per_feed)
            for h in headlines:
                h["source"] = name
            all_headlines.extend(headlines)

        df = pd.DataFrame(all_headlines)
        if not df.empty:
            df["fetched_at"] = datetime.now()
            # Convert published dates
            try:
                df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
            except Exception:
                df["published_dt"] = df["fetched_at"]

        return df

    def fetch_market_sentiment_summary(self) -> dict:
        """Quick summary of current market sentiment polarity from recent headlines.

        Returns:
            Dict with 'total_headlines', 'sources' keys.
        """
        df = self.fetch_all_headlines(max_per_feed=20)
        return {
            "total_headlines": len(df),
            "sources": df["source"].value_counts().to_dict() if not df.empty else {},
        }

    @staticmethod
    def _headers() -> dict:
        """Get browser-like headers to avoid blocking."""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }

    @staticmethod
    def sample_headlines_for_llm(n: int = 20) -> list[str]:
        half = n // 2
        positive_samples = [
            "央行超预期降准50bp，释放长期流动性约1.2万亿",
            "沪深300指数创年内新高，北向资金持续净流入",
            "国务院发布促进民营经济发展壮大意见",
            "人工智能板块持续走强，多只个股创历史新高",
            "消费数据超预期，社零总额同比增长5.8%",
            "新能源车产销两旺，渗透率突破50%",
            "外资机构上调中国GDP增长预期至5.2%",
            "半导体国产替代加速，龙头企业订单饱满",
            "央企改革深化提升行动启动，相关板块活跃",
            "中秋国庆消费旺季来临，旅游餐饮预订火爆",
        ]
        negative_samples = [
            "房地产销售数据持续低迷，房企资金链承压",
            "人民币汇率跌破7.3关口，出口型企业承压",
            "地方债务风险引关注，城投债收益率上行",
            "中美科技竞争升级，半导体出口管制加码",
            "通缩压力加大，CPI同比仅增长0.1%",
            "青年失业率居高不下，就业形势严峻",
            "地缘政治风险升温，台海局势紧张",
            "上市公司业绩预告密集暴雷，多股跌停",
            "信托产品逾期兑付风险蔓延，金融监管加码",
            "全球贸易增速放缓，出口订单连续下滑",
        ]
        selected = positive_samples[:half] + negative_samples[:half]
        import random
        random.shuffle(selected)
        return selected

        return selected
