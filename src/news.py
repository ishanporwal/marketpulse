"""Utilities for fetching financial news."""

import os
import re

import yfinance as yf
from dotenv import load_dotenv


load_dotenv()


def _get_company_name(ticker: str) -> str:
    """Return a simplified company name for headline filtering."""
    stock = yf.Ticker(ticker)
    info = stock.info

    name = (
        info.get("shortName")
        or info.get("longName")
        or ticker
    )

    # Remove common corporate suffixes so "Apple Inc." becomes "Apple".
    name = re.sub(
        r"\b(inc|incorporated|corp|corporation|plc|ltd|limited|company|co)\b\.?",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return " ".join(name.split())


def get_yahoo_news(ticker: str, limit: int = 10) -> list[dict]:
    """Fetch and filter recent Yahoo Finance news for a ticker."""
    stock = yf.Ticker(ticker)
    raw_news = stock.news

    company_name = _get_company_name(ticker).lower()
    ticker_lower = ticker.lower()

    articles = []

    for item in raw_news:
        content = item.get("content", {})

        title = content.get("title") or item.get("title")

        if not title:
            continue

        title_lower = title.lower()

        if (
            ticker_lower not in title_lower
            and company_name not in title_lower
        ):
            continue

        provider = content.get("provider", {})
        publisher = (
            provider.get("displayName")
            or item.get("publisher")
            or "Unknown"
        )

        canonical_url = content.get("canonicalUrl", {})
        link = canonical_url.get("url") or item.get("link")

        articles.append(
            {
                "title": title,
                "publisher": publisher,
                "link": link,
            }
        )

        if len(articles) >= limit:
            break

    return articles


def get_marketaux_news(ticker: str, limit: int = 3) -> list[dict]:
    """Fetch ticker-specific financial news from Marketaux."""
    raise NotImplementedError(
        "Marketaux support will be added before deployment."
    )


def get_stock_news(ticker: str, limit: int = 5) -> list[dict]:
    """Fetch financial news using the configured provider."""
    provider = os.getenv("NEWS_PROVIDER", "yahoo").lower()

    if provider == "marketaux":
        return get_marketaux_news(ticker, limit)

    return get_yahoo_news(ticker, limit)

def get_analyzed_news(ticker: str, limit: int = 5) -> list[dict]:
    """Return recent news with FinBERT sentiment labels."""
    from src.sentiment import analyze_sentiment

    articles = get_stock_news(ticker, limit)

    if not articles:
        return []

    sentiments = analyze_sentiment(
        [article["title"] for article in articles]
    )

    analyzed_articles = []

    for article, sentiment in zip(articles, sentiments):
        analyzed_articles.append(
            {
                **article,
                "sentiment": sentiment["label"],
                "confidence": sentiment["confidence"],
            }
        )

    return analyzed_articles
