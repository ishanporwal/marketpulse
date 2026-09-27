"""Utilities for fetching and analyzing financial news."""

import os
import re

import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"

VALID_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
    "mixed",
}

CONTEXTUAL_MARKERS = (
    " but ",
    " despite ",
    " although ",
    " yet ",
    " even as ",
)


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

        link = (
            canonical_url.get("url")
            or item.get("link")
        )

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


def get_marketaux_news(
    ticker: str,
    limit: int = 3,
) -> list[dict]:
    """Fetch ticker-specific financial news from Marketaux."""
    api_key = os.getenv("MARKETAUX_API_KEY")

    if not api_key:
        raise ValueError(
            "MARKETAUX_API_KEY is not set."
        )

    published_after = (
        datetime.now(timezone.utc)
        - timedelta(days=7)
    ).strftime("%Y-%m-%d")

    response = requests.get(
        "https://api.marketaux.com/v1/news/all",
        params={
            "api_token": api_key,
            "symbols": ticker,
            "filter_entities": "true",
            "must_have_entities": "true",
            "group_similar": "true",
            "language": "en",
            "published_after": published_after,
            "sort": "entity_match_score",
            "sort_order": "desc",
            "limit": min(limit, 3),
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    articles = []
    seen_titles = set()

    for item in data.get("data", []):
        title = item.get("title")
        link = item.get("url")

        if not title:
            continue

        # Normalize title so duplicate headlines are removed
        # even if Marketaux gives them different URLs.
        dedupe_key = " ".join(
            title.lower().split()
        )

        if dedupe_key in seen_titles:
            continue

        seen_titles.add(dedupe_key)

        articles.append(
            {
                "title": title,
                "publisher": item.get(
                    "source",
                    "Unknown",
                ),
                "link": link,
                "published_at": item.get(
                    "published_at"
                ),
            }
        )

    return articles[:limit]


def get_stock_news(
    ticker: str,
    limit: int = 5,
) -> list[dict]:
    """Fetch financial news using the configured provider."""
    provider = os.getenv(
        "NEWS_PROVIDER",
        "yahoo",
    ).lower()

    if provider == "marketaux":
        return get_marketaux_news(
            ticker,
            limit,
        )

    return get_yahoo_news(
        ticker,
        limit,
    )


def needs_llm_review(
    title: str,
    sentiment: dict,
) -> bool:
    """Return whether a FinBERT result needs contextual review."""
    confidence = sentiment["confidence"]
    probabilities = sentiment.get("probabilities", {})

    if confidence < 0.75:
        return True

    if len(probabilities) >= 2:
        scores = sorted(
            probabilities.values(),
            reverse=True,
        )

        if scores[0] - scores[1] < 0.20:
            return True

    if "?" in title:
        return True

    return False

def classify_with_gemini(
    ticker: str,
    title: str,
    finbert_sentiment: dict,
) -> str | None:
    """Use Gemini to contextually review an ambiguous headline."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    client = genai.Client(
        api_key=api_key
    )

    finbert_label = finbert_sentiment[
        "label"
    ]
    finbert_confidence = finbert_sentiment[
        "confidence"
    ]

    prompt = f"""
You are reviewing the sentiment of a financial news headline about {ticker}.

Your task is to determine the headline's overall investor-facing thesis
toward the company or its stock.

This is different from ordinary linguistic sentiment.

Use exactly one of these labels:

positive
- The overall thesis is materially favorable for the company or stock.

negative
- The overall thesis is materially unfavorable for the company or stock.

neutral
- The headline is primarily factual or informational and has no clear
  favorable or unfavorable thesis.

mixed
- The headline contains meaningful favorable and unfavorable implications,
  with neither clearly dominating.

Rules:
- Judge the overall thesis, not isolated positive or negative words.
- Historical price movement alone does not determine sentiment.
- A stock rally followed by an argument that the stock is overvalued
  should not automatically be considered positive.
- Focus only on the implications for {ticker}.
- Do not infer information that is not present in the headline.
- FinBERT's result is provided as a signal, but you may disagree with it.
- Return ONLY one word:
  positive, negative, neutral, or mixed

Ticker: {ticker}

Headline:
{title}

FinBERT result:
{finbert_label}

FinBERT confidence:
{finbert_confidence:.3f}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if not response.text:
            return None

        label = (
            response.text
            .strip()
            .lower()
            .replace(".", "")
        )

        if label not in VALID_SENTIMENTS:
            return None

        return label

    except Exception:
        return None


def get_analyzed_news(
    ticker: str,
    limit: int = 5,
) -> list[dict]:
    """Return news with hybrid FinBERT and contextual sentiment."""
    from src.sentiment import analyze_sentiment

    articles = get_stock_news(
        ticker,
        limit,
    )

    if not articles:
        return []

    finbert_results = analyze_sentiment(
        [
            article["title"]
            for article in articles
        ]
    )

    analyzed_articles = []

    for article, finbert in zip(
        articles,
        finbert_results,
    ):
        final_sentiment = finbert["label"]
        sentiment_source = "finbert"

        if needs_llm_review(
            article["title"],
            finbert,
        ):
            llm_sentiment = classify_with_gemini(
                ticker=ticker,
                title=article["title"],
                finbert_sentiment=finbert,
            )

            if llm_sentiment is not None:
                final_sentiment = llm_sentiment
                sentiment_source = "gemini"

        analyzed_articles.append(
            {
                **article,
                "sentiment": final_sentiment,
                "sentiment_source": sentiment_source,
                "finbert_sentiment": finbert["label"],
                "finbert_confidence": finbert[
                    "confidence"
                ],
                "confidence": finbert[
                    "confidence"
                ],
            }
        )

    return analyzed_articles
