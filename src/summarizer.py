"""Generate concise market summaries from model and news signals."""

import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

FEATURE_LABELS = {
    "return_1d": "1-day price movement",
    "return_5d": "5-day price momentum",
    "return_20d": "20-day price momentum",
    "volatility_5d": "short-term volatility",
    "volatility_20d": "recent volatility",
    "volume_zscore_20d": "trading volume",
    "high_low_range": "intraday price swings",
    "rsi_14": "recent price momentum",
    "relative_return_spy_5d": "recent performance versus the broader market",
    "relative_return_spy_20d": "longer-term performance versus the broader market",
}


def generate_market_summary(
    ticker: str,
    risk_score: float,
    contributions: list[dict],
    articles: list[dict],
) -> str:
    """Generate a short explanation of the current market signals."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    top_signals = contributions[:5]

    signal_text = "\n".join(
        (
            f"- {FEATURE_LABELS.get(item['feature'], item['feature'])}: "
            f"{'increases' if item['contribution'] > 0 else 'decreases'} "
            f"the model's risk score"
        )
        for item in top_signals
    )

    news_text = "\n".join(
        (
            f"- [{article['sentiment'].upper()}] "
            f"{article['title']}"
        )
        for article in articles
    )

    if not news_text:
        news_text = "- No recent relevant news was found."

    prompt = f"""
    You are writing a short market overview for someone viewing {ticker}
    on a financial dashboard.

    The quantitative model produced an abnormal-move risk score of
    {risk_score * 100:.1f}/100.

    The most influential quantitative signals are:
    {signal_text}

    Recent financial news headlines classified by FinBERT:
    {news_text}

    Write a concise 3-4 sentence summary for a general investor.

    Follow these rules carefully:

    - Treat the abnormal-move risk score as a model score, not a probability,
    forecast, or prediction of whether the stock will rise or fall.
    - Do not imply that a higher abnormal-move risk score means the stock
    is likely to decline. The score represents risk of an unusually large move
    in either direction.
    - Clearly separate quantitative model signals from news sentiment.
    - Describe model signals as factors that increase or decrease the model's
    score. Do not claim that they caused market behavior.
    - Translate quantitative features into simple investor-friendly language.
    - Do not mention raw feature names, raw contribution values, z-scores,
    indicator values, or machine-learning terminology.
    - Focus on the strongest signals rather than listing every feature.
    - If closely related signals point in opposite directions, describe them as
    mixed rather than presenting only one side.
    - Do not describe a signal as positive or negative for the stock price.
    Only describe whether it raises or lowers abnormal-move risk.
    - Base all news discussion only on the supplied headlines and FinBERT labels.
    - Do not infer causes, consequences, partnerships, strategy, market reaction,
    or relationships that are not explicitly stated in the supplied headline.
    - Do not add outside knowledge about the company or market.
    - If only one article is provided, refer to it as a single headline or article;
    do not generalize it into the overall state of financial news.
    - If multiple articles are provided and their sentiment labels disagree,
    describe the news sentiment as mixed.
    - Do not overstate FinBERT sentiment. Describe it as the sentiment assigned
    to the supplied news, not as evidence about future stock performance.
    - Avoid overly technical or promotional financial language such as
    "bullish", "bearish", "upward pull", "risk outlook", or "market catalyst".
    - Use direct, neutral language.
    - Do not provide buy, sell, hold, price-target, or investment recommendations.
    - Do not invent any facts.
    """
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text.strip()
