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
    latest_features: dict,
    contributions: list[dict],
    articles: list[dict],
) -> str:
    """Generate a short explanation of the current market signals."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    daily_move = latest_features["return_1d"] * 100
    recent_volatility = latest_features["volatility_20d"] * 100

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

    The quantitative model produced a 5-day abnormal-move risk score of
    {risk_score * 100:.1f}/100.

    IMPORTANT: This score estimates the likelihood of an unusually large
    future move RELATIVE TO THE STOCK'S OWN RECENT VOLATILITY. It is not
    a measure of whether the stock itself is currently volatile or safe.

    Current market context:
    - Latest 1-day price movement: {daily_move:+.1f}%
    - Recent 20-day daily volatility: {recent_volatility:.1f}%

    The most influential quantitative signals are:
    {signal_text}

    Recent financial news classified by FinBERT:
    {news_text}

    Write a concise 3-4 sentence summary.

    Rules:
    - Write for a normal investor, not a machine learning engineer.
    - Clearly distinguish current volatility from the abnormal-move risk score.
    - Never describe the stock itself as low risk, safe, or stable simply
    because the abnormal-move risk score is low.
    - If the stock currently has large price swings or high volatility,
    explicitly clarify that a low model score only means lower risk of an
    additional move that is unusually large relative to recent behavior.
    - Do not mention raw feature names such as "volume_zscore_20d".
    - Do not include raw model contribution values.
    - Explain quantitative signals in plain English.
    - Summarize only the news provided.
    - Do not infer relationships or causes beyond what the headlines state.
    - Treat the risk score as a model score, not a literal probability.
    - Do not invent facts, causes, events, or relationships.
    - Do not provide buy, sell, or hold recommendations.
    """    
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text.strip()
