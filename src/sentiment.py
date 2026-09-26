"""Financial news sentiment analysis using FinBERT."""

from functools import lru_cache

from transformers import pipeline


MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def get_sentiment_model():
    """Load and cache the FinBERT sentiment pipeline."""
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )


def analyze_sentiment(texts: list[str]) -> list[dict]:
    """Classify financial text as positive, neutral, or negative."""
    if not texts:
        return []

    classifier = get_sentiment_model()
    results = classifier(texts, truncation=True)

    return [
        {
            "label": result["label"].lower(),
            "confidence": float(result["score"]),
        }
        for result in results
    ]
