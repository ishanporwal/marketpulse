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
    """Classify financial text using FinBERT."""
    if not texts:
        return []

    classifier = get_sentiment_model()

    results = classifier(
        texts,
        truncation=True,
        top_k=None,
    )

    analyzed = []

    for result in results:
        probabilities = {
            item["label"].lower(): float(item["score"])
            for item in result
        }

        label = max(
            probabilities,
            key=probabilities.get,
        )

        analyzed.append(
            {
                "label": label,
                "confidence": probabilities[label],
                "probabilities": probabilities,
            }
        )

    return analyzed
