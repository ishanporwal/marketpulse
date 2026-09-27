import html
from collections import Counter

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.market_data import get_stock_history, get_stock_info
from src.news import get_analyzed_news
from src.predictor import predict_abnormal_move
from src.summarizer import generate_market_summary


FEATURE_LABELS = {
    "return_1d": "1-Day Price Movement",
    "return_5d": "5-Day Price Momentum",
    "return_20d": "20-Day Price Momentum",
    "volatility_5d": "Short-Term Volatility",
    "volatility_20d": "Recent Volatility",
    "volume_zscore_20d": "Trading Volume",
    "high_low_range": "Intraday Price Swings",
    "rsi_14": "Price Momentum (RSI)",
    "relative_return_spy_5d": "5-Day Performance vs. S&P 500",
    "relative_return_spy_20d": "20-Day Performance vs. S&P 500",
}


@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_summary(
    ticker,
    risk_score,
    latest_features,
    contributions,
    articles,
):
    """Cache generated market summaries for 30 minutes."""
    return generate_market_summary(
        ticker=ticker,
        risk_score=risk_score,
        latest_features=latest_features,
        contributions=contributions,
        articles=articles,
    )

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_news(ticker: str):
    """Cache analyzed financial news for 30 minutes."""
    return get_analyzed_news(ticker, limit=5)


def format_market_cap(market_cap):
    """Format market capitalization into a readable value."""
    if not market_cap:
        return "N/A"

    if market_cap >= 1_000_000_000_000:
        return f"${market_cap / 1_000_000_000_000:.2f}T"

    if market_cap >= 1_000_000_000:
        return f"${market_cap / 1_000_000_000:.1f}B"

    if market_cap >= 1_000_000:
        return f"${market_cap / 1_000_000:.1f}M"

    return f"${market_cap:,.0f}"


def get_overall_sentiment(articles: list[dict]) -> str:
    """Return an aggregate sentiment label for recent news."""
    if not articles:
        return "N/A"

    counts = Counter(
        article["sentiment"].lower()
        for article in articles
    )

    highest_count = max(counts.values())

    leaders = [
        sentiment
        for sentiment, count in counts.items()
        if count == highest_count
    ]

    if len(leaders) > 1:
        return "Mixed"

    return leaders[0].title()


def format_feature_value(feature: str, value: float) -> str:
    """Format a model feature into a readable value."""
    percentage_features = {
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_5d",
        "volatility_20d",
        "high_low_range",
        "relative_return_spy_5d",
        "relative_return_spy_20d",
    }

    if feature in percentage_features:
        return f"{value * 100:+.1f}%"

    if feature == "volume_zscore_20d":
        return f"{value:+.2f}σ"

    if feature == "rsi_14":
        return f"{value:.1f}"

    return f"{value:.3f}"


def sentiment_badge(sentiment: str) -> str:
    """Return styled HTML for a sentiment label."""
    sentiment = sentiment.lower()

    styles = {
        "positive": ("#166534", "#dcfce7"),
        "negative": ("#991b1b", "#fee2e2"),
        "neutral": ("#475569", "#f1f5f9"),
        "mixed": ("#92400e", "#fef3c7"),
    }
    
    foreground, background = styles.get(
        sentiment,
        ("#475569", "#f1f5f9"),
    )

    return (
        f'<span style="'
        f"color:{foreground};"
        f"background:{background};"
        f"padding:4px 9px;"
        f"border-radius:999px;"
        f"font-size:0.75rem;"
        f"font-weight:700;"
        f'letter-spacing:0.03em;">'
        f"{html.escape(sentiment.upper())}"
        f"</span>"
    )


st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1500px;
        }

        h1 {
            margin-bottom: 0.15rem;
        }

        div[data-testid="stMetric"] {
            padding: 0.25rem 0;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }

        .section-subtitle {
            color: #64748b;
            font-size: 0.9rem;
            margin-top: -0.4rem;
            margin-bottom: 1rem;
        }

        .signal-title {
            font-weight: 600;
            font-size: 0.95rem;
        }

        .signal-value {
            color: #64748b;
            font-size: 0.82rem;
        }

        .news-source {
            color: #64748b;
            font-size: 0.82rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Header and ticker search
# -------------------------------------------------------------------

st.title("MarketPulse")
st.caption(
    "ML-powered market intelligence using quantitative signals, "
    "financial news sentiment, and AI-generated explanations."
)

if "ticker" not in st.session_state:
    st.session_state.ticker = "NVDA"

with st.form("ticker_form"):
    input_col, button_col = st.columns([8, 1])

    with input_col:
        ticker_input = st.text_input(
            "Ticker",
            value=st.session_state.ticker,
            placeholder="Enter ticker, e.g. NVDA",
        )

    with button_col:
        st.write("")
        st.write("")

        submitted = st.form_submit_button(
            "Analyze",
            use_container_width=True,
        )

if submitted:
    cleaned_ticker = ticker_input.strip().upper()

    if cleaned_ticker:
        st.session_state.ticker = cleaned_ticker

ticker = st.session_state.ticker


# -------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------

try:
    data = get_stock_history(ticker)
    info = get_stock_info(ticker)

    risk_score, latest_features, contributions = predict_abnormal_move(
        ticker
    )

    recent_volatility = latest_features["volatility_20d"] * 100

    articles = get_cached_news(ticker)

    current_price = data["Close"].iloc[-1]
    previous_close = data["Close"].iloc[-2]

    daily_change = current_price - previous_close
    daily_change_percent = (
        daily_change / previous_close
    ) * 100

    overall_sentiment = get_overall_sentiment(articles)

    # ---------------------------------------------------------------
    # Company heading
    # ---------------------------------------------------------------

    st.divider()

    st.subheader(f"{info['name']} ({ticker})")

    company_details = []

    if info.get("sector") and info["sector"] != "N/A":
        company_details.append(info["sector"])

    if info.get("market_cap"):
        company_details.append(
            f"Market Cap {format_market_cap(info['market_cap'])}"
        )

    if company_details:
        st.caption(" · ".join(company_details))

    # ---------------------------------------------------------------
    # Top metrics
    # ---------------------------------------------------------------

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        with st.container(border=True):
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{daily_change_percent:+.2f}%",
            )

    with metric_col2:
        with st.container(border=True):
            st.metric(
                "5-Day Abnormal Move Risk",
                f"{risk_score * 100:.1f}/100",
            )

            st.progress(
                min(max(float(risk_score), 0.0), 1.0)
            )

            st.caption(
                "Risk of an unusually large move relative to "
                "the stock's own recent volatility."
            )

    with metric_col3:
        with st.container(border=True):
            st.metric(
                "Recent Volatility",
                f"{recent_volatility:.1f}%",
            )

            st.caption(
                "20-day standard deviation of daily returns."
            )

    with metric_col4:
        with st.container(border=True):
            st.metric(
                "Recent News Sentiment",
                overall_sentiment,
            )

            if articles:
                counts = Counter(
                    article["sentiment"].lower()
                    for article in articles
                )

                sentiment_parts = [
                    f"{counts.get('positive', 0)} positive",
                    f"{counts.get('neutral', 0)} neutral",
                    f"{counts.get('negative', 0)} negative",
                ]

                if counts.get("mixed", 0):
                    sentiment_parts.append(
                        f"{counts['mixed']} mixed"
                    )

                st.caption(
                    " · ".join(sentiment_parts)
                )

            else:
                st.caption("No recent relevant articles found.")

    # ---------------------------------------------------------------
    # Price + volume chart
    # ---------------------------------------------------------------

    st.subheader("Market Activity")
    st.markdown(
        '<div class="section-subtitle">'
        "Six months of price and trading volume."
        "</div>",
        unsafe_allow_html=True,
    )

    chart = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    chart.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close",
            line={
                "width": 2.2,
            },
            hovertemplate=(
                "%{x|%b %d, %Y}<br>"
                "$%{y:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    chart.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            opacity=0.45,
            hovertemplate=(
                "%{x|%b %d, %Y}<br>"
                "Volume: %{y:,.0f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    chart.update_layout(
        height=520,
        margin={
            "l": 10,
            "r": 10,
            "t": 15,
            "b": 10,
        },
        showlegend=False,
        hovermode="x unified",
    )

    chart.update_yaxes(
        title_text="Price ($)",
        row=1,
        col=1,
        gridcolor="rgba(128,128,128,0.15)",
    )

    chart.update_yaxes(
        title_text="Volume",
        row=2,
        col=1,
        showgrid=False,
    )

    chart.update_xaxes(
        showgrid=False,
        row=2,
        col=1,
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    # ---------------------------------------------------------------
    # AI summary
    # ---------------------------------------------------------------

    try:
        with st.spinner("Generating market summary..."):
            summary = get_cached_summary(
                ticker,
                risk_score,
                latest_features.to_dict(),
                contributions,
                articles,
            )

        st.subheader("AI Market Summary")

        with st.container(border=True):
            st.write(summary)

            st.caption(
                "Generated from the quantitative model signals and "
                "analyzed financial news shown below."
            )

    except Exception:
        st.warning(
            "The AI summary is temporarily unavailable. "
            "The underlying market and news analysis is still shown below."
        )

    # ---------------------------------------------------------------
    # Model signals
    # ---------------------------------------------------------------

    st.subheader("Model Signals")

    st.markdown(
        '<div class="section-subtitle">'
        "The strongest features influencing the current XGBoost risk score."
        "</div>",
        unsafe_allow_html=True,
    )

    top_contributions = contributions[:5]

    if top_contributions:
        max_contribution = max(
            abs(item["contribution"])
            for item in top_contributions
        )

        for item in top_contributions:
            feature = item["feature"]
            contribution = item["contribution"]
            value = item["value"]

            label = FEATURE_LABELS.get(
                feature,
                feature.replace("_", " ").title(),
            )

            increases_risk = contribution > 0

            direction_text = (
                "Raises risk"
                if increases_risk
                else "Lowers risk"
            )

            direction_icon = (
                "↑"
                if increases_risk
                else "↓"
            )

            left_col, right_col = st.columns(
                [5, 2]
            )

            with left_col:
                st.markdown(
                    f"**{label}**"
                )

                st.progress(
                    min(
                        abs(contribution)
                        / max_contribution,
                        1.0,
                    )
                )

            with right_col:
                st.markdown(
                    f"**{direction_icon} {direction_text}**"
                )

                st.caption(
                    f"Current value: "
                    f"{format_feature_value(feature, value)}"
                )

    # ---------------------------------------------------------------
    # Recent news
    # ---------------------------------------------------------------

    st.subheader("Recent News")

    st.markdown(
        '<div class="section-subtitle">'
        "Recent ticker-related financial headlines analyzed using "
        "FinBERT with contextual AI review when needed."
        "</div>",
        unsafe_allow_html=True,
    )
    
    if not articles:
        st.info(
            "No recent relevant news found for this ticker."
        )

    for article in articles:
        sentiment = article["sentiment"]

        with st.container(border=True):
            st.markdown(
                sentiment_badge(sentiment),
                unsafe_allow_html=True,
            )

            st.markdown(
                f"### [{article['title']}]"
                f"({article['link']})"
            )

            publisher = article.get(
                "publisher",
                "Unknown",
            )

            published_at = article.get(
                "published_at"
            )

            metadata = publisher

            if published_at:
                try:
                    published_date = published_at[:10]
                    metadata += (
                        f" · {published_date}"
                    )
                except (
                    TypeError,
                    IndexError,
                ):
                    pass

            sentiment_source = article.get(
                "sentiment_source",
                "finbert",
            )

            finbert_confidence = (
                article.get(
                    "finbert_confidence",
                    article.get(
                        "confidence",
                        0,
                    ),
                )
                * 100
            )

            if sentiment_source == "gemini":
                finbert_label = article.get(
                    "finbert_sentiment",
                    "unknown",
                ).title()

                sentiment_metadata = (
                    "Contextually reviewed by Gemini"
                    f" · FinBERT: {finbert_label} "
                    f"({finbert_confidence:.0f}%)"
                )
            else:
                sentiment_metadata = (
                    f"{finbert_confidence:.0f}% "
                    "FinBERT confidence"
                )

            st.caption(
                f"{metadata} · "
                f"{sentiment_metadata}"
            )
            
except Exception as error:
    st.error(
        f"Unable to analyze {ticker}: {error}"
    )
