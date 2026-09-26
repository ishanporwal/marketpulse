import streamlit as st
import plotly.graph_objects as go

from src.market_data import get_stock_history, get_stock_info
from src.news import get_analyzed_news
from src.predictor import predict_abnormal_move
from src.summarizer import generate_market_summary


@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_summary(
    ticker,
    risk_score,
    contributions,
    articles,
):
    """Cache generated market summaries for 30 minutes."""
    return generate_market_summary(
        ticker=ticker,
        risk_score=risk_score,
        contributions=contributions,
        articles=articles,
    )


st.set_page_config(
    page_title="MarketPulse",
    layout="wide",
)

st.title("MarketPulse")

if "ticker" not in st.session_state:
    st.session_state.ticker = "NVDA"

with st.form("ticker_form"):
    ticker_input = st.text_input(
        "Ticker",
        value=st.session_state.ticker,
    )

    submitted = st.form_submit_button("Analyze")

if submitted:
    st.session_state.ticker = ticker_input.upper()

ticker = st.session_state.ticker

try:
    data = get_stock_history(ticker)
    info = get_stock_info(ticker)

    risk_score, latest_features, contributions = predict_abnormal_move(ticker)
    articles = get_analyzed_news(ticker, limit=5)

    current_price = data["Close"].iloc[-1]
    previous_close = data["Close"].iloc[-2]

    daily_change = current_price - previous_close
    daily_change_percent = (daily_change / previous_close) * 100

    if articles:
        sentiment_labels = [
            article["sentiment"]
            for article in articles
        ]

        overall_sentiment = max(
            set(sentiment_labels),
            key=sentiment_labels.count,
        ).title()
    else:
        overall_sentiment = "N/A"

    st.subheader(f"{info['name']} ({ticker})")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{daily_change_percent:.2f}%",
        )

    with col2:
        st.metric(
            "Abnormal Move Risk Score",
            f"{risk_score * 100:.1f}/100",
        )

    with col3:
        st.metric(
            "Recent News Sentiment",
            overall_sentiment,
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close",
        )
    )

    fig.update_layout(
        title=f"{ticker} Price — Last 6 Months",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

    with st.spinner("Generating market summary..."):
        summary = get_cached_summary(
            ticker,
            risk_score,
            contributions,
            articles,
        )

    st.subheader("AI Market Summary")
    st.write(summary)

    st.subheader("Model Signals")

    for item in contributions[:5]:
        direction = (
            "Increases risk"
            if item["contribution"] > 0
            else "Decreases risk"
        )

        st.write(
            f"{item['feature']}: "
            f"{direction} "
            f"({item['contribution']:+.3f})"
        )
        
    st.subheader("Recent News")

    if not articles:
        st.info("No recent relevant news found.")

    for article in articles:
        sentiment = article["sentiment"].upper()
        confidence = article["confidence"] * 100

        st.markdown(
            f"**{sentiment} · {confidence:.0f}% confidence**"
        )

        st.markdown(
            f"[{article['title']}]({article['link']})"
        )

        st.caption(article["publisher"])

        st.divider()

except Exception as e:
    st.error(str(e))
