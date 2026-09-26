import streamlit as st
import plotly.graph_objects as go

from src.market_data import get_stock_history, get_stock_info


st.set_page_config(
    page_title="MarketPulse",
    layout="wide",
)

st.title("MarketPulse")

ticker = st.text_input(
    "Ticker",
    value="NVDA",
).upper()

try:
    data = get_stock_history(ticker)
    info = get_stock_info(ticker)

    current_price = data["Close"].iloc[-1]
    previous_close = data["Close"].iloc[-2]

    daily_change = current_price - previous_close
    daily_change_percent = (daily_change / previous_close) * 100

    st.subheader(f"{info['name']} ({ticker})")

    st.metric(
        label="Current Price",
        value=f"${current_price:.2f}",
        delta=f"{daily_change_percent:.2f}%"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close"
        )
    )

    fig.update_layout(
        title=f"{ticker} Price — Last 6 Months",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

except Exception as e:
    st.error(str(e))
