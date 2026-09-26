"""Utilities for fetching market data from Yahoo Finance."""

import pandas as pd
import yfinance as yf


def get_stock_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Return historical OHLCV data for a ticker."""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)

    if history.empty:
        raise ValueError(f"No market data found for ticker: {ticker}")

    return history


def get_stock_info(ticker: str) -> dict:
    """Return basic company metadata for a ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap"),
    }
