import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

from stock_screener import TICKERS, calculate_rsi, clamp

def calculate_historical_features(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()

    df["return_30d_pct"] = df["Close"].pct_change(30) * 100
    df["avg_volume_30d"] = df["Volume"].rolling(30).mean()
    df["volume_spike"] = df["Volume"] / df["avg_volume_30d"]
    df["volatility_30d_pct"] = df["Close"].pct_change().rolling(30).std() * 100

    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    df["above_sma_20"] = df["Close"] > df["sma_20"]
    df["above_sma_50"] = df["Close"] > df["sma_50"]

    df["rsi_14"] = calculate_rsi_series(df["Close"], period=14)
    df["technical_score"] = df.apply(calculate_technical_score_row, axis=1)

    df["latest_price"] = df["Close"]

    # What happened after the signal?
    df["future_return_30d"] = (df["Close"].shift(-30) / df["Close"] - 1) * 100

    return df

def calculate_rsi_series(close_prices: pd.Series, period: int = 14) -> pd.Series:
    delta = close_prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_technical_score_row(row: pd.Series) -> float:
    score = 0

    if row["Close"] > row["sma_20"]:
        score += 30

    if row["Close"] > row["sma_50"]:
        score += 30

    rsi = row["rsi_14"]

    if 50 <= rsi <= 70:
        score += 30
    elif 40 <= rsi < 50:
        score += 15
    elif 70 < rsi <= 75:
        score += 15
    elif rsi > 75:
        score -= 10

    if row["sma_20"] > row["sma_50"]:
        score += 10

    return clamp(score)

def calculate_score_row(row: pd.Series) -> float:
    return_30d = row["return_30d_pct"]

    if pd.isna(return_30d):
        momentum_score = 0
    else:
        momentum_score = clamp((return_30d + 10) * 3)

    avg_volume = row["avg_volume_30d"]
    volume_spike = row["volume_spike"]

    volume_score = 0

    if avg_volume >= 500_000:
        volume_score += 50
    elif avg_volume >= 250_000:
        volume_score += 30
    elif avg_volume >= 100_000:
        volume_score += 15

    if volume_spike >= 2:
        volume_score += 40
    elif volume_spike >= 1.5:
        volume_score += 30
    elif volume_spike >= 1.2:
        volume_score += 20
    elif volume_spike >= 1:
        volume_score += 10

    volume_score = clamp(volume_score)

    price = row["latest_price"]

    if price <= 0:
        price_score = 0
    elif price < 5:
        price_score = 40
    elif price < 10:
        price_score = 75
    elif price < 20:
        price_score = 90
    elif price < 30:
        price_score = 80
    else:
        price_score = 0

    volatility = row["volatility_30d_pct"]

    if pd.isna(volatility):
        risk_score = 0
    elif volatility <= 2:
        risk_score = 100
    elif volatility <= 3:
        risk_score = 85
    elif volatility <= 5:
        risk_score = 65
    elif volatility <= 8:
        risk_score = 40
    else:
        risk_score = 20

    final_score = (
        0.30 * momentum_score +
        0.20 * volume_score +
        0.15 * price_score +
        0.15 * risk_score +
        0.20 * row["technical_score"]
    )

    return round(final_score, 2)


def assign_score_bucket(score: float) -> str:
    if score >= 80:
        return "Strong Watch"
    elif score >= 65:
        return "Watch"
    elif score >= 50:
        return "Weak Watch"
    else:
        return "Ignore"


def run_backtest():
    all_results = []

    for ticker in TICKERS:
        print(f"Backtesting {ticker}...")

        hist = yf.Ticker(ticker).history(period="2y")

        if hist.empty:
            continue

        df = calculate_historical_features(hist)
        df["ticker"] = ticker
        df["growth_score"] = df.apply(calculate_score_row, axis=1)
        df["decision"] = df["growth_score"].apply(assign_score_bucket)

        all_results.append(df)

    results = pd.concat(all_results)

    results = results.dropna(subset=[
        "return_30d_pct",
        "volatility_30d_pct",
        "rsi_14",
        "future_return_30d",
        "growth_score"
    ])

    results["score_decile"] = pd.qcut(
        results["growth_score"],
        10,
        labels=False,
        duplicates="drop"
    )

    summary = results.groupby("decision").agg(
        signals=("future_return_30d", "count"),

        avg_future_return_30d=("future_return_30d", "mean"),
        median_future_return_30d=("future_return_30d", "median"),

        win_rate=("future_return_30d", lambda x: (x > 0).mean() * 100),
        beat_5pct_rate=("future_return_30d", lambda x: (x > 5).mean() * 100),
        beat_10pct_rate=("future_return_30d", lambda x: (x > 10).mean() * 100),

        max_loss=("future_return_30d", "min"),
        max_gain=("future_return_30d", "max"),
        avg_score=("growth_score", "mean")
    ).reset_index()

    summary = summary.sort_values(by="avg_future_return_30d", ascending=False)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    decile_summary = results.groupby("score_decile").agg(
        signals=("future_return_30d", "count"),
        avg_future_return_30d=("future_return_30d", "mean"),
        median_future_return_30d=("future_return_30d", "median"),
        win_rate=("future_return_30d", lambda x: (x > 0).mean() * 100),
        beat_5pct_rate=("future_return_30d", lambda x: (x > 5).mean() * 100),
        avg_score=("growth_score", "mean")
    ).reset_index()

    decile_summary.to_csv(
        output_dir / "scoring_model_decile_summary.csv",
        index=False
    )

    results.to_csv(output_dir / "historical_scoring_backtest.csv", index=True)
    summary.to_csv(output_dir / "scoring_model_backtest_summary.csv", index=False)

    print("\nBacktest Summary:")
    print(summary)

    print("\nSaved:")
    print(output_dir / "historical_scoring_backtest.csv")
    print(output_dir / "scoring_model_backtest_summary.csv")


if __name__ == "__main__":
    run_backtest()