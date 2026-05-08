import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


TICKERS = [
    "SOFI", "PLTR", "RIVN", "LCID", "OPEN",
    "F", "NIO", "GRAB", "HOOD", "SNAP",
    "RKLB", "IONQ", "ASTS", "DNA", "BBAI",
    "SMCI", "APLD", "SGML", "SNDK"
]


def get_stock_data(ticker: str) -> dict | None:
    """
    Pulls recent market data for one ticker and calculates simple screening metrics.
    """

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if hist.empty:
            return None

        latest_price = float(hist["Close"].iloc[-1])
        price_30d_ago = float(hist["Close"].iloc[-30]) if len(hist) >= 30 else np.nan

        return_30d = ((latest_price / price_30d_ago) - 1) * 100 if not np.isnan(price_30d_ago) else np.nan

        avg_volume_30d = float(hist["Volume"].tail(30).mean())
        latest_volume = float(hist["Volume"].iloc[-1])

        volume_spike = latest_volume / avg_volume_30d if avg_volume_30d > 0 else np.nan

        volatility_30d = hist["Close"].pct_change().tail(30).std() * 100

        return {
            "ticker": ticker,
            "latest_price": round(latest_price, 2),
            "return_30d_pct": round(return_30d, 2),
            "avg_volume_30d": round(avg_volume_30d, 0),
            "latest_volume": round(latest_volume, 0),
            "volume_spike": round(volume_spike, 2),
            "volatility_30d_pct": round(volatility_30d, 2),
        }

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def calculate_score(row: pd.Series) -> float:
    """
    Simple first-version scoring model.
    This is not a trading signal yet.
    It only ranks candidates.
    """

    score = 0

    # Price rule
    if row["latest_price"] < 30:
        score += 25

    # Momentum rule
    if row["return_30d_pct"] > 0:
        score += 25
    if row["return_30d_pct"] > 10:
        score += 10

    # Volume confirmation
    if row["avg_volume_30d"] > 500_000:
        score += 20

    # Volume spike
    if row["volume_spike"] > 1.2:
        score += 10

    # Risk penalty
    if row["volatility_30d_pct"] > 5:
        score -= 10

    return round(score, 2)


def run_screener():
    results = []

    for ticker in TICKERS:
        print(f"Checking {ticker}...")
        data = get_stock_data(ticker)

        if data:
            results.append(data)

    df = pd.DataFrame(results)

    if df.empty:
        print("No data found.")
        return

    # Filter stocks below $50
    df = df[df["latest_price"] < 50].copy()

    # Score stocks
    df["growth_score"] = df.apply(calculate_score, axis=1)

    # Sort by score
    df = df.sort_values(by="growth_score", ascending=False)

    # Save output
    today = datetime.now().strftime("%Y-%m-%d")

    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / f"stock_screener_{today}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

    print("\nTop candidates:")
    print(df)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    run_screener()