import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


TICKERS = [
    "SOFI", "PLTR", "RIVN", "LCID", "OPEN",
    "F", "NIO", "GRAB", "HOOD", "SNAP",
    "RKLB", "IONQ", "ASTS", "DNA", "BBAI",
    "SMCI", "APLD", "SGML", "SNDK", "KVYO"
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

        sma_20 = float(hist["Close"].tail(20).mean())
        sma_50 = float(hist["Close"].tail(50).mean())
        rsi_14 = calculate_rsi(hist["Close"], period=14)

        above_sma_20 = latest_price > sma_20
        above_sma_50 = latest_price > sma_50

        technical_score = calculate_technical_score(
            latest_price=latest_price,
            sma_20=sma_20,
            sma_50=sma_50,
            rsi_14=rsi_14
        )
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

            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "rsi_14": rsi_14,
            "above_sma_20": above_sma_20,
            "above_sma_50": above_sma_50,
            "technical_score": technical_score,
        }

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> float:
    """
    Calculates the latest RSI value.
    RSI helps identify momentum strength and overbought/oversold conditions.
    """

    delta = close_prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    latest_rsi = rsi.iloc[-1]

    if pd.isna(latest_rsi):
        return 0

    return round(float(latest_rsi), 2)


def calculate_technical_score(
    latest_price: float,
    sma_20: float,
    sma_50: float,
    rsi_14: float
) -> float:
    """
    Scores technical setup.
    """

    score = 0

    if latest_price > sma_20:
        score += 30

    if latest_price > sma_50:
        score += 30

    if 50 <= rsi_14 <= 70:
        score += 30
    elif 40 <= rsi_14 < 50:
        score += 15
    elif 70 < rsi_14 <= 75:
        score += 15
    elif rsi_14 > 75:
        score -= 10

    if sma_20 > sma_50:
        score += 10

    return clamp(score)


def clamp(value: float, min_value: float = 0, max_value: float = 100) -> float:
    return max(min_value, min(value, max_value))


def calculate_score(row: pd.Series) -> pd.Series:
    """
    Improved multi-factor scoring model.
    Returns separate component scores and a final growth score.
    """

    # 1. Momentum Score
    # Positive 30D return is good, but extreme moves are capped.
    return_30d = row["return_30d_pct"]

    if pd.isna(return_30d):
        momentum_score = 0
    else:
        momentum_score = clamp((return_30d + 10) * 3)
        # Example:
        # -10% return = 0
        # 0% return = 30
        # 10% return = 60
        # 20% return = 90

    # 2. Volume Score
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

    # 3. Price Opportunity Score
    # Stocks below $30 qualify, but very low-priced stocks are riskier.
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

    # 4. Risk Score
    # Lower volatility gets better score.
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

    # Final weighted score
    final_score = (
            0.30 * momentum_score +
            0.20 * volume_score +
            0.15 * price_score +
            0.15 * risk_score +
            0.20 * row["technical_score"]
    )

    return pd.Series({
        "momentum_score": round(momentum_score, 2),
        "volume_score": round(volume_score, 2),
        "price_score": round(price_score, 2),
        "risk_score": round(risk_score, 2),
        "growth_score": round(final_score, 2),
    })

def generate_reason(row: pd.Series) -> str:
    reasons = []

    if row["return_30d_pct"] > 10:
        reasons.append("strong 30D momentum")
    elif row["return_30d_pct"] > 0:
        reasons.append("positive 30D momentum")
    else:
        reasons.append("weak 30D momentum")

    if row["above_sma_20"] and row["above_sma_50"]:
        reasons.append("price above 20D and 50D moving averages")
    elif row["above_sma_20"]:
        reasons.append("price above 20D moving average")

    if 50 <= row["rsi_14"] <= 70:
        reasons.append("healthy RSI momentum")
    elif row["rsi_14"] > 75:
        reasons.append("RSI appears overextended")

    if row["volume_spike"] > 1.5:
        reasons.append("strong volume confirmation")
    elif row["volume_spike"] > 1.2:
        reasons.append("moderate volume confirmation")

    return ", ".join(reasons)


def assign_decision(score):
    if score >= 80:
        return "Strong Watch"
    elif score >= 65:
        return "Watch"
    elif score >= 50:
        return "Weak Watch"
    else:
        return "Ignore"

def generate_risk_note(row: pd.Series) -> str:
    risks = []

    if row["volatility_30d_pct"] > 8:
        risks.append("very high volatility")
    elif row["volatility_30d_pct"] > 5:
        risks.append("elevated volatility")

    if row["rsi_14"] > 75:
        risks.append("possible short-term overextension")

    if row["avg_volume_30d"] < 500_000:
        risks.append("low liquidity")

    if row["latest_price"] < 5:
        risks.append("very low-priced stock risk")

    if not risks:
        return "No major technical risk flagged"

    return ", ".join(risks)

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
    score_columns = df.apply(calculate_score, axis=1)
    df = pd.concat([df, score_columns], axis=1)

    # Sort by score
    df = df.sort_values(by="growth_score", ascending=False)
    df["decision"] = df["growth_score"].apply(assign_decision)
    df["reason"] = df.apply(generate_reason, axis=1)
    df["risk_note"] = df.apply(generate_risk_note, axis=1)

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