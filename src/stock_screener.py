import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


UNIVERSES = {
    "growth": [
        "SOFI", "PLTR", "RIVN", "LCID", "OPEN",
        "NIO", "HOOD", "SNAP", "RKLB", "IONQ",
        "ASTS", "BBAI", "SMCI", "APLD"
    ],
    "mega_cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "BRK-B", "JPM", "V"
    ],
    "stable_blue_chip": [
        "KO", "PEP", "MCD", "WMT", "COST",
        "PG", "JNJ", "UNH", "HD", "NKE"
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS",
        "C", "AXP", "BLK", "SCHW", "COF"
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
        "OXY", "MPC", "PSX", "VLO", "HAL"
    ],
    "semiconductors": [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM",
        "MU", "TSM", "ASML", "AMAT", "LRCX"
    ],
    "etfs": [
        "SPY", "QQQ", "IWM", "DIA", "XLK",
        "XLF", "XLE", "XLV", "XLY", "XLI"
    ]
}

TICKERS = sorted(set(
    ticker
    for tickers in UNIVERSES.values()
    for ticker in tickers
))

def get_universe_for_ticker(ticker: str) -> str:
    matching_universes = []

    for universe_name, tickers in UNIVERSES.items():
        if ticker in tickers:
            matching_universes.append(universe_name)

    if not matching_universes:
        return "unknown"

    return ", ".join(matching_universes)

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
            "universe": get_universe_for_ticker(ticker),
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

def calculate_reversal_score(row: pd.Series) -> float:
    """
    Scores potential reversal setups.

    A reversal setup means:
    - Stock has dropped recently
    - RSI is weak/oversold
    - Volume is starting to come in
    - Liquidity is acceptable

    This is different from momentum.
    Momentum = already strong and continuing
    Reversal = beaten down and possibly bouncing
    """

    score = 0

    return_30d = row["return_30d_pct"]
    rsi = row["rsi_14"]
    volume_spike = row["volume_spike"]
    avg_volume = row["avg_volume_30d"]
    price = row["latest_price"]
    volatility = row["volatility_30d_pct"]

    # 1. Recent selloff
    # Bigger drop = more reversal opportunity
    if pd.isna(return_30d):
        score += 0
    elif return_30d <= -25:
        score += 35
    elif return_30d <= -15:
        score += 28
    elif return_30d <= -8:
        score += 18
    elif return_30d <= -3:
        score += 8

    # 2. RSI oversold / weak
    if pd.isna(rsi):
        score += 0
    elif rsi < 30:
        score += 30
    elif rsi < 40:
        score += 22
    elif rsi < 45:
        score += 10

    # 3. Volume confirmation
    # Reversal without volume is weak
    if pd.isna(volume_spike):
        score += 0
    elif volume_spike >= 2:
        score += 25
    elif volume_spike >= 1.5:
        score += 18
    elif volume_spike >= 1.2:
        score += 10

    # 4. Liquidity filter
    # Avoid dead stocks
    if avg_volume >= 1_000_000:
        score += 10
    elif avg_volume >= 500_000:
        score += 7
    elif avg_volume >= 250_000:
        score += 3

    # 5. Avoid extremely dangerous setups
    if price < 3:
        score -= 10

    if volatility > 12:
        score -= 10

    return clamp(score)


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

        # Momentum-style growth score
    momentum_growth_score = (
            0.30 * momentum_score +
            0.20 * volume_score +
            0.15 * price_score +
            0.15 * risk_score +
            0.20 * row["technical_score"]
    )

    # New reversal score
    reversal_score = calculate_reversal_score(row)

    # Final setup score
    # We take the stronger setup, not the average.
    # Because momentum and reversal are different strategies.
    setup_score = max(momentum_growth_score, reversal_score)

    return pd.Series({
        "momentum_score": round(momentum_score, 2),
        "volume_score": round(volume_score, 2),
        "price_score": round(price_score, 2),
        "risk_score": round(risk_score, 2),

        # Existing score, now renamed conceptually as momentum growth score
        "growth_score": round(momentum_growth_score, 2),

        # New score
        "reversal_score": round(reversal_score, 2),

        # Best available setup score
        "setup_score": round(setup_score, 2),
    })

def generate_reason(row: pd.Series) -> str:
    reasons = []

    setup_type = row.get("setup_type", "Neutral")

    if setup_type == "Momentum":
        if row["return_30d_pct"] > 10:
            reasons.append("strong 30D momentum")
        elif row["return_30d_pct"] > 0:
            reasons.append("positive 30D momentum")

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

    elif setup_type == "Reversal":
        if row["return_30d_pct"] <= -15:
            reasons.append("sharp 30D selloff")
        elif row["return_30d_pct"] <= -8:
            reasons.append("recent weakness may create rebound setup")

        if row["rsi_14"] < 30:
            reasons.append("oversold RSI")
        elif row["rsi_14"] < 40:
            reasons.append("weak RSI near reversal zone")

        if row["volume_spike"] >= 1.5:
            reasons.append("volume spike may confirm reversal interest")
        elif row["volume_spike"] >= 1.2:
            reasons.append("moderate volume support")

    else:
        reasons.append("no strong momentum or reversal setup")

    return ", ".join(reasons)

def assign_setup_type(row: pd.Series) -> str:
    """
    Identifies whether the stock is a momentum setup,
    reversal setup, or neutral.
    """

    if row["growth_score"] >= 70 and row["growth_score"] >= row["reversal_score"]:
        return "Momentum"

    elif row["reversal_score"] >= 65 and row["reversal_score"] > row["growth_score"]:
        return "Reversal"

    else:
        return "Neutral"


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

    if row["volatility_30d_pct"] > 12:
        risks.append("extremely high volatility")
    elif row["volatility_30d_pct"] > 8:
        risks.append("very high volatility")
    elif row["volatility_30d_pct"] > 5:
        risks.append("elevated volatility")

    if row["setup_type"] == "Momentum" and row["rsi_14"] > 75:
        risks.append("possible short-term overextension")

    if row["setup_type"] == "Reversal":
        risks.append("reversal setup is higher risk and needs confirmation")

        if row["return_30d_pct"] < -25:
            risks.append("stock is in a deep selloff")

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

    # Assign setup type
    df["setup_type"] = df.apply(assign_setup_type, axis=1)

    # Sort by best setup score instead of only growth score
    df = df.sort_values(by="setup_score", ascending=False)

    # Decision now uses setup_score
    df["decision"] = df["setup_score"].apply(assign_decision)

    # Generate explanation and risk note
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