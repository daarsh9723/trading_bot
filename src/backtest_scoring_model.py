import yfinance as yf
import pandas as pd
from pathlib import Path

from stock_screener import (
    TICKERS,
    clamp,
    calculate_reversal_score,
    assign_setup_type,
    assign_decision,
    get_universe_for_ticker,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


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


def calculate_historical_features(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()

    df["latest_price"] = df["Close"]

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

    # What happened after the signal.
    df["future_return_30d"] = (df["Close"].shift(-30) / df["Close"] - 1) * 100

    return df


def calculate_setup_scores(row: pd.Series) -> pd.Series:
    """
    Recreates the same scoring logic used in the screener,
    but applies it historically for every date.
    """

    # 1. Momentum score
    return_30d = row["return_30d_pct"]

    if pd.isna(return_30d):
        momentum_score = 0
    else:
        momentum_score = clamp((return_30d + 10) * 3)

    # 2. Volume score
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

    # 3. Price score
    # Flat — price level alone is not an edge signal.
    # Very low-priced stocks are flagged in risk_note instead.
    price = row["latest_price"]

    if price <= 0:
        price_score = 0
    else:
        price_score = 50

    # 4. Risk score
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

    # Momentum-style score
    growth_score = (
        0.30 * momentum_score
        + 0.20 * volume_score
        + 0.15 * price_score
        + 0.15 * risk_score
        + 0.20 * row["technical_score"]
    )

    # Reversal-style score
    reversal_score = calculate_reversal_score(row)

    # Best available setup score
    setup_score = max(growth_score, reversal_score)

    return pd.Series({
        "momentum_score": round(momentum_score, 2),
        "volume_score": round(volume_score, 2),
        "price_score": round(price_score, 2),
        "risk_score": round(risk_score, 2),
        "growth_score": round(growth_score, 2),
        "reversal_score": round(reversal_score, 2),
        "setup_score": round(setup_score, 2),
    })


def summarize_future_returns(group_cols, results: pd.DataFrame) -> pd.DataFrame:
    """
    Standard summary table used by all grouped backtest reports.
    Keeping this in one function prevents mismatched summary logic.
    """

    return results.groupby(group_cols).agg(
        signals=("future_return_30d", "count"),
        avg_future_return_30d=("future_return_30d", "mean"),
        median_future_return_30d=("future_return_30d", "median"),
        win_rate=("future_return_30d", lambda x: (x > 0).mean() * 100),
        beat_5pct_rate=("future_return_30d", lambda x: (x > 5).mean() * 100),
        beat_10pct_rate=("future_return_30d", lambda x: (x > 10).mean() * 100),
        max_loss=("future_return_30d", "min"),
        max_gain=("future_return_30d", "max"),
        avg_setup_score=("setup_score", "mean"),
        avg_growth_score=("growth_score", "mean"),
        avg_reversal_score=("reversal_score", "mean"),
    ).reset_index()


def print_sanity_check(results: pd.DataFrame, label: str) -> None:
    print(f"\nBacktest sanity check - {label}:")
    print("Rows:", len(results))
    print("Tickers:", results["ticker"].nunique())
    print("Universes:", results["universe"].nunique())
    print("Ticker list:")
    print(sorted(results["ticker"].unique()))


def run_setup_backtest():
    all_results = []

    print("\nTicker universe loaded from stock_screener.py")
    print("Tickers expected:", len(TICKERS))
    print(sorted(TICKERS))

    for ticker in TICKERS:
        print(f"Backtesting setup model for {ticker}...")

        hist = yf.Ticker(ticker).history(period="3y")

        if hist.empty:
            print(f"No data for {ticker}")
            continue

        df = calculate_historical_features(hist)
        df["ticker"] = ticker
        df["universe"] = get_universe_for_ticker(ticker)

        score_columns = df.apply(calculate_setup_scores, axis=1)
        df = pd.concat([df, score_columns], axis=1)

        df["setup_type"] = df.apply(assign_setup_type, axis=1)
        df["decision"] = df["setup_score"].apply(assign_decision)

        all_results.append(df)

    if not all_results:
        print("No backtest data found.")
        return

    results = pd.concat(all_results)
    print_sanity_check(results, "before cleaning")

    # Cleaning step: remove rows that cannot be evaluated properly.
    results = results.dropna(subset=[
        "return_30d_pct",
        "volatility_30d_pct",
        "rsi_14",
        "future_return_30d",
        "growth_score",
        "reversal_score",
        "setup_score",
        "setup_type",
        "decision",
        "universe",
    ]).copy()

    print_sanity_check(results, "after cleaning")

    DATA_DIR.mkdir(exist_ok=True)

    # 1. Summary by setup type.
    setup_summary = summarize_future_returns(
        ["setup_type"],
        results,
    ).sort_values(by="avg_future_return_30d", ascending=False)

    # 2. Summary by setup type + decision.
    setup_decision_summary = summarize_future_returns(
        ["setup_type", "decision"],
        results,
    ).sort_values(
        by=["setup_type", "avg_future_return_30d"],
        ascending=[True, False],
    )

    # 3. Score decile summary.
    results["setup_score_decile"] = pd.qcut(
        results["setup_score"],
        10,
        labels=False,
        duplicates="drop",
    )

    decile_summary = summarize_future_returns(
        ["setup_score_decile"],
        results,
    ).sort_values(by="setup_score_decile")

    # 4. Summary by universe + setup type.
    universe_summary = summarize_future_returns(
        ["universe", "setup_type"],
        results,
    ).sort_values(
        by=["universe", "avg_future_return_30d"],
        ascending=[True, False],
    )

    # 5. Summary by universe + setup type + decision.
    # This is the new file we need before making universe-specific threshold decisions.
    universe_decision_summary = summarize_future_returns(
        ["universe", "setup_type", "decision"],
        results,
    ).sort_values(
        by=["universe", "setup_type", "avg_future_return_30d"],
        ascending=[True, True, False],
    )

    # Save everything from the same cleaned expanded results DataFrame.
    results.to_csv(DATA_DIR / "historical_setup_backtest.csv", index=True)
    setup_summary.to_csv(DATA_DIR / "setup_type_backtest_summary.csv", index=False)
    setup_decision_summary.to_csv(DATA_DIR / "setup_decision_backtest_summary.csv", index=False)
    decile_summary.to_csv(DATA_DIR / "setup_score_decile_summary.csv", index=False)
    universe_summary.to_csv(DATA_DIR / "universe_setup_backtest_summary.csv", index=False)
    universe_decision_summary.to_csv(DATA_DIR / "universe_decision_backtest_summary.csv", index=False)

    print("\nSetup Type Summary:")
    print(setup_summary)

    print("\nSetup + Decision Summary:")
    print(setup_decision_summary)

    print("\nSetup Score Decile Summary:")
    print(decile_summary)

    print("\nUniverse + Setup Type Summary:")
    print(universe_summary)

    print("\nUniverse + Setup Type + Decision Summary:")
    print(universe_decision_summary)

    print("\nSaved:")
    print(DATA_DIR / "historical_setup_backtest.csv")
    print(DATA_DIR / "setup_type_backtest_summary.csv")
    print(DATA_DIR / "setup_decision_backtest_summary.csv")
    print(DATA_DIR / "setup_score_decile_summary.csv")
    print(DATA_DIR / "universe_setup_backtest_summary.csv")
    print(DATA_DIR / "universe_decision_backtest_summary.csv")


if __name__ == "__main__":
    run_setup_backtest()