from pathlib import Path
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


st.set_page_config(
    page_title="Trading Bot Screener",
    page_icon="📈",
    layout="wide"
)


st.title("📈 Under-$30 Stock Screener")
st.write("Day 2 dashboard for your AI-assisted trading bot project.")


def get_latest_csv():
    csv_files = list(DATA_DIR.glob("stock_screener_*.csv"))

    if not csv_files:
        return None

    latest_file = max(csv_files, key=lambda file: file.stat().st_mtime)
    return latest_file


latest_csv = get_latest_csv()

if latest_csv is None:
    st.warning("No screener CSV found. Run `python src/stock_screener.py` first.")
    st.stop()


df = pd.read_csv(latest_csv)

st.caption(f"Loaded file: `{latest_csv.name}`")


# Sidebar filters
st.sidebar.header("Filters")

min_score = st.sidebar.slider(
    "Minimum Growth Score",
    min_value=0,
    max_value=100,
    value=50
)

max_price = st.sidebar.slider(
    "Maximum Stock Price",
    min_value=1,
    max_value=30,
    value=30
)

min_volume = st.sidebar.number_input(
    "Minimum 30D Average Volume",
    min_value=0,
    value=500_000,
    step=100_000
)


filtered_df = df[
    (df["growth_score"] >= min_score) &
    (df["latest_price"] <= max_price) &
    (df["avg_volume_30d"] >= min_volume)
].copy()


# Summary cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Stocks Loaded", len(df))
col2.metric("Stocks After Filter", len(filtered_df))

if not filtered_df.empty:
    col3.metric("Top Score", filtered_df["growth_score"].max())
    col4.metric("Average Price", round(filtered_df["latest_price"].mean(), 2))
else:
    col3.metric("Top Score", "N/A")
    col4.metric("Average Price", "N/A")


st.subheader("Ranked Candidates")

if filtered_df.empty:
    st.error("No stocks match your filters.")
else:
    filtered_df = filtered_df.sort_values(by="growth_score", ascending=False)

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


st.subheader("Top 5 Candidates")

top_5 = filtered_df.head(5)

if not top_5.empty:
    st.table(
        top_5[
            [
                "ticker",
                "latest_price",
                "return_30d_pct",
                "avg_volume_30d",
                "volume_spike",
                "volatility_30d_pct",
                "growth_score"
            ]
        ]
    )
else:
    st.info("No top candidates available.")