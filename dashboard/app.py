import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px


st.set_page_config(
    page_title="Quant Stock Screener",
    page_icon="📈",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def get_latest_screener_file():
    files = list(DATA_DIR.glob("stock_screener_*.csv"))

    if not files:
        return None

    latest_file = max(files, key=lambda file: file.stat().st_mtime)
    return latest_file


def load_data():
    latest_file = get_latest_screener_file()

    if latest_file is None:
        return None, None

    df = pd.read_csv(latest_file)
    return df, latest_file


def style_setup_type(setup_type):
    if setup_type == "Momentum":
        return "🚀 Momentum"
    elif setup_type == "Reversal":
        return "🔄 Reversal"
    else:
        return "⚪ Neutral"


def style_decision(decision):
    if decision == "Strong Watch":
        return "🔥 Strong Watch"
    elif decision == "Watch":
        return "👀 Watch"
    elif decision == "Weak Watch":
        return "⚠️ Weak Watch"
    else:
        return "❌ Ignore"


df, latest_file = load_data()

st.title("📈 Quant Stock Screener Dashboard")

if df is None:
    st.error("No screener CSV found. Run your screener first.")
    st.stop()

st.caption(f"Loaded file: `{latest_file.name}`")

required_columns = [
    "ticker",
    "latest_price",
    "return_30d_pct",
    "volume_spike",
    "volatility_30d_pct",
    "rsi_14",
    "growth_score",
    "reversal_score",
    "setup_score",
    "setup_type",
    "decision",
    "reason",
    "risk_note"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("Your CSV is missing these columns:")
    st.write(missing_columns)
    st.info("Run the updated screener with reversal_score and setup_score first.")
    st.stop()


# Clean display labels
df["setup_label"] = df["setup_type"].apply(style_setup_type)
df["decision_label"] = df["decision"].apply(style_decision)

# Sort by new master score
df = df.sort_values(by="setup_score", ascending=False)


# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("Filters")

setup_filter = st.sidebar.multiselect(
    "Setup Type",
    options=sorted(df["setup_type"].unique()),
    default=sorted(df["setup_type"].unique())
)

decision_filter = st.sidebar.multiselect(
    "Decision",
    options=sorted(df["decision"].unique()),
    default=sorted(df["decision"].unique())
)

min_score = st.sidebar.slider(
    "Minimum Setup Score",
    min_value=0,
    max_value=100,
    value=50
)

max_volatility = st.sidebar.slider(
    "Maximum 30D Volatility %",
    min_value=0.0,
    max_value=float(max(df["volatility_30d_pct"].max(), 1)),
    value=float(max(df["volatility_30d_pct"].max(), 1))
)

filtered_df = df[
    (df["setup_type"].isin(setup_filter)) &
    (df["decision"].isin(decision_filter)) &
    (df["setup_score"] >= min_score) &
    (df["volatility_30d_pct"] <= max_volatility)
].copy()


# =========================
# TOP METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Stocks Screened", len(df))

with col2:
    st.metric("Filtered Results", len(filtered_df))

with col3:
    strong_watch_count = len(filtered_df[filtered_df["decision"] == "Strong Watch"])
    st.metric("Strong Watch", strong_watch_count)

with col4:
    reversal_count = len(filtered_df[filtered_df["setup_type"] == "Reversal"])
    st.metric("Reversal Setups", reversal_count)


# =========================
# MAIN TABLE
# =========================

st.subheader("Ranked Candidates")

display_columns = [
    "ticker",
    "latest_price",
    "setup_score",
    "growth_score",
    "reversal_score",
    "setup_label",
    "decision_label",
    "return_30d_pct",
    "volume_spike",
    "volatility_30d_pct",
    "rsi_14",
    "reason",
    "risk_note"
]

display_df = filtered_df[display_columns].copy()

display_df = display_df.rename(columns={
    "ticker": "Ticker",
    "latest_price": "Price",
    "setup_score": "Setup Score",
    "growth_score": "Momentum Score",
    "reversal_score": "Reversal Score",
    "setup_label": "Setup Type",
    "decision_label": "Decision",
    "return_30d_pct": "30D Return %",
    "volume_spike": "Volume Spike",
    "volatility_30d_pct": "30D Volatility %",
    "rsi_14": "RSI 14",
    "reason": "Reason",
    "risk_note": "Risk Note"
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# =========================
# TOP CANDIDATE CARD
# =========================

st.subheader("Top Candidate")

if filtered_df.empty:
    st.warning("No stocks match the selected filters.")
else:
    top = filtered_df.iloc[0]

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.metric("Ticker", top["ticker"])
        st.metric("Price", f"${top['latest_price']:.2f}")

    with c2:
        st.metric("Setup Score", f"{top['setup_score']:.2f}")
        st.metric("Setup Type", style_setup_type(top["setup_type"]))

    with c3:
        st.write(f"**Decision:** {style_decision(top['decision'])}")
        st.write(f"**Reason:** {top['reason']}")
        st.write(f"**Risk:** {top['risk_note']}")


# =========================
# CHARTS
# =========================

st.subheader("Score Comparison")

if not filtered_df.empty:
    score_chart_df = filtered_df.sort_values(by="setup_score", ascending=False)

    fig = px.bar(
        score_chart_df,
        x="ticker",
        y=["growth_score", "reversal_score", "setup_score"],
        barmode="group",
        title="Momentum Score vs Reversal Score vs Setup Score"
    )

    st.plotly_chart(fig, use_container_width=True)


st.subheader("Risk vs Return Map")

if not filtered_df.empty:
    fig2 = px.scatter(
        filtered_df,
        x="volatility_30d_pct",
        y="return_30d_pct",
        size="setup_score",
        color="setup_type",
        hover_name="ticker",
        title="30D Return vs 30D Volatility",
        labels={
            "volatility_30d_pct": "30D Volatility %",
            "return_30d_pct": "30D Return %",
            "setup_score": "Setup Score",
            "setup_type": "Setup Type"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)


# =========================
# SETUP BREAKDOWN
# =========================

st.subheader("Setup Type Breakdown")

setup_counts = filtered_df["setup_type"].value_counts().reset_index()
setup_counts.columns = ["Setup Type", "Count"]

fig3 = px.pie(
    setup_counts,
    names="Setup Type",
    values="Count",
    title="Momentum vs Reversal vs Neutral"
)

st.plotly_chart(fig3, use_container_width=True)


# =========================
# DOWNLOAD
# =========================

st.subheader("Download Results")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered CSV",
    data=csv,
    file_name="filtered_quant_screener_results.csv",
    mime="text/csv"
)