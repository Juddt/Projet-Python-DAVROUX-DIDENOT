import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

time_zone_france = pytz.timezone("Europe/Paris")

#auto refresh every 5 min
st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

#page config, keep here only if not in app.py
try:
    st.set_page_config(page_title="single asset analysis module", layout="wide")
except Exception:
    pass

different_timeframes = {"1d": {"period": "1d", "interval": "5m"}, "5d": {"period": "5d", "interval": "15m"}, "1m": {"period": "1mo", "interval": "1h"}, "6m": {"period": "6mo", "interval": "1d"}, "1y": {"period": "1y", "interval": "1d"}, "5y": {"period": "5y", "interval": "1wk"}, "max": {"period": "max", "interval": "1wk"}}

sma_values = {"1d": {"fast": 9, "slow": 21}, "5d": {"fast": 10, "slow": 50}, "1m": {"fast": 20, "slow": 50}, "6m": {"fast": 50, "slow": 100}, "1y": {"fast": 50, "slow": 150}, "5y": {"fast": 50, "slow": 200}, "max": {"fast": 50, "slow": 200}}

window_rsi = {"1d": 7, "5d": 10, "1m": 14, "6m": 14, "1y": 21, "5y": 21, "max": 21}

number_days = {"1d": 1, "5d": 5, "1m": 30, "6m": 180, "1y": 365, "5y": 5 * 365}

strategies_labels = {"Buy & Hold": "Buy & Hold", "SMA Crossover": "SMA Crossover", "SMA + RSI": "SMA + RSI", "Simple Momentum": "Simple Momentum"}

st.title("single asset analysis module")
st.caption("last refresh: " + datetime.now(time_zone_france).strftime("%Y-%m-%d %H:%M:%S"))

st.sidebar.header("User Settings")
st.sidebar.caption("1 - Choose an asset")
st.sidebar.caption("2 - Set initial capital")
st.sidebar.caption("3 - Select a timeframe")
st.sidebar.caption("4 - Pick a strategy")
st.sidebar.caption("5 - Adjust parameters")

ticker = st.sidebar.text_input("Ticker", value="AAPL")
ticker = ticker.replace(",", " ").split()[0].upper()
capital = st.sidebar.number_input("Initial capital in $", 10, 100000, 100, step=1)

timeframe = st.sidebar.radio("Timeframe", options=list(different_timeframes.keys()), horizontal=True, index=3)
timeframes_config = different_timeframes[timeframe]
rsi_window_default = window_rsi[timeframe]

strategies_selected = st.sidebar.radio("Strategies", ["Buy & Hold", "SMA Crossover", "SMA + RSI", "Simple Momentum", "Strategy Comparaison"], index=0)
strategies_using_sma = strategies_selected in ["SMA Crossover", "SMA + RSI"]

momentum_parameter_value = 20
if strategies_selected == "Simple Momentum":
    with st.sidebar.expander("Momentum parameters", expanded=True):
        momentum_parameter_value = st.slider("Momentum lookback (number of periods)", min_value=5, max_value=200, value=20)

sma_fast_default = sma_values[timeframe]["fast"]
sma_slow_default = sma_values[timeframe]["slow"]

if strategies_using_sma:
    with st.sidebar.expander("Moving Average parameters", expanded=True):
        manual_sma = st.checkbox("Manual SMA")
        if manual_sma:
            sma_fast = st.slider("Fast SMA", min_value=5, max_value=100, value=int(sma_fast_default))
            sma_slow = st.slider("Slow SMA", min_value=20, max_value=300, value=int(sma_slow_default))
        else:
            sma_fast = int(sma_fast_default)
            sma_slow = int(sma_slow_default)
            st.caption(f"Automatic values for {timeframe} timeframe\n Fast SMA: {sma_fast}\n Slow SMA: {sma_slow}")
else:
    sma_fast = int(sma_fast_default)
    sma_slow = int(sma_slow_default)

if strategies_using_sma and int(sma_fast) >= int(sma_slow):
    st.warning("Wrong SMA values.\n\nFast SMA must be lower than Slow SMA.")
    st.stop()

rsi_window_val = 14
rsi_buy = 55
rsi_sell = 70

if strategies_selected == "SMA + RSI":
    with st.sidebar.expander("RSI parameters", expanded=True):
        rsi_window_val = st.slider("RSI window", min_value=5, max_value=30, value=int(rsi_window_default))
        rsi_buy = st.slider("RSI buy threshold", min_value=30, max_value=60, value=55)
        rsi_sell = st.slider("RSI sell threshold", min_value=60, max_value=90, value=70)
else:
    rsi_window_val = int(rsi_window_default)

@st.cache_data(ttl=300)
def download_data(ticker_in: str, period_in: str, interval_in: str) -> pd.DataFrame:
    buffer_map = {"1d": "5d", "5d": "1mo", "1mo": "3mo", "6mo": "1y", "1y": "2y", "5y": "10y", "max": "max"}
    period_large = buffer_map.get(period_in, period_in)

    raw = yf.download(ticker_in, period=period_large, interval=interval_in, progress=False)

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if ("Close" in raw.columns.get_level_values(0)) or ("Close" in raw.columns.get_level_values(1)):
            try:
                raw = raw["Close"]
            except Exception:
                try:
                    raw = raw.xs("Close", level=1, axis=1)
                except Exception:
                    return pd.DataFrame()
        else:
            return pd.DataFrame()

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="Close")

    if isinstance(raw, pd.DataFrame) and "Close" not in raw.columns and raw.shape[1] >= 1:
        raw = raw.iloc[:, 0].to_frame(name="Close")

    if isinstance(raw, pd.DataFrame) and "Close" not in raw.columns:
        return pd.DataFrame()

    df = raw[["Close"]].reset_index()
    df.columns = ["Datetime", "Close"]
    df = df.dropna()
    return df

def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    price_change = prices.diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi_value = 100 - (100 / (1 + relative_strength))
    return rsi_value

def max_drawdown(series: pd.Series) -> float:
    drawdown = (series - series.cummax()) / series.cummax()
    return float(drawdown.min())

annual_factor = {"5m": 252 * 78, "15m": 252 * 26, "1h": 252 * 7, "1d": 252, "1wk": 52}
factor = float(np.sqrt(annual_factor.get(timeframes_config["interval"], 252)))

def sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * factor)

def volatility(returns: pd.Series) -> float:
    if returns.std() == 0:
        return float("nan")
    return float(returns.std() * factor)

with st.spinner("Downloading data"):
    data = download_data(ticker, timeframes_config["period"], timeframes_config["interval"])

if data is None or data.empty:
    st.error("No data")
    st.stop()

last_close_raw = float(data["Close"].iloc[-1])
try:
    currency = yf.Ticker(ticker).info.get("currency", "USD")
except Exception:
    currency = "USD"

st.markdown(f"## {ticker} — {last_close_raw:.2f} {currency}", unsafe_allow_html=False)
st.caption(f"Last update: {datetime.now(time_zone_france).strftime('%H:%M:%S')}")

price = data.copy()

price["return"] = price["Close"].pct_change()
price["sma_fast"] = price["Close"].rolling(int(sma_fast)).mean()
price["sma_slow"] = price["Close"].rolling(int(sma_slow)).mean()
price["rsi"] = rsi(price["Close"], int(rsi_window_val))

position = np.where(price["sma_fast"] > price["sma_slow"], 1, 0)
price["Position"] = pd.Series(position, index=price.index).shift(1).fillna(0)
price["SMA_Crossover"] = ((1 + price["return"] * price["Position"]).fillna(1).cumprod() * float(capital))

price["Momentum_computed"] = price["Close"].pct_change(periods=int(momentum_parameter_value))
position_momentum = np.where(price["Momentum_computed"] > 0.005, 1, 0)
price["Position_Momentum"] = pd.Series(position_momentum, index=price.index).shift(1).fillna(0)
price["Simple_Momentum"] = ((1 + price["return"] * price["Position_Momentum"]).fillna(1).cumprod() * float(capital))

signal_sma_rsi = np.where((price["sma_fast"] > price["sma_slow"]) & (price["rsi"] < float(rsi_buy)), 1, np.where(price["rsi"] > float(rsi_sell), 0, np.nan))
price["Position_SMA_RSI"] = pd.Series(signal_sma_rsi, index=price.index).ffill().shift(1).fillna(0)

price["Buy_and_Hold"] = price["Close"] / float(price["Close"].iloc[0]) * float(capital)

if timeframe in number_days:
    last_dt = price["Datetime"].max()
    start_dt = last_dt - pd.Timedelta(days=int(number_days[timeframe]))
    price = price[price["Datetime"] >= start_dt].copy()

price["Buy_Signal"] = ((price["Position_SMA_RSI"] == 1) & (price["Position_SMA_RSI"].shift(1) == 0))
price["Sell_Signal"] = ((price["Position_SMA_RSI"] == 0) & (price["Position_SMA_RSI"].shift(1) == 1))

price["SMA_RSI"] = ((1 + price["return"] * price["Position_SMA_RSI"]).fillna(1).cumprod() * float(capital))

for col in ["Buy_and_Hold", "SMA_Crossover", "SMA_RSI", "Simple_Momentum"]:
    if price[col].notna().any():
        price[col] = price[col] / float(price[col].iloc[0]) * float(capital)

metrics = pd.DataFrame({"Buy & Hold": [price["Buy_and_Hold"].iloc[-1] / float(capital) - 1, sharpe(price["return"].fillna(0)), volatility(price["return"].fillna(0)), max_drawdown(price["Buy_and_Hold"])], "SMA Crossover": [price["SMA_Crossover"].iloc[-1] / float(capital) - 1, sharpe((price["return"] * price["Position"]).fillna(0)), volatility((price["return"] * price["Position"]).fillna(0)), max_drawdown(price["SMA_Crossover"])], "Simple Momentum": [price["Simple_Momentum"].iloc[-1] / float(capital) - 1, sharpe((price["return"] * price["Position_Momentum"]).fillna(0)), volatility((price["return"] * price["Position_Momentum"]).fillna(0)), max_drawdown(price["Simple_Momentum"])], "SMA + RSI": [price["SMA_RSI"].iloc[-1] / float(capital) - 1, sharpe((price["return"] * price["Position_SMA_RSI"]).fillna(0)), volatility((price["return"] * price["Position_SMA_RSI"]).fillna(0)), max_drawdown(price["SMA_RSI"])]}, index=["Total Return", "Sharpe Ratio", "Volatility", "Max Drawdown"]).round(3)

performance_metrics = metrics.loc[["Total Return", "Max Drawdown"]].copy()
risk_metrics = metrics.loc[["Sharpe Ratio", "Volatility"]].copy()

final_values = pd.DataFrame({"Final Value ($)": [price["Buy_and_Hold"].iloc[-1], price["SMA_Crossover"].iloc[-1], price["SMA_RSI"].iloc[-1], price["Simple_Momentum"].iloc[-1]]}, index=["Buy & Hold", "SMA Crossover", "SMA + RSI", "Simple Momentum"]).round(2)

if strategies_selected == "Strategy Comparaison":
    st.subheader("Strategy Comparaison")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(price["Datetime"], price["Buy_and_Hold"], label="Buy & Hold", linewidth=2)
    ax.plot(price["Datetime"], price["SMA_Crossover"], label="SMA Crossover")
    ax.plot(price["Datetime"], price["SMA_RSI"], label="SMA + RSI")
    ax.plot(price["Datetime"], price["Simple_Momentum"], label="Simple Momentum")
    ax.axhline(float(capital), linestyle="--", alpha=0.6)
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Performance Metrics")
    st.dataframe(performance_metrics.style.format("{:.2%}"), use_container_width=True)
    st.markdown("### Risk Metrics")
    st.dataframe(risk_metrics.style.format("{:.2f}"), use_container_width=True)
    st.markdown("### Final Portfolio Values")
    st.dataframe(final_values, use_container_width=True)
else:
    selected_col = strategies_labels[strategies_selected]
    charts, tables = st.columns([2, 1])

    with charts:
        if strategies_selected == "Buy & Hold":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(price["Datetime"], price["Close"])
            ax1.set_title("Asset price")
            ax1.grid(alpha=0.3)
            ax2.plot(price["Datetime"], price["Buy_and_Hold"])
            ax2.axhline(float(capital), linestyle="--", alpha=0.6)
            ax2.set_title("Buy & Hold performance")
            ax2.grid(alpha=0.3)
            st.pyplot(fig)

        elif strategies_selected == "SMA Crossover":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(price["Datetime"], price["Close"], label="Price")
            ax1.plot(price["Datetime"], price["sma_fast"], "--", label="Fast SMA")
            ax1.plot(price["Datetime"], price["sma_slow"], "--", label="Slow SMA")
            ax1.set_title("Price and SMAs")
            ax1.grid(alpha=0.3)
            ax1.legend()
            ax2.plot(price["Datetime"], price["SMA_Crossover"])
            ax2.axhline(float(capital), linestyle="--", alpha=0.6)
            ax2.set_title("SMA Crossover performance")
            ax2.grid(alpha=0.3)
            st.pyplot(fig)

        elif strategies_selected == "SMA + RSI":
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
            ax1.plot(price["Datetime"], price["Close"], label="Price")
            ax1.plot(price["Datetime"], price["sma_fast"], "--", label="Fast SMA")
            ax1.plot(price["Datetime"], price["sma_slow"], "--", label="Slow SMA")
            ax1.scatter(price.loc[price["Buy_Signal"], "Datetime"], price.loc[price["Buy_Signal"], "Close"], marker="^", s=80, label="Buy")
            ax1.scatter(price.loc[price["Sell_Signal"], "Datetime"], price.loc[price["Sell_Signal"], "Close"], marker="v", s=80, label="Sell")
            ax1.set_title("Price + SMAs + Buy/Sell signals")
            ax1.grid(alpha=0.3)
            ax1.legend()
            ax2.plot(price["Datetime"], price["rsi"])
            ax2.axhline(float(rsi_buy), linestyle="--")
            ax2.axhline(float(rsi_sell), linestyle="--")
            ax2.set_ylim(0, 100)
            ax2.set_title("RSI")
            ax2.grid(alpha=0.3)
            ax3.plot(price["Datetime"], price["SMA_RSI"])
            ax3.axhline(float(capital), linestyle="--", alpha=0.6)
            ax3.set_title("SMA + RSI performance")
            ax3.grid(alpha=0.3)
            st.pyplot(fig)

        elif strategies_selected == "Simple Momentum":
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
            ax1.plot(price["Datetime"], price["Close"])
            ax1.set_title("Price")
            ax1.grid(alpha=0.3)
            ax2.plot(price["Datetime"], price["Momentum_computed"])
            ax2.axhline(0, linestyle="--", alpha=0.6)
            ax2.set_title("Momentum")
            ax2.grid(alpha=0.3)
            ax3.plot(price["Datetime"], price["Simple_Momentum"])
            ax3.axhline(float(capital), linestyle="--", alpha=0.6)
            ax3.set_title("Momentum strategy performance")
            ax3.grid(alpha=0.3)
            st.pyplot(fig)

    with tables:
        st.subheader("Performance")
        st.dataframe(performance_metrics[[selected_col]].style.format("{:.2%}"), use_container_width=True)
        st.subheader("Risk")
        st.dataframe(risk_metrics[[selected_col]].style.format("{:.2f}"), use_container_width=True)
        st.subheader("Final Value")
        st.dataframe(final_values.loc[[selected_col]], use_container_width=True)
