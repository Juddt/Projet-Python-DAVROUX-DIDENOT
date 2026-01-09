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
    st.set_page_config(page_title="quant fusion multi assets", layout="wide")
except Exception:
    pass

different_timeframes = {"1d": {"period": "1d", "interval": "5m"}, "5d": {"period": "5d", "interval": "15m"}, "1m": {"period": "1mo", "interval": "1h"}, "6m": {"period": "6mo", "interval": "1d"}, "1y": {"period": "1y", "interval": "1d"}, "5y": {"period": "5y", "interval": "1wk"}, "max": {"period": "max", "interval": "1wk"}}

sma_values = {"1d": {"fast": 9, "slow": 21}, "5d": {"fast": 10, "slow": 50}, "1m": {"fast": 20, "slow": 50}, "6m": {"fast": 50, "slow": 100}, "1y": {"fast": 50, "slow": 150}, "5y": {"fast": 50, "slow": 200}, "max": {"fast": 50, "slow": 200}}

window_rsi = {"1d": 7, "5d": 10, "1m": 14, "6m": 14, "1y": 21, "5y": 21, "max": 21}

number_days = {"1d": 1, "5d": 5, "1m": 30, "6m": 180, "1y": 365, "5y": 5 * 365}

annual_factor = {"5m": 252 * 78, "15m": 252 * 26, "1h": 252 * 7, "1d": 252, "1wk": 52}

st.title("quant fusion multi assets")
st.caption("last refresh: " + datetime.now(time_zone_france).strftime("%Y-%m-%d %H:%M:%S"))

st.sidebar.header("User Settings")
st.sidebar.caption("portfolio + strategies on multiple assets")

default_tickers = "AAPL,MSFT,GOOGL,AMZN,META"
tickers_raw = st.sidebar.text_input("tickers (comma separated)", value=default_tickers)
tickers = [t.strip().upper() for t in tickers_raw.replace(";", ",").split(",") if t.strip() != ""]

capital = st.sidebar.number_input("Initial capital in $", 10, 100000, 100, step=1)

timeframe = st.sidebar.radio("Timeframe", options=list(different_timeframes.keys()), horizontal=True, index=3)
timeframes_config = different_timeframes[timeframe]

mode = st.sidebar.radio("Mode", ["Single strategy", "Strategy Comparaison"], index=1)

strategies_selected = st.sidebar.radio("Strategies", ["Buy & Hold", "SMA Crossover", "SMA + RSI", "Simple Momentum"], index=0, disabled=(mode == "Strategy Comparaison"))
strategies_using_sma = (mode == "Strategy Comparaison") or (strategies_selected in ["SMA Crossover", "SMA + RSI"])

momentum_parameter_value = 20
with st.sidebar.expander("Parameters", expanded=True):
    sma_fast_default = sma_values[timeframe]["fast"]
    sma_slow_default = sma_values[timeframe]["slow"]
    sma_fast = st.slider("Fast SMA", min_value=5, max_value=100, value=int(sma_fast_default))
    sma_slow = st.slider("Slow SMA", min_value=20, max_value=300, value=int(sma_slow_default))
    rsi_window_val = st.slider("RSI window", min_value=5, max_value=30, value=int(window_rsi[timeframe]))
    rsi_buy = st.slider("RSI buy threshold", min_value=30, max_value=60, value=55)
    rsi_sell = st.slider("RSI sell threshold", min_value=60, max_value=90, value=70)
    momentum_parameter_value = st.slider("Momentum lookback (periods)", min_value=5, max_value=200, value=int(momentum_parameter_value))

if strategies_using_sma and int(sma_fast) >= int(sma_slow):
    st.error("Fast SMA must be lower than Slow SMA.")
    st.stop()

@st.cache_data(ttl=300)
def download_multi_close(tickers_in: list[str], period_in: str, interval_in: str) -> pd.DataFrame:
    buffer_map = {"1d": "5d", "5d": "1mo", "1mo": "3mo", "6mo": "1y", "1y": "2y", "5y": "10y", "max": "max"}
    period_large = buffer_map.get(period_in, period_in)

    raw = yf.download(tickers_in, period=period_large, interval=interval_in, progress=False, group_by="ticker", auto_adjust=False)

    if raw is None or isinstance(raw, pd.DataFrame) is False or raw.empty:
        return pd.DataFrame()

    #case 1: multiindex columns -> (field, ticker) or (ticker, field)
    if isinstance(raw.columns, pd.MultiIndex):
        #try common layout: first level is field
        if "Close" in raw.columns.get_level_values(0):
            close_df = raw["Close"].copy()
            close_df.columns = [str(c).upper() for c in close_df.columns]
            close_df = close_df.dropna(how="all")
            close_df = close_df.reset_index().rename(columns={close_df.reset_index().columns[0]: "Datetime"})
            return close_df
        #try other layout: second level is field
        if "Close" in raw.columns.get_level_values(1):
            close_df = raw.xs("Close", level=1, axis=1).copy()
            close_df.columns = [str(c).upper() for c in close_df.columns]
            close_df = close_df.dropna(how="all")
            close_df = close_df.reset_index().rename(columns={close_df.reset_index().columns[0]: "Datetime"})
            return close_df
        return pd.DataFrame()

    #case 2: single ticker df with Close column
    if "Close" in raw.columns:
        close_df = raw[["Close"]].copy()
        close_df.columns = [tickers_in[0].upper()]
        close_df = close_df.dropna(how="all")
        close_df = close_df.reset_index().rename(columns={close_df.reset_index().columns[0]: "Datetime"})
        return close_df

    return pd.DataFrame()

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
    if series is None or series.empty:
        return 0.0
    drawdown = (series - series.cummax()) / series.cummax()
    return float(drawdown.min())

def sharpe_ratio(returns: pd.Series, factor_val: float) -> float:
    if returns is None or returns.empty:
        return float("nan")
    std = float(returns.std())
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(returns.mean() / std * factor_val)

def volatility(returns: pd.Series, factor_val: float) -> float:
    if returns is None or returns.empty:
        return float("nan")
    std = float(returns.std())
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(std * factor_val)

def normalize_weights(sig_row: pd.Series) -> pd.Series:
    s = sig_row.fillna(0.0).astype(float)
    tot = float(s.sum())
    if tot <= 0.0:
        return s * 0.0
    return s / tot

def compute_portfolio_strategies(close_df: pd.DataFrame, capital_val: float) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    x = close_df.copy()
    x = x.sort_values("Datetime")
    asset_cols = [c for c in x.columns if c != "Datetime"]

    #returns by asset
    ret = x[asset_cols].pct_change().fillna(0.0)

    #buy and hold equal weights
    w_bh = pd.Series([1.0 / len(asset_cols)] * len(asset_cols), index=asset_cols)
    port_ret_bh = (ret * w_bh).sum(axis=1)
    val_bh = (1.0 + port_ret_bh).cumprod() * float(capital_val)

    #signals per asset
    sma_fast_df = x[asset_cols].rolling(int(sma_fast)).mean()
    sma_slow_df = x[asset_cols].rolling(int(sma_slow)).mean()
    rsi_df = pd.DataFrame({c: rsi(x[c], int(rsi_window_val)) for c in asset_cols})
    mom_df = x[asset_cols].pct_change(periods=int(momentum_parameter_value))

    #sma crossover: 1 if fast > slow
    sig_sma = (sma_fast_df > sma_slow_df).astype(float)
    w_sma = sig_sma.apply(normalize_weights, axis=1).shift(1).fillna(0.0)
    port_ret_sma = (ret * w_sma).sum(axis=1)
    val_sma = (1.0 + port_ret_sma).cumprod() * float(capital_val)

    #sma + rsi: 1 if trend ok and rsi < buy, 0 if rsi > sell else keep last
    sig_sr = pd.DataFrame(index=x.index, columns=asset_cols, dtype=float)
    for c in asset_cols:
        base_buy = (sma_fast_df[c] > sma_slow_df[c]) & (rsi_df[c] < float(rsi_buy))
        base_sell = (rsi_df[c] > float(rsi_sell))
        sig_sr[c] = np.where(base_buy, 1.0, np.where(base_sell, 0.0, np.nan))
    sig_sr = sig_sr.ffill().fillna(0.0)
    w_sr = sig_sr.apply(normalize_weights, axis=1).shift(1).fillna(0.0)
    port_ret_sr = (ret * w_sr).sum(axis=1)
    val_sr = (1.0 + port_ret_sr).cumprod() * float(capital_val)

    #momentum: 1 if mom > 0, else 0
    sig_m = (mom_df > 0.0).astype(float)
    w_m = sig_m.apply(normalize_weights, axis=1).shift(1).fillna(0.0)
    port_ret_m = (ret * w_m).sum(axis=1)
    val_m = (1.0 + port_ret_m).cumprod() * float(capital_val)

    values = pd.DataFrame({"Buy & Hold": val_bh, "SMA Crossover": val_sma, "SMA + RSI": val_sr, "Simple Momentum": val_m})
    returns_dict = {"Buy & Hold": port_ret_bh, "SMA Crossover": port_ret_sma, "SMA + RSI": port_ret_sr, "Simple Momentum": port_ret_m}
    return values, returns_dict

with st.spinner("Downloading data"):
    close = download_multi_close(tickers, timeframes_config["period"], timeframes_config["interval"])

if close is None or close.empty:
    st.error("No data, check tickers")
    st.stop()

#crop window after buffer
if timeframe in number_days:
    last_dt = close["Datetime"].max()
    start_dt = last_dt - pd.Timedelta(days=int(number_days[timeframe]))
    close = close[close["Datetime"] >= start_dt].copy()

asset_cols_now = [c for c in close.columns if c != "Datetime"]
if len(asset_cols_now) < 2:
    st.error("Need at least 2 assets for fusion page")
    st.stop()

st.markdown("### assets in portfolio: " + ", ".join(asset_cols_now))
st.caption("interval: " + str(timeframes_config["interval"]) + " | period: " + str(timeframes_config["period"]))

values_df, returns_dict = compute_portfolio_strategies(close, float(capital))

factor_val = float(np.sqrt(annual_factor.get(timeframes_config["interval"], 252)))

#metrics table
rows = []
for name in ["Buy & Hold", "SMA Crossover", "SMA + RSI", "Simple Momentum"]:
    v = values_df[name]
    r = returns_dict[name]
    rows.append({"strategy": name, "total return": float(v.iloc[-1] / float(capital) - 1.0), "sharpe": sharpe_ratio(r, factor_val), "volatility": volatility(r, factor_val), "max drawdown": max_drawdown(v), "final value": float(v.iloc[-1])})
metrics = pd.DataFrame(rows).set_index("strategy")

if mode == "Strategy Comparaison":
    st.subheader("strategy comparaison (portfolio value)")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(close["Datetime"], values_df["Buy & Hold"], label="Buy & Hold", linewidth=2)
    ax.plot(close["Datetime"], values_df["SMA Crossover"], label="SMA Crossover")
    ax.plot(close["Datetime"], values_df["SMA + RSI"], label="SMA + RSI")
    ax.plot(close["Datetime"], values_df["Simple Momentum"], label="Simple Momentum")
    ax.axhline(float(capital), linestyle="--", alpha=0.6)
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("metrics")
    st.dataframe(metrics, use_container_width=True)
else:
    st.subheader("single strategy view")
    st.markdown("selected: **" + strategies_selected + "**")
    st.metric("portfolio value", f"{float(values_df[strategies_selected].iloc[-1]):.2f}")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(close["Datetime"], values_df[strategies_selected], label="strategy value")
    ax.axhline(float(capital), linestyle="--", alpha=0.6)
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("metrics")
    st.dataframe(metrics.loc[[strategies_selected]], use_container_width=True)
