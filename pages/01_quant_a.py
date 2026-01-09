import pandas as pd #import pandas
import numpy as np #import numpy
import yfinance as yf #import yfinance
import matplotlib.pyplot as plt #import matplotlib
from datetime import datetime #import datetime

#download data from yfinance with cache
#input:ticker str, period str, interval str
#output:data df with datetime and close
#notes:ttl 300 sec for refresh rule
#notes:use buffer period to compute indicators
@st.cache_data(ttl=300)
def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    buffer_map = {"1d": "5d", "5d": "1mo", "1mo": "3mo", "6mo": "1y", "1y": "2y", "5y": "10y", "max": "max"} #set buffer periods
    period_large = buffer_map.get(period, period) #get buffer period
    raw = yf.download(ticker, period=period_large, interval=interval, progress=False) #download data
    data = pd.DataFrame() #init df

    #check empty download case
    if raw is None or raw.empty:
        return data #return empty df

    data = raw[["Close"]].reset_index() #keep close and datetime
    data.columns = ["datetime", "close"] #rename cols
    data = data.dropna() #drop nan rows
    return data #return clean df

#compute rsi indicator
#input:close series, window int
#output:rsi series
#notes:use simple rolling mean
#notes:avoid div by zero
def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff() #compute delta
    gain = delta.clip(lower=0) #keep gains
    loss = -delta.clip(upper=0) #keep losses
    avg_gain = gain.rolling(window=window).mean() #avg gain
    avg_loss = loss.rolling(window=window).mean() #avg loss
    rs = avg_gain / avg_loss #compute rs
    rsi = 100 - (100 / (1 + rs)) #compute rsi
    return rsi #return rsi

#compute max drawdown from value curve
#input:value series
#output:float drawdown min
#notes:use value vs cummax
#notes:return 0 if empty
def compute_max_drawdown(value: pd.Series) -> float:
    mdd = 0.0 #init mdd

    #check empty series case
    if value is None or value.empty:
        return mdd #return 0

    running_max = value.cummax() #compute running max
    dd = (value - running_max) / running_max #compute drawdown
    mdd = float(dd.min()) #take min
    return mdd #return mdd

#compute sharpe ratio with annual factor
#input:returns series, factor float
#output:float sharpe
#notes:return nan if std is 0
#notes:mean/std scaled by factor
def compute_sharpe(returns: pd.Series, factor: float) -> float:
    sharpe = float("nan") #init sharpe
    if returns is None or returns.empty:
        return sharpe #return nan
    if float(returns.std()) == 0.0:
        return sharpe #return nan
    sharpe = float(returns.mean() / returns.std() * factor) #compute sharpe
    return sharpe #return sharpe

#compute annualized volatility with annual factor
#input:returns series, factor float
#output:float vol
#notes:return nan if std is 0
#notes:std scaled by factor
def compute_volatility(returns: pd.Series, factor: float) -> float:
    vol = float("nan") #init vol
    if returns is None or returns.empty:
        return vol #return nan
    if float(returns.std()) == 0.0:
        return vol #return nan
    vol = float(returns.std() * factor) #compute vol
    return vol #return vol

st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True) #auto refresh page 5 min
st.title("quant a single asset") #set title
st.caption("last refresh: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")) #show refresh time

timeframes = {"1d": {"period": "1d", "interval": "5m"}, "5d": {"period": "5d", "interval": "15m"}, "1mo": {"period": "1mo", "interval": "1h"}, "6mo": {"period": "6mo", "interval": "1d"}, "1y": {"period": "1y", "interval": "1d"}, "5y": {"period": "5y", "interval": "1wk"}, "max": {"period": "max", "interval": "1wk"}} #set timeframes map
sma_defaults = {"1d": {"fast": 9, "slow": 21}, "5d": {"fast": 10, "slow": 50}, "1mo": {"fast": 20, "slow": 50}, "6mo": {"fast": 50, "slow": 100}, "1y": {"fast": 50, "slow": 150}, "5y": {"fast": 50, "slow": 200}, "max": {"fast": 50, "slow": 200}} #set sma defaults
rsi_defaults = {"1d": 7, "5d": 10, "1mo": 14, "6mo": 14, "1y": 21, "5y": 21, "max": 21} #set rsi defaults

st.sidebar.header("user settings") #sidebar title
ticker = st.sidebar.text_input("ticker", value="AAPL") #select ticker
capital = st.sidebar.number_input("initial capital", min_value=10, max_value=100000, value=100, step=1) #select capital
timeframe_key = st.sidebar.radio("timeframe", options=list(timeframes.keys()), horizontal=True, index=3) #select timeframe
strategy = st.sidebar.radio("strategy", options=["buy & hold", "sma crossover", "sma + rsi", "simple momentum"], index=0) #select strategy

tf = timeframes[timeframe_key] #get timeframe config
sma_fast = sma_defaults[timeframe_key]["fast"] #get default sma fast
sma_slow = sma_defaults[timeframe_key]["slow"] #get default sma slow
rsi_window = rsi_defaults[timeframe_key] #get default rsi window
rsi_buy = 55 #set rsi buy
rsi_sell = 70 #set rsi sell
momentum_window = 12 #set default momentum

#allow manual params for sma strategies
if strategy in ["sma crossover", "sma + rsi"]:
    manual_sma = st.sidebar.checkbox("manual sma", value=False) #manual sma toggle

    #change sma params if manual
    if manual_sma:
        sma_fast = st.sidebar.slider("fast sma", min_value=5, max_value=100, value=int(sma_fast)) #set fast sma
        sma_slow = st.sidebar.slider("slow sma", min_value=20, max_value=300, value=int(sma_slow)) #set slow sma

#allow rsi params only for sma+rsi
if strategy == "sma + rsi":
    rsi_window = st.sidebar.slider("rsi window", min_value=5, max_value=30, value=int(rsi_window)) #set rsi window
    rsi_buy = st.sidebar.slider("rsi buy", min_value=30, max_value=60, value=int(rsi_buy)) #set rsi buy
    rsi_sell = st.sidebar.slider("rsi sell", min_value=60, max_value=90, value=int(rsi_sell)) #set rsi sell

#allow momentum param only for momentum strat
if strategy == "simple momentum":
    momentum_window = st.sidebar.slider("momentum window", min_value=1, max_value=50, value=int(momentum_window)) #set momentum window

data = download_data(ticker, period=tf["period"], interval=tf["interval"]) #download data

#check empty data case
if data is None or data.empty:
    st.error("no data, try another ticker or timeframe") #show error
else:
    data["return"] = data["close"].pct_change().fillna(0.0) #compute returns
    data["price_norm"] = (data["close"] / float(data["close"].iloc[0])) * float(capital) #normalize price to capital
    data["sma_fast"] = data["close"].rolling(window=int(sma_fast)).mean() #compute sma fast
    data["sma_slow"] = data["close"].rolling(window=int(sma_slow)).mean() #compute sma slow
    data["rsi"] = compute_rsi(data["close"], window=int(rsi_window)) #compute rsi

    annual_map = {"5m": 252 * 78, "15m": 252 * 26, "1h": 252 * 7, "1d": 252, "1wk": 52} #set annualization map
    ann_factor = float(np.sqrt(annual_map.get(tf["interval"], 252))) #set annual factor

    data["position"] = 0.0 #init position
    data["strategy_value"] = float(capital) #init strategy value

    #compute buy and hold strategy
    if strategy == "buy & hold":
        data["strategy_value"] = data["price_norm"] #use price norm as value
        strat_returns = data["return"] #set returns for metrics

    #compute sma crossover strategy
    if strategy == "sma crossover":
        pos = np.where(data["sma_fast"] > data["sma_slow"], 1.0, 0.0) #compute raw position
        data["position"] = pd.Series(pos).shift(1).fillna(0.0) #shift position
        data["strategy_value"] = (1.0 + data["return"] * data["position"]).fillna(1.0).cumprod() * float(capital) #compute strat value
        strat_returns = (data["return"] * data["position"]).fillna(0.0) #set strat returns

    #compute sma + rsi strategy
    if strategy == "sma + rsi":
        signal = np.where((data["sma_fast"] > data["sma_slow"]) & (data["rsi"] < float(rsi_buy)), 1.0, np.where(data["rsi"] > float(rsi_sell), 0.0, np.nan)) #compute signal
        data["position"] = pd.Series(signal).ffill().shift(1).fillna(0.0) #build position
        data["strategy_value"] = (1.0 + data["return"] * data["position"]).fillna(1.0).cumprod() * float(capital) #compute strat value
        strat_returns = (data["return"] * data["position"]).fillna(0.0) #set strat returns

    #compute simple momentum strategy
    if strategy == "simple momentum":
        data["momentum"] = data["close"].pct_change(periods=int(momentum_window)) #compute momentum
        pos_m = np.where(data["momentum"] > 0.0, 1.0, 0.0) #compute raw position
        data["position"] = pd.Series(pos_m).shift(1).fillna(0.0) #shift position
        data["strategy_value"] = (1.0 + data["return"] * data["position"]).fillna(1.0).cumprod() * float(capital) #compute strat value
        strat_returns = (data["return"] * data["position"]).fillna(0.0) #set strat returns

    st.subheader("current value") #set subtitle
    st.metric("last close", f"{float(data['close'].iloc[-1]):.2f}") #show last close
    st.metric("strategy value", f"{float(data['strategy_value'].iloc[-1]):.2f}") #show strat value

    st.subheader("main chart price + strategy") #set subtitle

    fig, ax = plt.subplots(figsize=(12, 5)) #create fig
    ax.plot(data["datetime"], data["close"], label="price") #plot raw price
    ax.plot(data["datetime"], data["strategy_value"], label="strategy value") #plot strat value
    ax.set_title("price and strategy value") #set title
    ax.grid(alpha=0.3) #set grid
    ax.legend() #show legend
    st.pyplot(fig) #display fig

    total_return = float((data["strategy_value"].iloc[-1] / float(capital)) - 1.0) #compute total return
    sharpe = compute_sharpe(strat_returns, ann_factor) #compute sharpe
    vol = compute_volatility(strat_returns, ann_factor) #compute vol
    mdd = compute_max_drawdown(data["strategy_value"]) #compute mdd

    st.subheader("metrics") #set subtitle
    metrics_df = pd.DataFrame({"value": [total_return, sharpe, vol, mdd]}, index=["total return", "sharpe", "volatility", "max drawdown"]) #build metrics df
    st.dataframe(metrics_df) #show metrics df