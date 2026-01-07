import streamlit as st #import streamlit
import pandas as pd #import pandas
from utils.data_loader import get_price_data, get_last_prices #import data fct
from utils.portfolio import compute_returns, normalize_prices, compute_asset_metrics, compute_correlation, normalize_weights, compute_max_drawdown, compute_portfolio_metrics, compute_rebalanced_portfolio #import portfolio fct

#set default tickers for quant b
#input:none
#output:tickers list
#notes:use gafam tickers
#notes:keep list on one line
def get_default_tickers() -> list[str]:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] #set default tickers
    return tickers #return tickers list

#clean prices df for display
#input:prices df
#output:cleaned prices df
#notes:drop fully empty cols
#notes:forward fill small gaps
def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    clean_df = prices.copy() #copy df

    #drop assets with no data
    clean_df = clean_df.dropna(axis=1, how="all") #drop empty cols

    #fill small gaps to avoid broken plots
    clean_df = clean_df.ffill() #forward fill nan
    return clean_df #return clean df

#convert df to csv bytes for download
#input:df
#output:bytes csv
#notes:use utf8 encoding
#notes:keep index in csv
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    csv_bytes = b"" #init bytes

    #check empty df case
    if df is None or df.empty:
        return csv_bytes #return empty bytes

    csv_str = df.to_csv(index=True) #convert to csv
    csv_bytes = csv_str.encode("utf-8") #encode bytes
    return csv_bytes #return bytes

st.set_page_config(page_title="portfolio", layout="wide") #set page config
st.title("portfolio multi assets") #set page title

default_tickers = get_default_tickers() #get default tickers
selected_tickers = st.multiselect("tickers", options=default_tickers, default=default_tickers) #select tickers
period = st.selectbox("period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2) #select period
interval = st.selectbox("interval", options=["1d", "1wk", "1mo"], index=0) #select interval
base_value = st.number_input("base value", min_value=10.0, max_value=1000.0, value=100.0, step=10.0) #set base value
rebalance_freq = st.selectbox("rebalance", options=["D", "W", "M"], index=1) #select rebalance freq

#load prices when user clicks
load_data = st.button("load data") #set load button

prices = pd.DataFrame() #init prices df

#check if user wants to load data
if load_data:
    #download close prices for selected tickers
    prices = get_price_data(selected_tickers, period=period, interval=interval) #get prices df

#check if we have prices to display
if prices is None or prices.empty:
    st.info("click load data to download prices") #show info
else:
    prices = clean_prices(prices) #clean prices df
    last_prices = get_last_prices(prices) #get last prices
    returns = compute_returns(prices) #compute returns
    norm_prices = normalize_prices(prices, base_value=base_value) #compute normalized prices
    metrics = compute_asset_metrics(returns) #compute asset metrics
    corr = compute_correlation(returns) #compute corr matrix

    st.subheader("last prices") #set subtitle
    st.dataframe(last_prices) #show last prices

    st.subheader("normalized prices chart") #set subtitle
    st.line_chart(norm_prices) #plot normalized prices

    st.subheader("returns chart") #set subtitle
    st.line_chart(returns) #plot returns

    st.subheader("asset metrics") #set subtitle
    st.dataframe(metrics) #show metrics table

    st.subheader("correlation matrix") #set subtitle
    st.dataframe(corr) #show corr df

    st.subheader("portfolio weights") #set subtitle
    raw_weights = {} #init raw weights dict

    #create sliders for each asset
    for ticker in prices.columns:
        raw_weights[ticker] = st.slider(f"weight {ticker}", min_value=0.0, max_value=1.0, value=1.0, step=0.05) #set weight slider

    weights = normalize_weights(raw_weights) #normalize weights
    port_value, port_returns = compute_rebalanced_portfolio(returns, weights, rebalance_freq=rebalance_freq, base_value=base_value) #compute portfolio
    port_metrics = compute_portfolio_metrics(port_returns, periods_per_year=252) #compute portfolio metrics
    mdd = compute_max_drawdown(port_value) #compute max drawdown

    st.subheader("portfolio value chart") #set subtitle
    st.line_chart(port_value) #plot portfolio value

    st.subheader("portfolio metrics") #set subtitle
    c1, c2, c3, c4 = st.columns(4) #create metrics cols
    c1.metric("ann return", f"{port_metrics.get('ann_return', 0.0) * 100:.2f}%") #show ann return
    c2.metric("ann vol", f"{port_metrics.get('ann_vol', 0.0) * 100:.2f}%") #show ann vol
    c3.metric("sharpe", f"{port_metrics.get('sharpe', 0.0):.2f}") #show sharpe
    c4.metric("max drawdown", f"{mdd * 100:.2f}%") #show mdd

    st.subheader("export") #set subtitle
    st.download_button("download prices csv", data=df_to_csv_bytes(prices), file_name="prices.csv", mime="text/csv") #download prices
    st.download_button("download returns csv", data=df_to_csv_bytes(returns), file_name="returns.csv", mime="text/csv") #download returns
    st.download_button("download corr csv", data=df_to_csv_bytes(corr), file_name="correlation.csv", mime="text/csv") #download corr

    st.subheader("close prices table") #set subtitle
    st.dataframe(prices.tail(20)) #show last rows
