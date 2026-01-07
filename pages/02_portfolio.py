import streamlit as st #import streamlit
import pandas as pd #import pandas
from utils.data_loader import get_price_data, get_last_prices #import data fct
from utils.portfolio import compute_returns, normalize_prices, compute_asset_metrics #import portfolio fct

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

st.set_page_config(page_title="portfolio", layout="wide") #set page config
st.title("portfolio multi assets") #set page title

default_tickers = get_default_tickers() #get default tickers
selected_tickers = st.multiselect("tickers", options=default_tickers, default=default_tickers) #select tickers
period = st.selectbox("period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"]) #select period
interval = st.selectbox("interval", options=["1d", "1wk", "1mo"]) #select interval
base_value = st.number_input("base value", min_value=10.0, max_value=1000.0, value=100.0, step=10.0) #set base value

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

    st.subheader("last prices") #set subtitle
    st.dataframe(last_prices) #show last prices

    st.subheader("normalized prices chart") #set subtitle
    st.line_chart(norm_prices) #plot normalized prices

    st.subheader("returns chart") #set subtitle
    st.line_chart(returns) #plot returns

    st.subheader("asset metrics") #set subtitle
    st.dataframe(metrics) #show metrics table

    st.subheader("close prices table") #set subtitle
    st.dataframe(prices.tail(20)) #show last rows
