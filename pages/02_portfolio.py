import streamlit as st #import streamlit
import pandas as pd #import pandas
from utils.data_loader import get_price_data, get_last_prices #import data fct

#set default tickers for quant b
#input:none
#output:tickers list
#notes:use gafam tickers
#notes:keep list on one line
def get_default_tickers() -> list[str]:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] #set default tickers
    return tickers #return tickers list

st.set_page_config(page_title="portfolio", layout="wide") #set page config
st.title("portfolio multi assets") #set page title

default_tickers = get_default_tickers() #get default tickers
selected_tickers = st.multiselect("tickers", options=default_tickers, default=default_tickers) #select tickers
period = st.selectbox("period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"]) #select period
interval = st.selectbox("interval", options=["1d", "1wk", "1mo"]) #select interval

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
    last_prices = get_last_prices(prices) #get last prices

    st.subheader("last prices") #set subtitle
    st.dataframe(last_prices) #show last prices

    st.subheader("close prices") #set subtitle
    st.dataframe(prices.tail(20)) #show last rows
