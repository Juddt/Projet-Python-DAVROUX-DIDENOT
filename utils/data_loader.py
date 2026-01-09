import pandas as pd #import pandas
import yfinance as yf #import yfinance

#download close prices for multi assets
#input:tickers list, period str, interval str
#output:prices df with datetime index and tickers columns
#notes:use auto adjust and keep close only
#notes:disable threads to avoid yfinance cache lock
def get_price_data(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    prices = pd.DataFrame() #init empty df

    #download market data from yfinance
    try:
        data = yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=True, progress=False, threads=False) #download data
    except Exception:
        return prices #return empty df

    #handle yfinance multiindex output
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy() #keep close only

    #handle yfinance single ticker output
    if (not isinstance(data.columns, pd.MultiIndex)) and ("Close" in data.columns):
        prices = data[["Close"]].copy() #keep close col
        if len(tickers) > 0:
            prices.columns = [tickers[0]] #rename to ticker

    prices = prices.dropna(how="all") #drop fully empty rows
    return prices #return prices df

#get last prices for quick display
#input:prices df with tickers columns
#output:series with last row values
#notes:return empty series if df empty
def get_last_prices(prices: pd.DataFrame) -> pd.Series:
    last_prices = pd.Series(dtype=float) #init empty series

    #check empty prices case
    if prices is None or prices.empty:
        return last_prices #return empty series

    last_prices = prices.iloc[-1].copy() #take last row
    return last_prices #return last prices
