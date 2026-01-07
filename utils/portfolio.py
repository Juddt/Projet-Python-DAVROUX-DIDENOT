import pandas as pd #import pandas

#compute simple returns from prices
#input:prices df with tickers columns
#output:returns df with same columns
#notes:use pct change on rows
#notes:drop first nan row
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = pd.DataFrame() #init empty df

    #check empty prices case
    if prices is None or prices.empty:
        return returns #return empty df

    returns = prices.pct_change() #compute returns
    returns = returns.dropna(how="all") #drop fully empty rows
    return returns #return returns df

#normalize prices with base value
#input:prices df with tickers columns, base_value float
#output:normalized df with base_value at start
#notes:divide by first valid row
#notes:keep df shape and cols
def normalize_prices(prices: pd.DataFrame, base_value: float) -> pd.DataFrame:
    norm_prices = pd.DataFrame() #init empty df

    #check empty prices case
    if prices is None or prices.empty:
        return norm_prices #return empty df

    first_row = prices.iloc[0] #take first row
    norm_prices = prices.divide(first_row) #normalize by first row
    norm_prices = norm_prices.multiply(base_value) #scale to base value
    return norm_prices #return norm df

#compute basic metrics for each asset
#input:returns df with tickers columns
#output:metrics df with columns mean_return and vol
#notes:mean is average return per step
#notes:vol is std of returns
def compute_asset_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame() #init empty df

    #check empty returns case
    if returns is None or returns.empty:
        return metrics #return empty df

    metrics = pd.DataFrame(index=returns.columns) #init metrics df
    metrics["mean_return"] = returns.mean() #compute mean return
    metrics["vol"] = returns.std() #compute vol
    return metrics #return metrics df
