import numpy as np #import numpy
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
#notes:divide by first row
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

#compute correlation matrix of returns
#input:returns df with tickers columns
#output:corr df
#notes:use pandas corr
#notes:drop fully empty cols
def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    corr = pd.DataFrame() #init empty df

    #check empty returns case
    if returns is None or returns.empty:
        return corr #return empty df

    clean_returns = returns.dropna(axis=1, how="all") #drop empty cols
    corr = clean_returns.corr() #compute corr
    return corr #return corr df

#normalize weights dict to sum 1
#input:raw weights dict
#output:weights dict normalized
#notes:keep only positive weights
#notes:return empty dict if sum is zero
def normalize_weights(raw_weights: dict) -> dict:
    weights = {} #init weights dict
    total = 0.0 #init total

    #compute sum of positive weights
    for k in raw_weights:
        if float(raw_weights[k]) > 0:
            total = total + float(raw_weights[k]) #add to total

    #check invalid sum
    if total <= 0:
        return weights #return empty dict

    #normalize weights values
    for k in raw_weights:
        if float(raw_weights[k]) > 0:
            weights[k] = float(raw_weights[k]) / total #normalize weight

    return weights #return weights dict

#compute portfolio returns from asset returns
#input:returns df with tickers columns, weights dict
#output:series of portfolio returns
#notes:weights must sum to 1
#notes:missing weights are ignored
def compute_portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    port_returns = pd.Series(dtype=float) #init empty series

    #check empty returns case
    if returns is None or returns.empty:
        return port_returns #return empty series

    common_assets = [col for col in returns.columns if col in weights] #get common assets

    #check if we have at least one asset
    if len(common_assets) == 0:
        return port_returns #return empty series

    w = pd.Series(weights) #convert weights to series
    weighted_returns = returns[common_assets].multiply(w, axis=1) #apply weights
    port_returns = weighted_returns.sum(axis=1) #sum weighted returns
    return port_returns #return portfolio returns

#compute portfolio value from returns
#input:portfolio returns series, base_value float
#output:series of portfolio value
#notes:use cumulative product
#notes:start at base value
def compute_portfolio_value(port_returns: pd.Series, base_value: float) -> pd.Series:
    port_value = pd.Series(dtype=float) #init empty series

    #check empty returns case
    if port_returns is None or port_returns.empty:
        return port_value #return empty series

    port_value = (1 + port_returns).cumprod() * float(base_value) #compute portfolio value
    return port_value #return portfolio value

#compute max drawdown from value curve
#input:value series
#output:max drawdown float
#notes:drawdown is value/running max - 1
#notes:return 0 if empty
def compute_max_drawdown(value: pd.Series) -> float:
    mdd = 0.0 #init mdd

    #check empty value case
    if value is None or value.empty:
        return mdd #return 0

    running_max = value.cummax() #compute running max
    dd = (value / running_max) - 1.0 #compute drawdown
    mdd = float(dd.min()) #take min drawdown
    return mdd #return mdd

#compute annualized portfolio metrics
#input:port_returns series, periods_per_year int
#output:metrics dict with ann return vol sharpe
#notes:sharpe without rf
#notes:return empty dict if empty series
def compute_portfolio_metrics(port_returns: pd.Series, periods_per_year: int) -> dict:
    metrics = {} #init dict

    #check empty returns case
    if port_returns is None or port_returns.empty:
        return metrics #return empty dict

    mean_r = float(port_returns.mean()) #compute mean return
    vol_r = float(port_returns.std()) #compute vol return
    ann_return = mean_r * float(periods_per_year) #annualize return
    ann_vol = vol_r * float(np.sqrt(periods_per_year)) #annualize vol
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else float("nan") #compute sharpe
    metrics = {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe} #set metrics dict
    return metrics #return metrics

#simulate portfolio with drift and rebalancing
#input:returns df, weights dict, rebalance freq str, base_value float
#output:tuple value series and returns series
#notes:allocations drift with asset returns
#notes:rebalance reset allocations at each new period
def compute_rebalanced_portfolio(returns: pd.DataFrame, weights: dict, rebalance_freq: str, base_value: float) -> tuple[pd.Series, pd.Series]:
    port_value = pd.Series(dtype=float) #init value series
    port_returns = pd.Series(dtype=float) #init returns series

    #check empty returns case
    if returns is None or returns.empty:
        return port_value, port_returns #return empty

    common_assets = [col for col in returns.columns if col in weights] #get common assets

    #check if we have at least one asset
    if len(common_assets) == 0:
        return port_value, port_returns #return empty

    work = returns[common_assets].copy() #copy returns
    w = pd.Series({k: float(weights[k]) for k in common_assets}) #build weight series
    w_sum = float(w.sum()) #compute sum

    #check weight sum
    if w_sum <= 0:
        return port_value, port_returns #return empty

    w = w / w_sum #normalize weights
    groups = work.index.to_period(rebalance_freq) #build rebalance groups
    alloc = (w * float(base_value)).to_numpy(dtype=float) #init allocations
    prev_value = float(base_value) #init prev value
    values = [] #init values list
    rets = [] #init returns list

    #loop on each time step for simulation
    for i in range(len(work.index)):
        current_period = groups[i] #get period
        if i == 0:
            last_period = current_period #init last period
        if current_period != last_period:
            alloc = (w * prev_value).to_numpy(dtype=float) #rebalance allocations
            last_period = current_period #update last period
        row = work.iloc[i].to_numpy(dtype=float) #get returns row
        alloc = alloc * (1.0 + row) #update allocations
        value = float(np.sum(alloc)) #compute total value
        r = (value / prev_value) - 1.0 #compute step return
        prev_value = value #update prev value
        values.append(value) #store value
        rets.append(r) #store return

    port_value = pd.Series(values, index=work.index, name="portfolio_value") #build value series
    port_returns = pd.Series(rets, index=work.index, name="portfolio_returns") #build returns series
    return port_value, port_returns #return series
