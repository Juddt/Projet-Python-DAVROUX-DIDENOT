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

    #align weights with returns columns
    common_assets = [col for col in returns.columns if col in weights] #get common assets
    if len(common_assets) == 0:
        return port_returns #return empty series

    weighted_returns = returns[common_assets].multiply(pd.Series(weights), axis=1) #apply weights
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

    port_value = (1 + port_returns).cumprod() * base_value #compute portfolio value
    return port_value #return portfolio value
