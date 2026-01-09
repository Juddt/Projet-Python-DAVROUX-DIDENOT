import os #import os
import argparse #import argparse
import pandas as pd #import pandas
from datetime import datetime #import datetime
from utils.data_loader import get_price_data #import data fct
from utils.portfolio import compute_returns, compute_max_drawdown, compute_portfolio_metrics #import portfolio fct

#get default tickers for report
#input:none
#output:tickers list
#notes:use gafam tickers
#notes:keep list on one line
def get_default_tickers() -> list[str]:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] #set default tickers
    return tickers #return tickers list

#compute open close and daily stats for each asset
#input:prices df with datetime index
#output:report df with open close vol
#notes:open is first row close
#notes:close is last row close
def build_asset_report(prices: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame() #init report df

    #check empty prices case
    if prices is None or prices.empty:
        return report #return empty df

    #compute returns for vol
    returns = compute_returns(prices) #compute returns
    report = pd.DataFrame(index=prices.columns) #init report index
    report["open"] = prices.iloc[0] #set open proxy
    report["close"] = prices.iloc[-1] #set close
    report["return"] = (prices.iloc[-1] / prices.iloc[0]) - 1.0 #compute period return
    report["vol"] = returns.std() #compute vol
    return report #return report df

#save report df as csv in reports dir
#input:report df, report_path str
#output:none
#notes:create dir if missing
#notes:overwrite file if exists
def save_report(report: pd.DataFrame, report_path: str) -> None:
    report_dir = os.path.dirname(report_path) #get report dir

    #create reports folder if needed
    if report_dir != "":
        os.makedirs(report_dir, exist_ok=True) #create dir

    report.to_csv(report_path, index=True) #write csv

#run daily report generation
#input:args namespace
#output:int code
#notes:download prices and compute stats
#notes:write csv in reports folder
def run_daily_report(args: argparse.Namespace) -> int:
    tickers = args.tickers.split(",") #parse tickers str
    period = args.period #get period
    interval = args.interval #get interval
    out_dir = args.output_dir #get output dir
    report_date = args.report_date #get report date str

    #fallback if tickers empty
    if len(tickers) == 1 and tickers[0].strip() == "":
        tickers = get_default_tickers() #set default tickers

    prices = get_price_data(tickers, period=period, interval=interval) #download prices

    #check missing data case
    if prices is None or prices.empty:
        print("no data, report not generated") #print error
        return 1 #return error

    prices = prices.dropna(axis=1, how="all") #drop empty cols
    prices = prices.ffill() #fill small gaps
    asset_report = build_asset_report(prices) #build asset report

    returns = compute_returns(prices) #compute returns for portfolio
    equal_w = {c: 1.0 for c in prices.columns} #set eq weights
    port_returns = pd.Series(dtype=float) #init port returns

    #compute portfolio returns as equal weight
    if returns is not None and (not returns.empty):
        port_returns = returns.mean(axis=1) #compute eq weight returns

    port_value = (1.0 + port_returns).cumprod() #compute port value
    mdd = compute_max_drawdown(port_value) #compute max drawdown
    port_metrics = compute_portfolio_metrics(port_returns, periods_per_year=args.periods_per_year) #compute metrics

    summary = pd.DataFrame(index=["portfolio"]) #init summary df
    summary["ann_return"] = [port_metrics.get("ann_return", 0.0)] #set ann return
    summary["ann_vol"] = [port_metrics.get("ann_vol", 0.0)] #set ann vol
    summary["sharpe"] = [port_metrics.get("sharpe", 0.0)] #set sharpe
    summary["max_drawdown"] = [mdd] #set mdd

    full_report = pd.concat([asset_report, summary], axis=0) #concat report

    #build output filename
    if report_date == "":
        report_date = datetime.now().strftime("%Y-%m-%d") #set date str

    report_path = os.path.join(out_dir, f"daily_report_{report_date}.csv") #set report path
    save_report(full_report, report_path) #save report
    print("report saved:", report_path) #print ok
    return 0 #return ok

#parse args from cli
#input:none
#output:args namespace
#notes:keep defaults simple
#notes:tickers as comma separated str
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser() #init parser
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL,AMZN,META") #set tickers arg
    parser.add_argument("--period", type=str, default="5d") #set period arg
    parser.add_argument("--interval", type=str, default="1d") #set interval arg
    parser.add_argument("--output_dir", type=str, default="reports") #set output dir arg
    parser.add_argument("--report_date", type=str, default="") #set report date arg
    parser.add_argument("--periods_per_year", type=int, default=252) #set annual factor
    args = parser.parse_args() #parse args
    return args #return args

#run main prg
if __name__ == "__main__":
    args = parse_args() #parse cli args
    code = run_daily_report(args) #run report
    raise SystemExit(code) #exit with code
