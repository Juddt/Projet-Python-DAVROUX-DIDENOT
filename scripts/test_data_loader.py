from utils.data_loader import get_price_data #import data fct

#run a small test for data download
#input:none
#output:none
#notes:print shape and few rows
#notes:use daily interval to avoid yfinance limits
def main() -> None:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] #set tickers list
    prices = get_price_data(tickers, period="6mo", interval="1d") #download prices

    #check if prices df is empty
    if prices is None or prices.empty:
        print("no data, change period or interval") #print error
        return #stop prog

    print("prices shape:", prices.shape) #print df shape
    print(prices.tail(3)) #print last rows

#run main prg
if __name__ == "__main__":
    main() #call main
