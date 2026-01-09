# Projet-Python-DAVROUX-DIDENOT

This project is a Streamlit dashboard developed as part of a university course on Python, Git and Linux.
The objective is to analyze financial assets using quantitative strategies, both on single assets and on portfolios.

The application is composed of three main parts.

- Quant A is dedicated to single asset analysis.
It allows the user to select a ticker, an initial capital and a timeframe.
Several trading strategies are implemented and compared on the same asset.

- Quant B focuses on portfolio analysis.
Multiple assets can be selected at the same time and the global portfolio behavior is analyzed in terms of performance and risk.

- Quant Merge combines both approaches.
The same strategies used in Quant A are applied to several assets in order to study their behavior at the portfolio level.

The dashboard is fully interactive and updates market data automatically.

Technologies used are Python, Streamlit, Pandas, NumPy, Matplotlib, yfinance, Git and Linux.







Project structure

The main application is launched from app.py.
Each module is implemented as a Streamlit page inside the pages folder.
The project follows the standard Streamlit multipage structure.

Quant A – Single Asset Analysis

This module analyzes one asset at a time.
The user can choose the ticker, the capital and the timeframe.
Data is refreshed every five minutes.

Implemented strategies are Buy and Hold, SMA Crossover, SMA with RSI and Simple Momentum.
For each strategy, the following metrics are displayed: total return, Sharpe ratio, volatility and maximum drawdown.

Quant B – Portfolio Analysis

This module allows the analysis of several assets simultaneously.
Assets are combined into a portfolio and the global performance is evaluated.
The focus is on portfolio-level metrics rather than individual trading signals.

Quant Merge – Multi Asset Strategies

This module applies the same strategies as Quant A to multiple assets.
It allows direct comparison of strategies across several stocks, such as GAFAM.
This part represents the final and most complete module of the project.






Deployment

The application is deployed using Streamlit Community Cloud.
This provides a public and permanent link accessible from any operating system.
The execution environment is Linux-based.

How to run locally

Install dependencies using the requirements file.
Run the application with streamlit run app.py.
The dashboard will be available on localhost.

Conclusion

This project demonstrates the implementation of quantitative trading strategies using Python.
It combines data analysis, visualization and deployment in a complete financial dashboard.
The final application satisfies all requirements of the assignment.
