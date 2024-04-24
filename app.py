import pandas as pd
from datetime import date

import warnings

from functions import *
warnings.filterwarnings("ignore")

import streamlit as st


st.title("Advanced Portfolio Builder")

st.header("Initial Portfolio Analysis")


# Price data
own_portfolio_input = st.text_input("Enter assets separated by comma (example: AAPL, BTC-USD, CL=F, GC=F):")
own_portfolio = [asset.strip() for asset in own_portfolio_input.split(',')]
initial_date = st.date_input("Enter start date:", date(2023, 1, 1))
last_date = st.date_input("Enter end date:", date(2024, 1, 1))

if own_portfolio_input:
    # Get price data
    data = price_data(own_portfolio, initial_date, last_date, "Close")
    correlations = data.corr()

    # Calculate returns
    returns = data.pct_change()

    # Show the correlation matrix
    #st.header("Correlation Matrix")
    #st.write("Use this matrix for analyzing assets correlation")
    #sns.set_style("white")
    #plt.figure(figsize=(10, 8))
    #ax = sns.heatmap(correlations, annot=True)
    #st.plot(plt)  # Show the plot in Streamlit

else:
    # Wait for the user to enter assets
    st.info("Please enter assets to continue.")
# Portfolio backtest 

st.header("Portfolio Backtest")

st.markdown("### Explanation of Input Parameters:")
st.markdown("- **Train periods:** Represents the amount of data used to optimize the portfolio.")
st.markdown("- **Test periods:** Indicates the amount of data used to test the optimized portfolio in past iterations.")
st.markdown("- **Periods type:** Allows you to choose between years, months, or weeks.")

st.markdown("### Example:")
st.markdown("""
- **Start Date:** 2023-01-01
- **End Date:** 2024-01-01
- **Train Periods:** 3
- **Test Periods:** 1
- **Period Type:** Months

Using this configuration, you will simulate optimizing your portfolio using the last 3 months' data for investing in the next month and rebalancing the portfolio monthly.
""")


train_periods = st.text_input("Enter the amount of periods for training or build the portfolio", "3")
test_periods = st.text_input("Enter the amount of periods for testing the portfolio", "1")
period_type = st.text_input("Enter the period you want to use: years, months or weeks")
benchmark_asset = st.text_input("Choose a asset(ticker) as a benchmark. S&P500 is for default", "^SPX")

if train_periods and test_periods and period_type and benchmark_asset:
    train_periods = int(train_periods)
    test_periods = int(test_periods)
    
    windows_train, windows_test = generate_windows(data,initial_date,last_date, train_periods, test_periods, period_type)
    
    profits = []
    dates = []
    errors = []
    weights_results = []
    
    for train_window, test_window in zip(windows_train, windows_test):
        profit, date, error, weights_result = get_results_portfolio(train_window, test_window)
        profits.append(profit)
        dates.append(date)
        errors.append(error)
        weights_results.append(weights_result)
        
    results = pd.DataFrame({"Date": dates, "Profit": profits, "Calc_error":errors})
    results["Total_profit"] = results["Profit"].cumsum()
    
    weights_data = (round(pd.DataFrame(weights_results),2))
    weights_data["Date"] = results["Date"]
    
    final_results = pd.merge(results, weights_data, on="Date")
    final_results = final_results.set_index("Date")
    weights_data = weights_data.set_index("Date")

    
    
    data = price_data([benchmark_asset], initial_date,last_date, "Close")
    data = pd.DataFrame(data)
    data["returns"] = data["Close"].pct_change()*100
    data = data.dropna()
    data["benchmark_profit"] = data["returns"].cumsum()
    
    if period_type == "years":
        time = "Y"
    if period_type == "months":
        time = "M"  
    if period_type == "weeks":
        time = "W"  
    
    data = data.resample(time).agg({"benchmark_profit":"last"})
    data = data.reset_index()
    final_results = final_results.reset_index()
    
    plot_results = pd.merge(final_results, data, on="Date")
    plot_results = plot_results.set_index("Date")
    
    # Results
    
    st.header("Backtest results")
    
    st.write("Profit is measured as cumulative return of prices over time in %")
    line_chart_st(final_results, ["Total_profit"],"Portfolio strategy total return over time")
    line_chart_st(data, ["benchmark_profit"],"Benchmark return over time")
    
    line_chart_st(weights_data, weights_data.columns.to_list(), "Weights over time")


# Portfolio in real time

st.header("Get your portfolio in real time")

own_portfolio_input = st.text_input("Enter assets separated by comma for building your portfolio in real time:")
own_portfolio = [asset.strip() for asset in own_portfolio_input.split(',')]
#initial_date = st.date_input("Enter start date (just put a date bigger than your training amount):", date(2021, 1, 1))
#last_date = st.date_input("Enter end date:", date(2024, 1, 1))
initial_date = "2000-01-01"
today = date.today().strftime('%Y-%m-%d')

current_date = today 
train_periods = st.text_input("Enter the amount of periods that you used in backtesting")
#test_periods = st.text_input("Enter the amount of periods for testing the portfolio")
period_type = st.text_input("Enter the type of period you used in backtesting")

if own_portfolio_input and train_periods and period_type:
    train_periods = int(train_periods)
    data_realtime = price_data(own_portfolio, initial_date,today, "Close")
    real_time_weights, train_window = get_real_time_weights(data_realtime, initial_date, current_date, train_periods, period_type)
    
    final_weights = pd.DataFrame(round(real_time_weights,4)*100).rename(columns={0: 'Weights in %'})
    
    st.header("Weights for the last moment where you had had to rebalance your portfolio")
    st.table(final_weights)
    st.header("Last prices data")
    st.table(train_window)