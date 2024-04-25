import pandas as pd
from datetime import date

import warnings

from functions import *
warnings.filterwarnings("ignore")

import streamlit as st


menu_selection = st.sidebar.radio("Navigation", ["Portfolio Analysis", "What is Behind?"])

linkedin_url = "https://www.linkedin.com/in/edwar-valenzuela/?originalSubdomain=co"

st.sidebar.markdown(f"[Linkedin]({linkedin_url})")


if menu_selection == "Portfolio Analysis":

    st.title("Advanced Portfolio Builder")

    st.header("Initial Portfolio Analysis")


    st.write("You can search tickers in https://stockanalysis.com/stocks/")

    # Price data

    own_portfolio_input = st.text_input("Enter assets tickers separated by comma (example: AAPL, BTC-USD, CL=F, GC=F)")
    own_portfolio = [asset.strip() for asset in own_portfolio_input.split(',')]
    initial_date = st.date_input("Enter start date:", date(2015, 1, 1))
    last_date = st.date_input("Enter end date:", date(2024, 1, 1))

    if own_portfolio_input:
        # Get price data
        data = price_data(own_portfolio, initial_date, last_date, "Close")
        correlations = data.corr()

        # Calculate returns
        returns = data.pct_change()
        cum_returns = returns.dropna().cumsum()*100
        # Show the correlation matrix
        #st.header("Correlation Matrix")
        #st.write("Use this matrix for analyzing assets correlation")
        sns.set_style("white")
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(correlations, annot=True)
        plt.title('Correlation Matrix')
        plt.show()

        line_chart_st(cum_returns, cum_returns.columns.to_list(), "Assets returns over time")

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
    - **Start Date:** 2015-01-01
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
    condition = st.selectbox("Choose between maximizing sharpe ratio or reduce volatility.", 
                                ["Maximize Sharpe Ratio", "Reduce Volatility"])

    if condition == "Maximize Sharpe Ratio":
        pf_condition = "sharpe"
    elif condition == "Reduce Volatility":
        pf_condition = "volatility"

    if train_periods and test_periods and period_type and benchmark_asset:
        train_periods = int(train_periods)
        test_periods = int(test_periods)

        windows_train, windows_test = generate_windows(data,initial_date,last_date, train_periods, test_periods, period_type)

        profits = []
        dates = []
        errors = []
        weights_results = []

        pf_condition = "volatility"

        for train_window, test_window in zip(windows_train, windows_test):
            profit, date, error, weights_result = get_results_portfolio(train_window, test_window, pf_condition)
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

        max_dd = max_drawdown(final_results["Total_profit"])


        # Results

        st.header("Backtest results")

        st.write("The Max drawdown (maximum observed loss from a peak to a trough of an investment, before a new peak is attained) is: ", f"<span style='color:orange'>{round(max_dd, 2)}%</span>", unsafe_allow_html=True)

        #st.write("Profit is measured as cumulative return of prices over time in %")
        line_chart_2_st(final_results, data, "Total_profit", "benchmark_profit", "Portfolio strategy total return vs benchmark return over time")

        line_chart_st(weights_data, weights_data.columns.to_list(), "Weights over time")


    # Portfolio in real time

    st.header("Get your portfolio in real time")

    own_portfolio_input = st.text_input("Enter assets separated by comma for building your portfolio in real time:")
    own_portfolio = [asset.strip() for asset in own_portfolio_input.split(',')]
    #initial_date = st.date_input("Enter start date (just put a date bigger than your training amount):", date(2021, 1, 1))
    #last_date = st.date_input("Enter end date:", date(2024, 1, 1))
    initial_date = "2010-01-01"
    today = date.today().strftime('%Y-%m-%d')

    current_date = today 
    train_periods = st.text_input("Enter the amount of periods that you used in backtesting for training")
    #test_periods = st.text_input("Enter the amount of periods for testing the portfolio")
    period_type = st.text_input("Enter the type of period you used in backtesting")

    if own_portfolio_input and train_periods and period_type:
        train_periods = int(train_periods)
        data_realtime = price_data(own_portfolio, initial_date,today, "Close")
        real_time_weights, train_window = get_real_time_weights(data_realtime, initial_date, current_date, train_periods, period_type)

        final_weights = pd.DataFrame(round(real_time_weights,4)*100).rename(columns={0: 'Weights in %'})

        final_weights = round(final_weights,2)
        st.header("Weights for the last moment where you had had to rebalance your portfolio")
        st.table(final_weights)

        weights_plot_st(final_weights)

        st.header("Last prices data")
        st.table(train_window)

elif menu_selection == "What is Behind?":
    # Explanation text
    st.title("Explanation")

    text = """
    ### Portfolio Theory:
    
    Imagine you have some money to invest, and you're considering buying different things with it, like stocks, bonds, or maybe even real estate. Portfolio theory is a way to help you decide how to mix these investments together to get the best possible return while managing risk.
    
    The basic idea is that by spreading your money across different types of investments, you can reduce the overall risk of your portfolio. This is because if one investment does poorly, hopefully, another one will do well enough to make up for it.
    
    But it's not just about spreading your money around randomly. Portfolio theory uses mathematical formulas to help you find the best mix of investments based on things like:
    
    - **Expected return:** How much money you expect to make from each investment.
    - **Risk:** How much the value of each investment might go up and down over time.
    - **Correlation:** How closely the returns of different investments are related to each other.
    
    By considering these factors, portfolio theory helps you build a portfolio that gives you the best possible return for the amount of risk you're willing to take.
    
    ### Optimization:
    
    Once you've decided on the types of investments you want to include in your portfolio, optimization is about finding the best way to allocate your money among them. It's like solving a puzzle to figure out the right balance that maximizes your return while minimizing your risk.
    
    Optimization algorithms use mathematical techniques to explore different combinations of investments and find the one that gives you the best results according to your goals and constraints. For example, you might want to maximize your return while keeping your risk below a certain level, or you might want to minimize your risk while aiming for a specific level of return.
    
    ### Rolling Windows for Backtesting:
    
    Backtesting is a way to test how well a trading strategy or investment approach would have performed in the past. It's like simulating your strategy in historical market conditions to see how it would have fared.
    
    One challenge in backtesting is deciding how much historical data to use. Using too much data may not accurately reflect how your strategy would perform in current market conditions, while using too little data may not provide a robust assessment of its effectiveness.
    
    This is where rolling windows come in. Instead of testing your strategy on the entire historical dataset at once, rolling windows divide the data into smaller, overlapping chunks, and test the strategy on each chunk separately. For example, if you have 10 years of historical data, you might use rolling windows of 1-year periods, moving forward one year at a time.
    
    By using rolling windows, you can evaluate the performance of your portfolio strategy across different periods of time, capturing various market conditions such as bull markets, bear markets, and periods of volatility. This approach helps to reduce the uncertainty associated with selecting the amount of historical data to use, as it provides a more comprehensive view of your strategy's performance over time.
    """

    st.markdown(text)