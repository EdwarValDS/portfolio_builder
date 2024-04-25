import pandas as pd
from datetime import date

import seaborn as sns
import matplotlib.pyplot as plt
import base64

import warnings

from functions import *
warnings.filterwarnings("ignore")

import streamlit as st


st.set_page_config(page_title='Portfolio Strategy Builder',page_icon='💼') #layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
        """
        <style>
        {% include 'styles.css' %}
        </style>
        """,
        unsafe_allow_html=True
    )

menu_selection = st.sidebar.radio("Navigation", ["Portfolio Analysis", "Theory"])

linkedin_url = "https://www.linkedin.com/in/edwar-valenzuela/?originalSubdomain=co"

st.sidebar.markdown(f"[Linkedin]({linkedin_url})")



if menu_selection == "Portfolio Analysis":

    st.title("Portfolio Strategy Builder")

    st.header("Logic behind")

    markdown_text = """
    When it comes to optimizing a portfolio, the question arises of how much past data to use, how often to rebalance, or which assets represent a better portfolio. When an optimal portfolio is found in the past, whether seeking to reduce volatility or maximize profit, there's a risk of over-optimization that may not guarantee the same results in the future.

    However, there are techniques that can enable better decision-making when it comes to portfolio management. By using rolling window data splitting, this application allows you to simulate portfolio strategies by specifying the size of the dataset to optimize portfolios, test the portfolio itself over a certain number of periods, and determine an optimal period for rebalancing the portfolio. All of this through backtesting based on the amount of data you specify to download from the past.

    Afterwards, you can infer what the performance of your portfolio strategy would have been using a certain amount of past data and having rebalanced the portfolio at regular intervals. Similarly, you can decide whether to maximize returns or reduce risk by comparing the performance of the strategy with different weights used over time against an index or benchmark such as the S&P500.

    In the graph, you can observe the logic behind portfolio construction and testing. In this case, data from 2023 to 2024 of 3 assets was used. The first 3 months are used to find an optimal portfolio, then 1 month to find the portfolio's performance as if you had invested in real-time. This process is repeated throughout the year, with monthly rebalancing. You can specify these parameters when testing the portfolio strategy in the past.
    """

    st.markdown(markdown_text)


    file_ = open("ventanas_rodantes.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    #"""<h2 style='text-align: center;'>Logic behind</h2>"""
    st.markdown(

        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    st.header("Initial Portfolio Analysis")


    st.write("You can search tickers in https://stockanalysis.com/stocks/ or https://finance.yahoo.com/")

    # Price data

    def validate_tickers(tickers):
        for ticker in tickers:
            if not yf.Ticker(ticker).history(period="1d").empty and len(tickers)>1:
                continue
            else:
                return False
        return True

    own_portfolio_input = st.text_input("Enter two or more assets tickers separated by comma (example: AAPL, BTC-USD, CL=F, GC=F)")
    own_portfolio = [asset.strip().upper() for asset in own_portfolio_input.split(',')]
    initial_date = st.date_input("Enter start date:", date(2016, 1, 1))
    last_date = st.date_input("Enter end date:", date(2024, 1, 1))


    if own_portfolio_input:
        if validate_tickers(own_portfolio):
            # Get price data
            data = price_data(own_portfolio, initial_date, last_date, "Close")
            # Continue with data analysis
            correlations = data.corr()
            returns = data.pct_change()
            cum_returns = returns.dropna().cumsum()*100
            # Show the correlation matrix
            sns.set_style("white")
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(correlations, annot=True)
            plt.title('Correlation Matrix', color="white")
            plt.tick_params(axis='x', colors='white')  # Establecer el color de los valores del eje x
            plt.tick_params(axis='y', colors='white') 
            st.pyplot()
            line_chart_st(cum_returns, cum_returns.columns.to_list(), "Assets returns over time")
        else:
            st.error("One or more tickers not found or incorrectly written. Please check and try again.")
    else:
        st.warning("Please enter assets before attempting to download data.")
  
    # Portfolio backtest 

    st.header("Portfolio Backtest")

    st.markdown("### Explanation of Input Parameters:")
    st.markdown("- **Train periods:** Represents the amount of data used to optimize the portfolio.")
    st.markdown("- **Test periods:** Indicates the amount of data used to test the optimized portfolio in past iterations.")
    st.markdown("- **Periods type:** Allows you to choose between years, months, or weeks.")

    st.markdown("### Example:")
    st.markdown("""
    - **Start Date:** 2016-01-01
    - **End Date:** 2024-01-01
    - **Train Periods:** 5
    - **Test Periods:** 1
    - **Period Type:** Months
    - Using this configuration, you will simulate optimizing your portfolio using the last 5 months' data for investing in the next month and rebalancing the portfolio monthly.
    """)

    st.header("Inputs")
    train_periods = st.text_input("Enter the number of periods for building(optimize) the portfolio in each iteration", "5")
    test_periods = st.text_input("Enter the number of periods for testing the portfolio in each iteration", "1")

    period_options = ["months", "weeks", "years"]
    period_type = st.selectbox("Select the period you want to use:", period_options)
    benchmark_asset = st.text_input("Choose a asset(ticker) as a benchmark. S&P500 is for default", "^SPX")

    condition = st.selectbox("Choose between maximizing sharpe ratio or reduce volatility.", 
                                ["Maximize Sharpe Ratio", "Reduce Volatility"])

    if condition == "Maximize Sharpe Ratio":
        pf_condition = "sharpe"
    elif condition == "Reduce Volatility":
        pf_condition = "volatility"

    if train_periods and test_periods and period_type and benchmark_asset and own_portfolio_input:
        train_periods = int(train_periods)
        test_periods = int(test_periods)

        windows_train, windows_test = generate_windows(data,initial_date,last_date, train_periods, test_periods, period_type)

        profits = []
        dates = []
        errors = []
        weights_results = []


        for train_window, test_window in zip(windows_train, windows_test):
            profit, date, error, weights_result = get_results_portfolio(train_window, test_window, pf_condition)
            profits.append(profit)
            dates.append(date)
            errors.append(error)
            weights_results.append(weights_result)

        results = pd.DataFrame({"Date": dates, "Profit": profits, "Calc_error":errors})
        results["Strategy profit in %"] = results["Profit"].cumsum()

        weights_data = (round(pd.DataFrame(weights_results),2))
        weights_data["Date"] = results["Date"]

        final_results = pd.merge(results, weights_data, on="Date")
        final_results = final_results.set_index("Date")
        weights_data = weights_data.set_index("Date")



        data = price_data([benchmark_asset], initial_date,last_date, "Close")
        data = pd.DataFrame(data)
        data = data.loc[data.index>=final_results.index.min()]
        data["returns"] = data["Close"].pct_change()*100
        data = data.dropna()
        

        data["Benchmark profit in %"] = data["returns"].cumsum()
        benchmark_return = data["returns"].sum()

        if period_type == "years":
            time = "Y"
        if period_type == "months":
            time = "M"  
        if period_type == "weeks":
            time = "W"  

        data = data.resample(time).agg({"Benchmark profit in %":"last"})

        max_dd = max_drawdown(final_results["Strategy profit in %"])

        strategy_return = results["Profit"].sum()
        # Results

        st.header("Backtest results")

        st.write("The final return of the portfolio strategy was: ", f"<span style='color:green'>{round(strategy_return, 2)}%</span>", unsafe_allow_html=True)
        st.write("The Max drawdown (maximum observed loss from a peak to a trough of an investment, before a new peak is attained) was: ", f"<span style='color:orange'>{round(max_dd, 2)}%</span>", unsafe_allow_html=True)
        st.write("The return of the benchmark was: ", f"<span style='color:skyblue'>{round(benchmark_return, 2)}%</span>", unsafe_allow_html=True)

        #st.write("Profit is measured as cumulative return of prices over time in %")
        line_chart_2_st(final_results, data, "Strategy profit in %", "Benchmark profit in %", "Portfolio strategy total return vs benchmark return over time")

        line_chart_st(weights_data, weights_data.columns.to_list(), "Weights over time")





    # Portfolio in real time

    st.header("Get your portfolio in real time")

    own_portfolio_input = st.text_input("Enter assets tickers separated by comma for building your portfolio in real time:")
    own_portfolio = [asset.strip() for asset in own_portfolio_input.split(',')]
    #initial_date = st.date_input("Enter start date (just put a date bigger than your training amount):", date(2021, 1, 1))
    #last_date = st.date_input("Enter end date:", date(2024, 1, 1))
    initial_date = "2015-01-01"
    today = date.today().strftime('%Y-%m-%d')

    current_date = today 
    train_periods = st.text_input("Enter the number of periods that you used in backtesting for optimizing portfolios")
    #test_periods = st.text_input("Enter the amount of periods for testing the portfolio")
    period_options = ["months", "weeks", "years"]
    period_type = st.selectbox("Select the period you used in backtesting", period_options)

    condition = st.selectbox("Choose between maximizing sharpe ratio or reduce volatility.", 
                                ["Maximize Sharpe Ratio", "Reduce Volatility"])
    if condition == "Maximize Sharpe Ratio":
        pf_condition = "sharpe"
    elif condition == "Reduce Volatility":
        pf_condition = "volatility"

    if own_portfolio_input and train_periods and period_type:
        try:
            train_periods = int(train_periods)
            data_realtime = price_data(own_portfolio, initial_date,today, "Close")
            real_time_weights, train_window = get_real_time_weights(data_realtime, initial_date, current_date, train_periods, period_type, pf_condition)

            final_weights = pd.DataFrame(round(real_time_weights,4)*100).rename(columns={0: 'Weights in %'})

            final_weights = round(final_weights,2)
            st.header("Weights for the last moment where you had had to rebalance your portfolio")
            st.table(final_weights)

            weights_plot_st(final_weights)

            st.header("Last prices data")
            st.write(train_window)
        except ValueError:
            st.warning("Please enter a valid number for the number of periods.")
    #else:
    #    st.error("Please enter data properly.")

    st.write("")
    st.write("")
    st.write("")
    st.markdown(f"<p style='text-align:center;'>Elaborated by Edwar Valenzuela <br><br> <a href='{linkedin_url}'>Linkedin</a></p>", unsafe_allow_html=True)


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