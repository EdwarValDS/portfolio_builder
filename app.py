import pandas as pd
from datetime import date

import seaborn as sns
import matplotlib.pyplot as plt
import base64

import warnings

from functions import *
from constant import * 

warnings.filterwarnings("ignore")

import streamlit as st


st.set_page_config(page_title='Portfolio Strategy Builder',page_icon='💼') #layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)



menu_selection = st.sidebar.radio("Navigation", ["Portfolio Analysis", "Theory"])

linkedin_url = "https://www.linkedin.com/in/edwar-valenzuela/?originalSubdomain=co"





#st.sidebar.markdown(f"[Linkedin]({linkedin_url})")


if menu_selection == "Portfolio Analysis":

    st.title("Portfolio Strategy Builder", )

    st.header("Logic behind", divider="gray")
    st.write("")

    st.markdown("""
    <div style="text-align: justify">
        <p>When it comes to optimizing a portfolio, the question arises of how much past data to use, how often to rebalance, or which assets represent a better portfolio. When an optimal portfolio is found in the past, whether seeking to reduce volatility or maximize profit, there's a risk of over-optimization that may not guarantee the same results in the future.</p>
        <p>However, there are techniques that can enable better decision-making when it comes to portfolio management. By using rolling window data splitting, this application allows you to simulate portfolio strategies by specifying the size of the dataset to optimize portfolios, test the portfolio itself over a certain number of periods, and determine an optimal period for rebalancing the portfolio. All of this through backtesting based on the amount of data you specify to download from the past.</p>
        <p>Afterwards, you can infer what the performance of your portfolio strategy would have been using a certain amount of past data and having rebalanced the portfolio at regular intervals. Similarly, you can decide whether to maximize returns or reduce risk by comparing the performance of the strategy with different weights used over time against an index or benchmark such as the S&P500.</p>
        <p>In the graph, you can observe the logic behind portfolio construction and testing. In this case, data from 2023 to 2024 of 3 assets was used. The first 3 months are used to find an optimal portfolio, then 1 month to find the portfolio's performance as if you had invested in real-time. This process is repeated throughout the year, with monthly rebalancing. You can specify these parameters when testing the portfolio strategy in the past.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    file_ = open("ventanas_rodantes.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    #"""<h2 style='text-align: center;'>Logic behind</h2>"""
    st.markdown(

        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    st.write("")
    st.header("Initial Portfolio Analysis", divider="gray")
    st.write("")



    st.write("You can search tickers in https://stockanalysis.com/stocks/ or https://finance.yahoo.com/")

    # Price data

    def validate_tickers(tickers):
        for ticker in tickers:
            if not yf.Ticker(ticker).history(period="1d").empty and len(tickers)>1:
                continue
            else:
                return False
        return True

    own_portfolio_input = st.text_input("Enter two or more assets tickers separated by comma (example: AAPL, BTC-USD, GC=F)")
    own_portfolio = [asset.strip().upper() for asset in own_portfolio_input.split(',')]
    initial_date = st.date_input("Enter start date:", date(2016, 1, 1))
    last_date = st.date_input("Enter end date:", date(2024, 1, 1))

    if own_portfolio_input:
        with st.spinner('Getting data...'):

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
  
    st.write("")
    #st.write(data)

    # Portfolio backtest 

    st.header("Portfolio Backtest", divider="gray")
    st.write("")

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
    st.write("")
    train_periods = st.text_input("Enter the number of periods for building(optimize) the portfolio in each iteration", "5")
    test_periods = st.text_input("Enter the number of periods for testing the portfolio in each iteration", "1")

    period_options = ["months", "weeks", "years"]
    period_type = st.selectbox("Select the period you want to use:", period_options)
    benchmark_asset = st.text_input("Choose a asset(ticker) as a benchmark. S&P500 is for default", "^SPX")

    condition = st.selectbox("Choose an option to optimize portfolio in each iteration", 
                                ["Maximize Sharpe Ratio", "Reduce Volatility","Expected return", "Volatility limit"])

    amount_selection = st.number_input("""If Expected return or volatility limit is selected. Enter the expected return or maximum level of volatility 
                                         you want to specify (%)""", min_value=0, max_value=100, value=10, 
                                         help="""Enter a percentage value. If portfolio can not be optimized in one iteration, the optimization for that
                                         iteration is made based in minimizing volatility as mucho as possible""")

    rf_rate = st.number_input("""Specify the risk-free rate (%) (default is 0)""", 
                              min_value=0, max_value=100,value=0, help="""Enter a percentage value for the risk-free rate. This only applies when 
                              "Maximize sharpe ration" is selected""")

    amount_selection = amount_selection/100
    rf_rate = rf_rate/100

    if condition == "Maximize Sharpe Ratio":
        pf_condition = "sharpe"
    elif condition == "Reduce Volatility":
        pf_condition = "volatility"
    elif condition == "Expected return":
        pf_condition = "target_return"
    elif condition == "Volatility limit":
        pf_condition = "max_volatility"


    if st.button("Run backtest"):
        with st.spinner('Running backtest...'):
            if train_periods and test_periods and period_type and benchmark_asset and own_portfolio_input:
                train_periods = int(train_periods)
                test_periods = int(test_periods)

                windows_train, windows_test = generate_windows(data,initial_date,last_date, train_periods, test_periods, period_type)

                profits = []
                errors = []
                weights_results = []
                end_dates = []
                start_dates= []
                pf_conds = []

                for train_window, test_window in zip(windows_train, windows_test):
                    profit, date, error, weights_result, pf_cond, start_date = get_results_portfolio(train_window, test_window, rf_rate, pf_condition, 
                                                                                                    target_return = amount_selection, max_volatility=amount_selection)
                    profits.append(profit)
                    end_dates.append(date)
                    errors.append(error)
                    weights_results.append(weights_result)
                    pf_conds.append(pf_cond)
                    start_dates.append(start_date)

                results = pd.DataFrame({"Date": end_dates, "Profit": profits, "Calc_error":errors})
                results["Strategy profit in %"] = results["Profit"].cumsum()

                weights_data = (round(pd.DataFrame(weights_results),2))
                weights_data["Date"] = results["Date"]

                final_results = pd.merge(results, weights_data, on="Date")
                final_results = final_results.set_index("Date")
                weights_data = weights_data.set_index("Date")


                data = price_data([benchmark_asset], initial_date,last_date, "Close")
                data = pd.DataFrame(data)
                data = data.loc[(data.index>=final_results.index.min()) & (data.index<=final_results.index.max())]
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
    st.write("")

    # Portfolio in real time

    st.header("Get your portfolio in real time", divider="gray")

    st.write("")
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

    condition_real_time = st.selectbox("Choose between maximizing sharpe ratio or reduce volatility for getting weights.", 
                                ["Maximize Sharpe Ratio", "Reduce Volatility","Expected return", "Volatility limit"])
    
    amount_selection_realtime = st.number_input("""Enter the expected return or maximum level of volatility 
                                         you want to specify (%) (only if you choose expected return or volatility limit)""", min_value=0, max_value=100, value=10, 
                                         help="""Enter a percentage value. If portfolio can not be optimized in one iteration, the optimization for that
                                         iteration is made based in minimizing volatility as mucho as possible""")

    rf_rate_realtime = st.number_input("""Specify the risk-free rate (%) (default is 0).""", 
                              min_value=0, max_value=100,value=0, help="""Enter a percentage value for the risk-free rate. This only applies when 
                              "Maximize sharpe ration" is selected""")

    amount_selection_realtime = amount_selection_realtime/100
    rf_rate_realtime = rf_rate_realtime/100

    if condition_real_time == "Maximize Sharpe Ratio":
        pf_condition_real_time = "sharpe"
    elif condition_real_time == "Reduce Volatility":
        pf_condition_real_time = "volatility"
    elif condition_real_time == "Expected return":
        pf_condition_real_time = "target_return"
    elif condition_real_time == "Volatility limit":
        pf_condition_real_time = "max_volatility"


    if st.button("Get weights"):
        with st.spinner('Optimizing portfolio...'):
            try:
                if own_portfolio_input and train_periods and period_type:
                
                        try:
                            train_periods = int(train_periods)
                            data_realtime = price_data(own_portfolio, initial_date,today, "Close")
                            real_time_weights, train_window = get_real_time_weights(data_realtime, initial_date, current_date, train_periods, period_type, pf_condition_real_time,
                                                                                    rf_rate_realtime, target_return=amount_selection_realtime, max_volatility=amount_selection_realtime)

                            final_weights = pd.DataFrame(round(real_time_weights,4)*100).rename(columns={0: 'Weights in %'})

                            final_weights = round(final_weights,2)
                            st.header("Weights for the last moment where you had had to rebalance your portfolio")
                            st.table(final_weights)

                            weights_plot_st(final_weights)

                            st.header("Last prices data")
                            st.write(train_window)
                        except:
                            st.error("Please, enter data properly")
            except:
                st.error("Please, enter data properly")

    st.write("")
    st.write("")
    st.write("")
    st.markdown(f"<p style='text-align:center;'>Developed by Edwar Valenzuela <br><br> <a href='{linkedin_url}'>Linkedin</a></p>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.caption("""Advertisement: The use of this application is the sole responsibility of the user. 
                    Portfolio and investment management entails financial risks and it is important to understand that results obtained in the past do not guarantee future results.
                    By using this application, the user agrees that the app developer will not be liable for any losses or damages resulting from investment decisions made based on the results of this application.""")




elif menu_selection == "Theory":
    # Explanation text
    st.title("Explanation")

    text = theory_txt

    st.markdown(text)