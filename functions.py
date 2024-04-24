import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib as plt


from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

import streamlit as st
import plotly.express as px

from pandas.tseries.offsets import MonthEnd, YearEnd, Week



def price_data(tickers, start_date, end_date, column):

    price = yf.download(tickers, start=start_date, end=end_date)[column]
    price.index = pd.to_datetime(price.index)
    return price

def line_chart(data, variables_list, title, width=1000, height=500, y_title=""):
    fig = go.Figure()

    for var in variables_list:
        fig.add_trace(go.Scatter(x=data.index, y=data[var], mode='lines', name=var))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_title,
        template='plotly_dark',
        height=height,
        width=width
    )

    fig.show()

def line_chart_st(data, variables_list, title, width=1000, height=500, y_title=""):
    fig = px.line()

    for var in variables_list:
        fig.add_scatter(x=data.index, y=data[var], mode='lines', name=var)

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_title,
        template='plotly_dark',
        height=height,
        width=width
    )

    st.plotly_chart(fig)

def max_drawdown(columns):
    max_profit = columns.expanding().max()  
    drawdown = (max_profit - columns)  
    max_drawdown = drawdown.max()  
    return max_drawdown

def generate_windows(data, initial_date, last_date, train_periods, test_periods, period_type="months"):
    windows_train = []
    windows_test = []
    start_date = pd.to_datetime(initial_date)
    end_date = pd.to_datetime(last_date)
    
    while True:
        if period_type == "weeks":
            train_end = start_date + pd.DateOffset(weeks=train_periods)
            test_end = train_end + pd.DateOffset(weeks=test_periods)
        elif period_type == "months":
            train_end = start_date + pd.DateOffset(months=train_periods)
            test_end = train_end + pd.DateOffset(months=test_periods)
        elif period_type == "years":
            train_end = start_date + pd.DateOffset(years=train_periods)
            test_end = train_end + pd.DateOffset(years=test_periods)
        else:
            raise ValueError("Period type must be 'weeks', 'months', or 'years'")
        
        if test_end > end_date:
            break
        
        train_window = data.loc[(data.index >= start_date) & (data.index < train_end)]
        test_window = data.loc[(data.index >= train_end) & (data.index < test_end)]
        
        windows_train.append(train_window)
        windows_test.append(test_window)
        
        if period_type == "weeks":
            start_date += pd.DateOffset(weeks=1)
        elif period_type == "months":
            start_date += pd.DateOffset(months=1)
        elif period_type == "years":
            start_date += pd.DateOffset(years=1)
    
    return windows_train, windows_test

def get_results_portfolio(train_window, test_window):

    mu = expected_returns.mean_historical_return(train_window)
    sigma = risk_models.sample_cov(train_window)
    
    ef = EfficientFrontier(mu, sigma)

    try:
        weights = ef.max_sharpe()
        weights_array = pd.Series(weights)
    except Exception as e:
        # Use equal weights in error case
        try:
            weights = [1 / len(mu)] * len(mu)
            weights_array = pd.Series(weights, index=mu.index)
        except:
            # If weights cannot be calculated, set all weights to 0
            weights_array = pd.Series([0] * len(mu), index=mu.index)

    returns = test_window.pct_change()
    
    initial_value = 10000
    
    try:
        returns["pf_returns"] = returns.dot(weights_array) * 100
        returns = round(returns, 2)
        
        profit = returns["pf_returns"].sum()
        date_result = returns.index[-1]
        error = False
    except:
        profit = 0
        date_result = returns.index[-1]
        error = True

    return profit, date_result, error, weights_array

def get_real_time_weights(data, initial_date, current_date, train_periods, period_type="months"):
    start_date = pd.to_datetime(initial_date)
    end_date = pd.to_datetime(current_date)
    
    while True:
        if period_type == "weeks":
            train_end = start_date + pd.DateOffset(weeks=train_periods)
        elif period_type == "months":
            train_end = start_date + pd.DateOffset(months=train_periods)
        elif period_type == "years":
            train_end = start_date + pd.DateOffset(years=train_periods)
        else:
            raise ValueError("Period type must be 'weeks', 'months', or 'years'")
        
        if train_end > end_date:
            break
        
        train_window = data.loc[(data.index >= start_date) & (data.index < train_end)]
        
        mu = expected_returns.mean_historical_return(train_window)
        sigma = risk_models.sample_cov(train_window)
        
        ef = EfficientFrontier(mu, sigma)

        try:
            weights = ef.max_sharpe()
            weights_array = pd.Series(weights)
        except Exception as e:
            # Use equal weights in error case
            try:
                weights = [1 / len(mu)] * len(mu)
                weights_array = pd.Series(weights, index=mu.index)
            except:
                # If weights cannot be calculated, set all weights to 0
                weights_array = pd.Series([0] * len(mu), index=mu.index)
        
        if period_type == "weeks":
            start_date += pd.DateOffset(weeks=1)
        elif period_type == "months":
            start_date += pd.DateOffset(months=1)
        elif period_type == "years":
            start_date += pd.DateOffset(years=1)
    
    return weights_array, train_window
