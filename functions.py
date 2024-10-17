import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import subprocess
#subprocess.run(["git", "clone", "https://github.com/robertmartin8/PyPortfolioOpt.git"])

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
    price = price.tz_localize(None)
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

def line_chart_2(backtest_results, benchmark_data, var1, var2, title, width=1000, height=500, y_title=""):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=backtest_results.index, y=backtest_results[var1], mode='lines', name=var1, fill="tozeroy", line=dict(color="rgba(39, 255, 54, 0.8)")))
    fig.add_trace(go.Scatter(x=benchmark_data.index, y=benchmark_data[var2], mode='lines', name=var2, fill="tozeroy", line=dict(color="rgba(24, 172, 230, 0.5)")))

    fig.update_layout(
    title=title,
    xaxis_title='Date',
    yaxis_title=y_title,
    template='plotly_dark',
    height=height,
    width=width
    )

    fig.show()

def line_chart_st(data, variables_list, title, width=600, height=500, y_title=""):
    plt.figure(figsize=(width/100, height/100))  # Convertir las dimensiones de píxeles a pulgadas
    for var in variables_list:
        plt.plot(data.index, data[var], label=var)

    plt.title(title, color='white')  # Establecer el color del título
    plt.xlabel('Date', color='white')  # Establecer el color del texto del eje x
    plt.ylabel(y_title, color='white')  # Establecer el color del texto del eje y
    plt.tick_params(axis='x', colors='white')  # Establecer el color de los valores del eje x
    plt.tick_params(axis='y', colors='white')  # Establecer el color de los valores del eje y
    plt.legend()
    plt.style.use('dark_background')  # Estilo oscuro
    plt.gca().set_facecolor('#121212')  # Ajustar el color de fondo
    plt.tight_layout()

    st.pyplot(plt.gcf())

def line_chart_2_st(backtest_results, benchmark_data, var1, var2, title, width=6, height=5, y_title=""):
    plt.figure(figsize=(width, height))

    plt.plot(backtest_results.index, backtest_results[var1], color="lime", label=var1, alpha=0.8)
    plt.plot(benchmark_data.index, benchmark_data[var2], color="deepskyblue", label=var2, alpha=0.5)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(y_title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.gca().set_facecolor('#121212')  
    plt.xticks(color='white')  
    plt.yticks(color='white')  

    plt.tight_layout()
    st.pyplot(plt.gcf())

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
            test_start = train_end
            test_end = test_start + pd.DateOffset(weeks=test_periods)
        elif period_type == "months":
            train_end = start_date + pd.DateOffset(months=train_periods)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_periods)
        elif period_type == "years":
            train_end = start_date + pd.DateOffset(years=train_periods)
            test_start = train_end
            test_end = test_start + pd.DateOffset(years=test_periods)
        else:
            raise ValueError("Period type must be 'weeks', 'months', or 'years'")
        
        if test_end > end_date:
            break
        
        train_window = data.loc[(data.index >= start_date) & (data.index < train_end)]
        test_window = data.loc[(data.index >= test_start) & (data.index < test_end)]
        
        windows_train.append(train_window)
        windows_test.append(test_window)
        
        if period_type == "weeks":
            start_date += pd.DateOffset(weeks=test_periods)  # Avanzar según el período de prueba
        elif period_type == "months":
            start_date += pd.DateOffset(months=test_periods)  # Avanzar según el período de prueba
        elif period_type == "years":
            start_date += pd.DateOffset(years=test_periods)  # Avanzar según el período de prueba
    
    return windows_train, windows_test

def get_results_portfolio(train_window, test_window, rf_rate, pf_condition="sharpe", target_return=None, max_volatility=None):
    mu = expected_returns.mean_historical_return(train_window)
    sigma = risk_models.sample_cov(train_window)
    
    ef = EfficientFrontier(mu, sigma)

    try: 
        if pf_condition == "sharpe":
            condition = ef.max_sharpe(risk_free_rate=rf_rate)
        elif pf_condition == "volatility":
            condition = ef.min_volatility()
        elif pf_condition == "target_return" and target_return is not None:
            condition = ef.efficient_return(target_return)
        elif pf_condition == "max_volatility" and max_volatility is not None:
            condition = ef.efficient_risk(max_volatility)
    except Exception as e:
        try:
        #print("Error during portfolio optimization:", e)
        # Si ocurre un error durante la optimización de la cartera, establecemos pf_condition en "volatility" y optimizamos en función de la volatilidad mínima
            pf_condition = "volatility"
            condition = ef.min_volatility()
        except:
            pf_condition = "equal_weights"
            n_assets = len(mu)
            equal_weights = np.array([1/n_assets] * n_assets)
            condition = equal_weights

    if pf_condition == "equal_weights":
        weights_array = pd.Series(equal_weights, index=mu.index)
    else:
        weights_array = pd.Series(condition, index=mu.index)

    returns = test_window.pct_change()
    
    initial_value = 10000
    
    try:
        returns["pf_returns"] = returns.dot(weights_array) * 100
        returns = round(returns, 2)
        
        profit = returns["pf_returns"].sum()
        
        start_dt = returns.index.min()
        date_result = returns.index[-1]
        error = False
    except:
        profit = 0
        start_dt = returns.index.min()
        date_result = returns.index[-1]
        error = True

    return profit, date_result, error, weights_array, pf_condition, start_dt

def get_real_time_weights(data, initial_date, current_date, train_periods, period_type="months", pf_condition = "sharpe",
                          rf_rate=0, target_return=None, max_volatility=None):
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
            if pf_condition == "sharpe":
                condition = ef.max_sharpe(risk_free_rate=rf_rate)
            elif pf_condition == "volatility":
                condition = ef.min_volatility()
            elif pf_condition == "target_return" and target_return is not None:
                condition = ef.efficient_return(target_return)
            elif pf_condition == "max_volatility" and max_volatility is not None:
                condition = ef.efficient_risk(max_volatility)
        except Exception as e:
            try:
            #print("Error during portfolio optimization:", e)
            # Si ocurre un error durante la optimización de la cartera, establecemos pf_condition en "volatility" y optimizamos en función de la volatilidad mínima
                pf_condition = "volatility"
                condition = ef.min_volatility()
            except:
                pf_condition = "equal_weights"
                n_assets = len(mu)
                equal_weights = np.array([1/n_assets] * n_assets)
                condition = equal_weights

        if pf_condition == "equal_weights":
            weights_array = pd.Series(equal_weights, index=mu.index)
        else:
            weights_array = pd.Series(condition, index=mu.index)

        #try:
        #    weights = condition
        #    weights_array = pd.Series(weights)
        #except Exception as e:
        #    # Use equal weights in error case
        #    try:
        #        weights = [1 / len(mu)] * len(mu)
        #        weights_array = pd.Series(weights, index=mu.index)
        #    except:
        #        # If weights cannot be calculated, set all weights to 0
        #        weights_array = pd.Series([0] * len(mu), index=mu.index)
        
        if period_type == "weeks":
            start_date += pd.DateOffset(weeks=1)
        elif period_type == "months":
            start_date += pd.DateOffset(months=1)
        elif period_type == "years":
            start_date += pd.DateOffset(years=1)
    
    return weights_array, train_window

def weights_plot(final_weights):

    fig = go.Figure()

    fig.add_trace(go.Bar(x=final_weights.index,
                         y=final_weights['Weights in %'],
                         marker_color='rgba(39, 93, 245, 0.8)',
                         text=final_weights['Weights in %'],
                         textposition='auto'))

    fig.update_layout(title='Asset Weights in the Portfolio',
                      xaxis=dict(title='Assets', tickangle=45),
                      yaxis=dict(title='Weights (%)'),
                      template='plotly_dark',
                      autosize=False,
                      width=800,
                      height=500)

    fig.show()

    
def weights_plot_st(final_weights):
    plt.figure(figsize=(10, 6))
    plt.bar(final_weights.index, final_weights['Weights in %'], color='tab:blue')
    plt.title('Asset Weights in the Portfolio', fontsize=16, color='white')
    plt.xlabel('Assets', fontsize=12, color='white')
    plt.ylabel('Weights (%)', fontsize=12, color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
 
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().set_facecolor('#121212') 
    for i, val in enumerate(final_weights['Weights in %']):
        plt.text(i, val + 0.5, f'{val}%', color='white', ha='center', fontsize=15)
 

    plt.tight_layout()
    # Mostrar el gráfico
    st.pyplot(plt.gcf())