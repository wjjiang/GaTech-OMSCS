"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'wjiang84'

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object
    # Your code should work correctly with either input
    
    # Step 1: read csv
    df_orders = pd.read_csv(orders_file, parse_dates=True, na_values=['nan'])
    ## Step 1.1: sort by date
    df_orders = df_orders.sort_values(by=["Date"])
    ## Step 1.2: make sure column 'Date' has format of datetime yyyy-mm-dd
    df_orders["Date"] = pd.to_datetime(df_orders["Date"])
    
    # Step 2: read stock data from database
    ## Step 2.1: get start and end date
    start_date = df_orders["Date"].iloc[0]
    end_date = df_orders["Date"].iloc[-1]
    ## Step 2.2: get symbols w/o duplication, save in type of 'list'
    symbols = list(set(df_orders["Symbol"]))
    ## Step 2.3: read stock data from database by calling get_data from util.py
    df_prices = get_data(symbols, pd.date_range(start_date, end_date))
    ## Step 2.4: add additional column ['Cash']
    df_prices["CASH"] = 1.0
    ## Step 2.5: Fill N/A: ffill -> bfill
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    ## Rename index (date) to be able to quote
    df_prices.index.rename("Date", inplace = True)
    
    # Step 3: build df_trade data frame, to track changes per buy/sell occurs
    df_trades = pd.DataFrame(columns = df_prices.columns)
    df_trades["Date"] = list(df_prices.index)
    df_trades.set_index("Date", inplace=True)
    df_trades.fillna(value=0.0, inplace=True)
    
    # Step 4: build df_holdings data frame, to track the holding of each stock at each day
    df_holdings = df_trades.copy()
    df_holdings["CASH"].iloc[0] = start_val
    
    # Step 5: build df_values data frame, each element = df_prices * df_holdings
    #df_values = df_trades.copy()
    
    # Step 6: build df_portfolio_val = sum over columns of df_values
    df_portfolio_val = pd.DataFrame(columns = ["Date", "Portfolio"])
    df_portfolio_val["Date"] = list(df_prices.index)
    df_portfolio_val.set_index("Date", inplace=True)
    df_portfolio_val.fillna(value=0.0, inplace=True)
    
    # Step 7: set up for-loop to calculate values for each data frame
    ## Step 7.1: prev_date records the previous date for each loop
    prev_date = start_date
    ## Step 7.2: record order dates w/o duplication
    order_dates = set(pd.to_datetime(df_orders["Date"]))
    ## Step 7.3: for-loop
    for curr_date, price_trading in df_prices.iterrows():
        ### Step 7.3.1: copy previous date's values to current date
        df_holdings.loc[curr_date, :] = df_holdings.loc[prev_date, :]
        prev_date = curr_date
    
        ### Step 7.3.2: skip non-order-dates
        if curr_date not in order_dates:
            continue

        ### Step 7.3.3: get current position and order
        curr_order = df_orders[df_orders["Date"] == curr_date]
        #curr_holdings = df_holdings.loc[curr_date, :].copy()

        ### Step 7.3.4: loop over each element in the current order
        for index, order_row in curr_order.iterrows():
			#### Step 7.3.4.1: find the symbol of the specific stock that's traded on "curr_date"
			#### then return its adjusted close value
            symbol = order_row["Symbol"]
            price = price_trading[symbol]
			#### Step 7.3.4.2: return the share of the specific stock that's traded on "curr_date"
            shares = order_row["Shares"]
			#### Step 7.3.4.3: the amount of cash used to buy/sell
            cash_change = shares * price
			#### Step 7.3.4.4: calculate transaction cost, consisting of market impact and commission
            market_impact = cash_change * impact
            transaction_cost = market_impact + commission
            #### Step 7.3.4.5: distinguish long/short
			if order_row["Order"] == "BUY":
				long_short = 1
			else:
				long_short = -1
			#### Step 7.3.4.6: update curr_holdings
			#curr_holdings[symbol] += (shares * long_short)
			#curr_holdings["CASH"] -= (cash_change*long_short + transaction_cost)
			#### Step 7.3.4.7: update df_trades
			df_trades.loc[curr_date, [symbol]] += (shares * long_short)
			df_trades.loc[curr_date, ["CASH"]] -= (cash_change*long_short + transaction_cost)

        ### Step 7.3.5: update df_holdings
        df_holdings.loc[curr_date, :] += df_trades.loc[curr_date, :]

    # Step 8: build df_values data frame, each element = df_prices * df_holdings
    df_values = df_prices * df_holdings

    # Step 9: calculate portfolio-level value
    df_portfolio_val["Portfolio"] = df_values.sum(axis=1)

    df_prices.to_csv("df_prices.csv")
    df_trades.to_csv("df_trades.csv")
    df_holdings.to_csv("df_holdings.csv")
    df_values.to_csv("df_values.csv")

    return df_portfolio_val

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()