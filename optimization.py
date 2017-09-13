"""MC1-P2: Optimize a portfolio."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
import scipy.optimize as spo
from util import get_data, plot_data


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    # Automatically adds SPY
    prices_all = get_data(syms, dates)
    # Fill N/A: ffill -> bfill
    prices_all.fillna(method="ffill", inplace=True)
    prices_all.fillna(method="bfill", inplace=True)
    # Select only portfolio symbols
    prices = prices_all[syms]
    # Select only SPY, for comparison later
    prices_SPY = prices_all['SPY']

    # Get daily portfolio value
    prices_SPY = (prices_SPY.pct_change().fillna(0)+1).cumprod()
    
    # find the allocations for the optimal portfolio
    ## Step 1: initialize equally-weighted allocation
    allocs_init = np.ones(len(syms)) / len(syms)
    ## Step 2: optimize allocation by minimizing sddr
    #### Step 2.1: define some parameters
    rfr = 0.0
    sf = 252.0
    cons = ({"type":'eq', 'fun': lambda x: np.sum(x)-1})
    bounds = ((0.0,1.0),) * len(syms)
    
    #### Step 2, Function 1: Return the portfolio value only
    def get_portfolio_value(allocs):
        ## Step 2.1: calculate weighted prices
        weighted_prices = prices.divide(prices.ix[0]) * allocs
        ## Step 2.2: calculate cumulative return time-series
        port_val = weighted_prices.apply(sum, axis=1)
        ## Step 2.3: return
        return port_val
    
    #### Step 3, Function 2: Return portfolio statistics
    def portfolio_stats(port_val):
        ## Step 3.1: calculate cumulative return
        cr = port_val[-1] - 1
        ## Step 3.2: calculate daily return time-series, notice no N/A's exist
        dr = port_val.pct_change()
        ## Step 3.3: calculate average of daily return
        adr = dr.mean()
        ## Step 3.4: calculate standard derivation of daily return
        sddr = dr.std()
        ## Step 3.5: calculate Sharpe's ratio
        sr = np.sqrt(sf) * (adr-rfr) / sddr
        ## Step 3.6: return
        return cr, adr, sddr, sr
    
    #### Step 4, Function 3: Return sddr only
    def sddr_func(allocs):
        ## Step 4.1: get cumulative return time-series from function 1
        port_val = get_portfolio_value(allocs)
        ## Step 4.2: calculate daily return time-series, notice no N/A's exist
        dr = port_val.pct_change()
        ## Step 4.3: calculate standard derivation of daily return
        sddr = dr.std()
        ## Step 4.4: return
        return sddr
    
    #### Step 5: Minimize sddr
    min_result = spo.minimize(sddr_func, x0=allocs_init, method='SLSQP', \
                              bounds=bounds, constraints=cons)
    allocs = min_result.x
    
    ## Step 3: calculate portfolio statistics based on optimized weights
    port_val = get_portfolio_value(allocs)
    cr, adr, sddr, sr = portfolio_stats(port_val)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], 
                            keys=['Portfolio', 'SPY'], 
                            axis=1)
        plt.figure(figsize=(20,15))
        plt.plot(df_temp['Portfolio'], lw=3, label='Portfolio')
        plt.plot(df_temp['SPY'], lw=3, label='SPY')
        plt.legend(loc=0, fontsize=25)
        plt.grid(True, linestyle='--')
        plt.title("Daily Portfolio Value and SPY", fontsize=30)
        plt.xlabel("Date", fontsize=24)
        plt.ylabel("Price", fontsize=24)
        plt.xticks(rotation=45, size=18)
        plt.yticks(size=18)
        plt.savefig('report.pdf')

    return allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, \
                                                        ed = end_date, \
                                                        syms = symbols, \
                                                        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
