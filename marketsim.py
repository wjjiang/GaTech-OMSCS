"""2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'wwan9'

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 100000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    df_orders = pd.read_csv(orders_file)
    #print(df_orders)
    df_orders = df_orders.sort_values(by=['Date'])
    df_orders["Date"] = pd.to_datetime(df_orders["Date"])

    start_date = df_orders.Date.iloc[0]
    end_date = df_orders.Date.iloc[-1]
    symbol_list = list(set(df_orders.Symbol))
    # read the stock price into price table and add a cash column
    df_prices = get_data(symbol_list, pd.date_range(start_date, end_date))
    df_prices['Cash'] = 1
    df_prices.index.rename("Date", inplace = True)

    df_trades = pd.DataFrame(columns=df_prices.columns)
    df_trades['Date'] = list(df_prices.index)
    df_trades.set_index("Date", inplace=True)
    df_trades.fillna(value=0, inplace=True)

    df_holds = pd.DataFrame(columns=df_prices.columns)
    df_holds['Date'] = list(df_prices.index)
    df_holds.set_index("Date", inplace=True)
    df_holds.fillna(value=0, inplace=True)
    df_holds["Cash"][0] = start_val

    df_portfolio_val = pd.DataFrame(columns=["Date"])
    df_portfolio_val['Date'] = list(df_prices.index)
    df_portfolio_val.set_index("Date", inplace=True)
    df_portfolio_val.fillna(value=0, inplace=True)

    order_date = set(pd.to_datetime(df_orders.Date))
    prev_date = start_date

    for date, price_row in df_prices.iterrows():
        df_holds.loc[date, :] = df_holds.loc[prev_date, :]
        prev_date = date

        if date not in order_date:
            continue
        order_tmp = df_orders[df_orders.Date == date]
        tmp_position = df_holds.loc[date, :].copy()
        for index, order_row in order_tmp.iterrows():
            symbol = order_row.Symbol
            direct = 0
            price =  price_row[symbol]
            shares = order_row["Shares"]
            if order_row["Order"] == "BUY":
                direct = 1
            else:
                direct = -1
           # print(order_tmp)
            cost = shares * direct * price
            tmp_position[symbol] += direct * shares
            tmp_position["Cash"] += -1 * cost
           # print(tmp_position)
            tmp_price = df_prices.loc[date, :]
            #print(tmp_position * tmp_price)
            abs_sum = np.sum(np.abs(tmp_position[:-1]) * tmp_price[:-1])
            total_sum = np.sum(tmp_position * tmp_price)
            leverage = (abs_sum) / (total_sum)

            #print ("level: ", leverage)
            if (abs(tmp_position[symbol]) > 400):
                continue
            df_trades.loc[date, [symbol]] += shares * direct
            df_trades.loc[date, ["Cash"]] += -1 * cost
        df_holds.loc[date, :] += df_trades.loc[date,:]
        #print("************************")
        #print(df_holds.loc[date, :])
    trade_orders_df =pd.DataFrame()
    trade_orders_df = df_orders[df_orders["Open_Pos"] == 1][["Date", "Order"]]
    print("trade_orders_df: ", trade_orders_df)
    print("df_holds: ",df_holds)
    df_portfolio_val["Portfolio_vals"] = (df_prices * df_holds).sum(axis =1)
    return df_portfolio_val, trade_orders_df

def performance(portfolio_val):
    return_val = portfolio_val.pct_change()
    mean_return = return_val.mean()
    std_return = return_val.std()
    cum_return = (portfolio_val[-1] - portfolio_val[0]) / portfolio_val[0]

    return cum_return, mean_return, std_return

def best_bench_simulate():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    best_order = "./order/best_order.csv"
    bench_order = "./order/benchmark_order.csv"
    sv = 100000
    # Process orders
    portvals_best, _ = compute_portvals(orders_file = best_order, start_val = sv)
    portvals_bench, _= compute_portvals(orders_file = bench_order, start_val = sv)

    if isinstance(portvals_best, pd.DataFrame):
        portvals_best = portvals_best[portvals_best.columns[0]]
        portvals_bench = portvals_bench[portvals_bench.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    dates = portvals_bench.index[portvals_bench.index < portvals_best.index[0]]
    ts_tmp = pd.Series(sv, dates, name ="Portfolio_vals")
    print(ts_tmp.append(portvals_best))
    portvals_best = ts_tmp.append(portvals_best)

    start_date = portvals_bench.index[0]
    end_date = portvals_bench.index[-1]
    cum_ret, avg_daily_ret, std_daily_ret = performance(portvals_best)
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench = performance(portvals_bench)

    fig_df = pd.DataFrame(columns=["Date","BenchMark", "BestPortfolio"])
    fig_df["Date"] = portvals_bench.index
    fig_df["BenchMark"] = portvals_bench.values
    fig_df["BestPortfolio"] = portvals_best.values
    fig_df.set_index("Date", inplace = True)
    fig_df = fig_df / fig_df.iloc[0,:]
    ax_ma = fig_df[["BenchMark", "BestPortfolio"]].plot(color=['k','b'])
    ax_ma.get_figure().savefig('./figure/bench_best.jpg')

    print(fig_df)

    # Compare portfolio against $SPX
    print ("Date Range: {} to {}".format(start_date, end_date))
    print ()
    print ("Cumulative Return of BestPortfolio: {}".format(cum_ret))
    print ("Cumulative Return of BenchMark : {}".format(cum_ret_bench))
    print ()
    print ("Standard Deviation of BestPortfolio: {}".format(std_daily_ret))
    print ("Standard Deviation of BenchMark : {}".format(std_daily_ret_bench))
    print ()
    print ("Average Daily Return of BestPortfolio: {}".format(avg_daily_ret))
    print ("Average Daily Return of BenchMark : {}".format(avg_daily_ret_bench))
    print ()
    print ("Final Portfolio Value of BestPortfolio: {}".format(portvals_best[-1]))
    print ("Final Portfolio Value of BenchMark: {}".format(portvals_bench[-1]))


def ts_fillna(df_bench, df):
    dates_prev = df_bench.index[df_bench.index < df.index[0]]
    ts_tmp_prev = pd.Series(df[0], dates_prev, name ="Portfolio_vals")
    df = ts_tmp_prev.append(df)

    dates_after = df_bench.index[df_bench.index > df.index[-1]]
    ts_tmp_after = pd.Series(df[-1], dates_after, name ="Portfolio_vals")
    df = df.append(ts_tmp_after)
    return df

def manual_bench_simulate():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    best_order = "./order/order_manual.csv"
    bench_order = "./order/benchmark_order.csv"
    sv = 100000
    # Process orders
    portvals_best, trade_orders_df = compute_portvals(orders_file = best_order, start_val = sv)
    portvals_bench,_ = compute_portvals(orders_file = bench_order, start_val = sv)

    if isinstance(portvals_best, pd.DataFrame):
        portvals_best = portvals_best[portvals_best.columns[0]]
        portvals_bench = portvals_bench[portvals_bench.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    portvals_best = ts_fillna(portvals_bench, portvals_best)

    start_date = portvals_bench.index[0]
    end_date = portvals_bench.index[-1]
    cum_ret, avg_daily_ret, std_daily_ret = performance(portvals_best)
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench = performance(portvals_bench)


    fig_df = pd.DataFrame(columns=["Date","BenchMark", "ManualPortfolio"])
    fig_df["Date"] = portvals_bench.index
    fig_df["BenchMark"] = portvals_bench.values
    fig_df["ManualPortfolio"] = portvals_best.values
    fig_df.set_index("Date", inplace = True)
    fig_df = fig_df / fig_df.iloc[0,:]
    trade_orders_df.set_index("Date", inplace=True)
    ax_ma = fig_df.plot(color= ['k','b'])
    print(type(trade_orders_df))
    for index, row in trade_orders_df.iterrows():
        if (trade_orders_df.loc[index, "Order"] == "BUY"):
            ax_ma.axvline(x=index, color='g',linestyle='--')
        else:
            ax_ma.axvline(x=index, color='r',linestyle='--')
    ax_ma.get_figure().savefig('./figure/bench_Manual.jpg')

    # Compare portfolio against $SPX
    print ("Date Range: {} to {}".format(start_date, end_date))
    print ()
    print ("Cumulative Return of ManualPortfolio: {}".format(cum_ret))
    print ("Cumulative Return of BenchMark : {}".format(cum_ret_bench))
    print ()
    print ("Standard Deviation of ManualPortfolio: {}".format(std_daily_ret))
    print ("Standard Deviation of BenchMark : {}".format(std_daily_ret_bench))
    print ()
    print ("Average Daily Return of ManualPortfolio: {}".format(avg_daily_ret))
    print ("Average Daily Return of BenchMark : {}".format(avg_daily_ret_bench))
    print ()
    print ("Final Portfolio Value of ManualPortfolio: {}".format(portvals_best[-1]))
    print ("Final Portfolio Value of BenchMark: {}".format(portvals_bench[-1]))


def machine_manual_bench_simulate():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    machine_order = "./order/order_machine_test.csv"
    manual_order = "./order/order_manual_test.csv"
    bench_order = "./order/benchmark_order_test.csv"
    sv = 100000
    # Process orders
    portvals_machine, trade_orders_df = compute_portvals(orders_file = machine_order, start_val = sv)
    portvals_manual, _ = compute_portvals(orders_file = manual_order, start_val = sv)
    portvals_bench,_ = compute_portvals(orders_file = bench_order, start_val = sv)

    if isinstance(portvals_machine, pd.DataFrame):
        portvals_machine = portvals_machine[portvals_machine.columns[0]]
        portvals_manual = portvals_manual[portvals_manual.columns[0]]
        portvals_bench = portvals_bench[portvals_bench.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.

    portvals_machine = ts_fillna(portvals_bench, portvals_machine)
    portvals_manual = ts_fillna(portvals_bench, portvals_manual)

    start_date = portvals_bench.index[0]
    end_date = portvals_bench.index[-1]
    cum_ret_machine, avg_daily_ret_machine, std_daily_ret_machine = performance(portvals_machine)
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench = performance(portvals_bench)
    cum_ret_manual, avg_daily_ret_manual, std_daily_ret_manual = performance(portvals_manual)


    fig_df = pd.DataFrame(columns=["Date","BenchMark", "ManualPortfolio", "MachinePortfolio"])
    fig_df["Date"] = portvals_bench.index
    fig_df["BenchMark"] = portvals_bench.values
    fig_df["ManualPortfolio"] = portvals_manual.values
    fig_df["MachinePortfolio"] = portvals_machine.values
    fig_df.set_index("Date", inplace = True)
    trade_orders_df.set_index("Date", inplace=True)


    ax_ma = fig_df.plot(color= ['k','b','g'])
    print(type(trade_orders_df))
    for index, row in trade_orders_df.iterrows():
        if (trade_orders_df.loc[index, "Order"] == "BUY"):
            ax_ma.axvline(x=index, color='g',linestyle='--')
        else:
            ax_ma.axvline(x=index, color='r',linestyle='--')
    ax_ma.get_figure().savefig('./figure/bench_Manual_Machine.jpg')



    # Compare portfolio against $SPX
    print ("Date Range: {} to {}".format(start_date, end_date))
    print ()
    print ("Cumulative Return of MachinePortfolio: {}".format(cum_ret_machine))
    print ("Cumulative Return of ManualPortfolio: {}".format(cum_ret_manual))
    print ("Cumulative Return of BenchMark : {}".format(cum_ret_bench))
    print ()
    print ("Standard Deviation of MachinePortfolio: {}".format(std_daily_ret_machine))
    print ("Standard Deviation of ManualPortfolio: {}".format(std_daily_ret_manual))
    print ("Standard Deviation of BenchMark : {}".format(std_daily_ret_bench))
    print ()
    print ("Average Daily Return of MachinePortfolio: {}".format(avg_daily_ret_machine))
    print ("Average Daily Return of ManualPortfolio: {}".format(avg_daily_ret_manual))
    print ("Average Daily Return of BenchMark : {}".format(avg_daily_ret_bench))
    print ()
    print ("Final Portfolio Value of MachinePortfolio: {}".format(portvals_machine[-1]))
    print ("Final Portfolio Value of ManualPortfolio: {}".format(portvals_manual[-1]))
    print ("Final Portfolio Value of BenchMark: {}".format(portvals_bench[-1]))

if __name__ == "__main__":
    #best_bench_simulate()
    #manual_bench_simulate()
    machine_manual_bench_simulate()
