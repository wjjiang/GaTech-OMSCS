import pandas as pd
import numpy as np
import datetime as dt
import util

def compare_best_benchmark(df, initial_val = 100000):
    benchmark_best_df = df[["AAPL"]].copy()

    benchmark_best_df["BenchMark"] = benchmark_best_df["AAPL"] /  benchmark_best_df["AAPL"][0]
    bench_stock_val = benchmark_best_df["BenchMark"] * 200 *benchmark_best_df["AAPL"][0]
    bench_money = initial_val - 200 *benchmark_best_df["AAPL"][0]
    bench_portfolio = bench_stock_val + bench_money
    print("bench_portfolio: ", bench_portfolio)

    benchmark_best_df["BestPortfolio"] = abs(benchmark_best_df[["AAPL"]].shift(1) - benchmark_best_df[["AAPL"]]) / benchmark_best_df[["AAPL"]]
    benchmark_best_df["BestPortfolio"][0] = 0
    portfolio_benchmark = bench_stock_val + bench_money
    portfolio_best = (benchmark_best_df["BestPortfolio"]*200 *benchmark_best_df["AAPL"][0]).cumsum() + initial_val
    


if __name__ == "__main__":
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2012,1,1)
    dates = pd.date_range(sd_train, ed_train)
    data_df = util.get_data(symbols=["AAPL"], dates)
