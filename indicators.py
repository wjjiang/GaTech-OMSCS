import pandas as pd
import numpy as np
import datetime as dt
import util

class TA():
    def __init__(self, df):
        self.df = df.copy()

    def roc(self, lag):
        return (self.df - self.df.shift(lag)) / self.df

    def rsi(self, period=14):
        ret = (self.df - self.df.shift(1)) / self.df
        def _rsi(x):
            up = np.sum(x[x > 0])
            down = np.sum(x[x < 0])
            rs = up / abs(down)
            return 100 - 100 /(1 + rs)
        res = pd.rolling_apply(ret, period, _rsi)
        return res

    def ma(self, period=5):
        return pd.rolling_mean(self.df, period)

def indicators_generate(sd_train = dt.datetime(2008,1,1), ed_train = dt.datetime(2009,12,31), symbols=["AAPL"]):
    dates_train = pd.date_range(sd_train, ed_train)
    data_train_df = util.get_data(symbols, dates_train)

    symbols_train_df = data_train_df[symbols]
    ta_transfer = TA(symbols_train_df)
    symbols_train_df["ROC_5"] = ta_transfer.roc(5)
    symbols_train_df["RSI_14"] = ta_transfer.rsi(14)
    symbols_train_df["MA_5"] = ta_transfer.ma(5)
    symbols_train_df["MA_15"] = ta_transfer.ma(15)
    symbols_train_df["MA_5_diff_15"] = (symbols_train_df["MA_5"] - symbols_train_df["MA_15"])/symbols_train_df["AAPL"]

    symbols_train_df.dropna(inplace=True)

    #Normalize for figures
    symbols_fig_df = symbols_train_df.copy()
    symbols_feature_df = symbols_train_df[["ROC_5", "RSI_14", "MA_5_diff_15"]].copy()


    symbols_fig_df = symbols_fig_df / symbols_fig_df.iloc[0,:]
    ax_ma = symbols_fig_df[["AAPL","MA_5","MA_15","MA_5_diff_15"]].plot()
    ax_ma.get_figure().savefig('./figure/ma.jpg')

    ax_roc = symbols_fig_df[["AAPL","ROC_5"]].plot()
    ax_roc.get_figure().savefig('./figure/roc.jpg')

    ax_rsi = symbols_fig_df[["AAPL","RSI_14"]].plot()
    ax_rsi.get_figure().savefig('./figure/rsi.jpg')

    ax_ma = symbols_fig_df[["AAPL","MA_5","MA_15","MA_5_diff_15"]].plot()
    ax_ma.get_figure().savefig('./figure/ma.jpg')

    return symbols_train_df

if __name__ == "__main__":
    symbols_train_df = indicators_generate(sd_train = dt.datetime(2008,1,1), ed_train = dt.datetime(2009,12,31), symbols=["AAPL"])
    print("symbols_train_df: ", symbols_train_df)
