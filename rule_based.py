import pandas as pd
import numpy as np
import datetime as dt
import util
import indicators

def order_generator(signal_df, holding_period = 21):
    order_df = pd.DataFrame(columns=["Date", "Symbol", "Order", "Shares", "Open_Pos"])
    count = -1
    pos = 0
    empty_pos = True
    end_date = signal_df.index[-1]
    for index, row in signal_df.iterrows():

        if not empty_pos and (count == 0 or end_date == index):
            direct_tmp = order_df.loc[pos-1, "Order"]
            direct = "BUY"
            if (direct_tmp == "BUY"):
                direct = "SELL"
            order_df.loc[pos] = [index, "AAPL", direct, 200, 0]
            pos += 1
            empty_pos = True

        if count < 0 and row["Direction"] != 0:
            direct = "BUY"
            if row["Direction"] < 0 :
                direct = "SELL"
            order_df.loc[pos] = [index, "AAPL", direct, 200, 1]
            pos += 1
            count = holding_period
            empty_pos = False
        count -= 1

    return order_df

def roc_rsi_ma_strat(df):
    df["Direction"] = 0
    condition_short = (df["ROC_5"] < -0.04) & (df["RSI_14"] < 40) & (df["MA_5_diff_15"] < -0.015)
    condition_long = (df["ROC_5"] > 0.05) & (df["RSI_14"] > 65) & (df["MA_5_diff_15"] > 0.015)
    df.loc[condition_short, "Direction"] = -1
    df.loc[condition_long, "Direction"] = 1
    features = ["ROC_5","RSI_14", "MA_5_diff_15"]
    df[features] = (df[features] - df[features].mean()) / df[features].std()

    fig_scatter = df[["ROC_5", "RSI_14", "Direction"]]
    print(fig_scatter)
    fig_scatter.to_csv("manual_scatter.csv")

    order_df = order_generator(signal_df=df, holding_period = 21)
    print("order_df: ", order_df)

    order_df.to_csv("./order/order_manual.csv")
    return df


if __name__ == "__main__":
    symbols_train_df = indicators.indicators_generate(sd_train = dt.datetime(2008,1,1), ed_train = dt.datetime(2009,12,31), symbols=["AAPL"])
    symbols_test_df = indicators.indicators_generate(sd_train = dt.datetime(2010,1,1), ed_train = dt.datetime(2011,12,31), symbols=["AAPL"])
    roc_rsi_ma_strat(symbols_train_df)
