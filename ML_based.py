import pandas as pd
import numpy as np
import datetime as dt
import util
import indicators
import RTLearner

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

def roc_rsi_ma_RTLeaner_strat(df, df_test=None):
    df["return"] = df["AAPL"].pct_change(21)
    df["return"] = df["return"].shift(-21)
    df["label"] = 0
    df.loc[df["return"] < -0.17, "label"] = -1
    df.loc[df["return"] >  0.17, "label"] = 1
    df.dropna(inplace=True)
    features = ["ROC_5", "RSI_14", "MA_5_diff_15"]
    df[features] = (df[features] - df[features].mean()) / df[features].std()
    X_feature = df[features]
    Y_label = df["label"]

    rt_leaner = RTLearner.RTLearner(leaf_size = 5)
    rt_leaner.addEvidence(dataX = X_feature.values, dataY = Y_label.values)
    Y_predict_label = rt_leaner.query(X_feature.values)
    df["signal"] = Y_predict_label
    df["Direction"] = 0
    df.loc[df["signal"] < 0, "Direction"] = -1
    df.loc[df["signal"] > 0 , "Direction"] = 1

    df[["ROC_5","RSI_14","label"]].to_csv("machine_before_scatter.csv")
    df[["ROC_5","RSI_14","signal"]].to_csv("machine_after_scatter.csv")

    order_df = order_generator(df)
    order_df.to_csv("./order/order_machine.csv")

    if  df_test is None :
        df_test[features] = (df_test[features] - df_test[features].mean()) / df_test[features].std()
        df_test["label"] = 0
        X_test_feature = df_test[features]
        Y_test_predict_label = rt_leaner.query(X_test_feature.values)
        df_test["signal"] = Y_test_predict_label
        df_test["Direction"] = 0
        df_test.loc[df_test["signal"] < 0, "Direction"] = -1
        df_test.loc[df_test["signal"] > 0 , "Direction"] = 1
        order_df_test = order_generator(df_test)
        order_df_test.to_csv("./order/order_machine_test.csv")
    return rt_leaner


if __name__ == "__main__":
    symbols_train_df = indicators.indicators_generate(sd_train = dt.datetime(2008,1,1), ed_train = dt.datetime(2009,12,31), symbols=["AAPL"])
    symbols_test_df = indicators.indicators_generate(sd_train = dt.datetime(2010,1,1), ed_train = dt.datetime(2011,12,31), symbols=["AAPL"])

    rt_leaner = roc_rsi_ma_RTLeaner_strat(symbols_train_df, symbols_test_df)
