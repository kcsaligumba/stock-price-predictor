import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def naive_last_value(df: pd.DataFrame) -> pd.Series:
    # Predict next close = today's close
    return df["Close"].shift(0)

def sma_crossover(df: pd.DataFrame) -> pd.Series:
    # Long when SMA_5 > SMA_20 else hold last close
    pred = df["Close"].copy()
    mask = df["SMA_5"] > df["SMA_20"]
    pred[~mask] = df["SMA_20"][~mask]  # crude proxy
    return pred

def evaluate(y_true, y_pred, name="model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")

def main(data_dir: str, ticker: str):
    path = Path(data_dir) / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    # Align targets
    y_true = df["Target_Close_t+1"]
    preds_naive = naive_last_value(df)
    preds_sma = sma_crossover(df)
    # Shift preds to match t+1
    preds_naive = preds_naive.shift(1).dropna()
    preds_sma = preds_sma.shift(1).dropna()
    y_eval = y_true.loc[preds_naive.index]
    print(f"Evaluating on {len(y_eval)} samples for {ticker}")
    evaluate(y_eval, preds_naive.loc[y_eval.index], name="Naive")
    evaluate(y_eval, preds_sma.loc[y_eval.index], name="SMA")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--ticker", default="AAPL")
    args = p.parse_args()
    main(args.data, args.ticker)
