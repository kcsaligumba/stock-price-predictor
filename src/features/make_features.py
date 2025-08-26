import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: Date, Open, High, Low, Close, Volume
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = SMAIndicator(df["Close"], window=5).sma_indicator()
    df["SMA_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    df["RSI_14"] = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["Target_Close_t+1"] = df["Close"].shift(-1)
    df["Target_Return_t+1"] = df["Return"].shift(-1)
    return df.dropna().reset_index(drop=True)

def process_dir(input_dir: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for csv in Path(input_dir).glob("*.csv"):
        df = pd.read_csv(csv, parse_dates=["Date"])
        feats = build_features(df)
        feats.to_csv(Path(output_dir) / csv.name, index=False)
        print(f"Wrote {csv.name}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw")
    p.add_argument("--output", default="data/processed")
    args = p.parse_args()
    process_dir(args.input, args.output)
