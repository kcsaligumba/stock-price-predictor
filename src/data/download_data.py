import argparse
import os
from pathlib import Path
import yfinance as yf
import pandas as pd

def fetch(tickers, period="5y", interval="1d", out_dir="data/raw"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for t in tickers:
        print(f"Downloading {t}...")
        df = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            print(f"Warning: no data for {t}")
            continue
        df.reset_index().to_csv(os.path.join(out_dir, f"{t}.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"])
    p.add_argument("--period", default="5y")
    p.add_argument("--interval", default="1d")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    fetch(args.tickers, args.period, args.interval, args.out)
