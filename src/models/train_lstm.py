import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class WindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_cols, target_col="Target_Close_t+1", lookback=30):
        self.X = df[input_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.lb = lookback

    def __len__(self):
        return len(self.X) - self.lb

    def __getitem__(self, idx):
        x = self.X[idx:idx+self.lb]
        y = self.y[idx+self.lb-1]  # predict next close at last step
        return torch.from_numpy(x), torch.tensor(y)

class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1:, :]  # last step
        return self.head(out).squeeze(-1).squeeze(-1)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total += loss.item() * len(xb)
    return total / len(loader.dataset)

def main(data_dir: str, ticker: str, epochs: int, lookback: int, batch_size: int):
    df = pd.read_csv(Path(data_dir) / f"{ticker}.csv", parse_dates=["Date"])
    input_cols = ["Close", "SMA_5", "SMA_20", "RSI_14", "MACD", "MACD_Signal"]
    df = df.dropna().reset_index(drop=True)
    split = int(len(df)*0.8)
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    train_ds = WindowDataset(train_df, input_cols, lookback=lookback)
    val_ds = WindowDataset(val_df, input_cols, lookback=lookback)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(n_features=len(input_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{epochs}  train_loss={tr:.6f}  val_loss={va:.6f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    main(args.data, args.ticker, args.epochs, args.lookback, args.batch_size)
