"""
Entraînement du modèle LSTM de prédiction de séries temporelles.

Usage:
    python train.py --ticker MC.PA --epochs 50 --seq-len 60
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Chemins
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(ROOT, "data", "raw")
PRED_PATH = os.path.join(ROOT, "data", "predictions.csv")
MODEL_DIR = os.path.join(ROOT, "model")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from model.lstm_model import LSTMForecaster
from data.download_data import download_stock


# ─── Dataset ────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


# ─── Métriques ──────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ─── Entraînement ───────────────────────────────────────────────────────────

def train(ticker: str = "MC.PA", epochs: int = 50, seq_len: int = 60,
          hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2,
          lr: float = 1e-3, batch_size: int = 32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[config] Ticker={ticker}  Epochs={epochs}  SeqLen={seq_len}  Device={device}\n")

    # 1. Données
    csv_path = os.path.join(DATA_RAW, f"{ticker.replace('.', '_')}.csv")
    if not os.path.exists(csv_path):
        df = download_stock(ticker)
    else:
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)

    prices = df["close"].values.reshape(-1, 1).astype(np.float32)

    # 2. Normalisation
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    # 3. Séquences
    X, y = build_sequences(prices_scaled, seq_len)
    X = X  # shape (N, seq_len, 1)

    n = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TimeSeriesDataset(X_val,   y_val),   batch_size=batch_size)
    test_loader  = DataLoader(TimeSeriesDataset(X_test,  y_test),  batch_size=batch_size)

    # 4. Modèle
    model = LSTMForecaster(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 5. Boucle d'entraînement
    train_losses, val_losses = [], []

    for epoch in tqdm(range(1, epochs + 1), desc="Entraînement", unit="epoch"):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(xb)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            tqdm.write(f"  Époque {epoch:3d}/{epochs}  |  Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}")

    # 6. Évaluation sur test
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds_scaled.append(model(xb.to(device)).cpu().numpy())
            trues_scaled.append(yb.numpy())

    preds_scaled = np.concatenate(preds_scaled)
    trues_scaled = np.concatenate(trues_scaled)

    preds = scaler.inverse_transform(preds_scaled)
    trues = scaler.inverse_transform(trues_scaled)

    metrics = {
        "MAE":  mae(trues, preds),
        "RMSE": rmse(trues, preds),
        "MAPE": mape(trues, preds),
    }

    print(f"\n{'─'*40}")
    print(f"  Résultats sur le jeu de test")
    print(f"{'─'*40}")
    print(f"  MAE  : {metrics['MAE']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  MAPE : {metrics['MAPE']:.2f}%")
    print(f"{'─'*40}\n")

    # 7. Sauvegarde modèle
    model_path = os.path.join(MODEL_DIR, "saved_model.pth")
    torch.save({
        "model_state": model.state_dict(),
        "scaler": scaler,
        "hyperparams": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "seq_len": seq_len,
        },
        "metrics": metrics,
        "ticker": ticker,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, model_path)
    print(f"[model] Sauvegardé → {model_path}")

    # 8. Sauvegarde prédictions
    test_dates = df.index[n_train + n_val + seq_len : n_train + n_val + seq_len + len(preds)]
    pred_df = pd.DataFrame({
        "date": test_dates,
        "actual": trues.flatten(),
        "predicted": preds.flatten(),
    })
    pred_df.to_csv(PRED_PATH, index=False)
    print(f"[data] Prédictions sauvegardées → {PRED_PATH}")

    # 9. Graphiques
    _plot_losses(train_losses, val_losses)
    _plot_predictions(pred_df, ticker)

    return metrics


# ─── Visualisation ──────────────────────────────────────────────────────────

def _plot_losses(train_losses, val_losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Validation Loss")
    ax.set_xlabel("Époque")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Courbe d'apprentissage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Training loss → {path}")


def _plot_predictions(pred_df: pd.DataFrame, ticker: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pred_df["date"], pred_df["actual"],    label="Réel",      alpha=0.8)
    ax.plot(pred_df["date"], pred_df["predicted"], label="Prédit",    alpha=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix de clôture")
    ax.set_title(f"Prédiction LSTM — {ticker}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "predictions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Prédictions → {path}")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement LSTM — Prédiction boursière")
    parser.add_argument("--ticker",  default="MC.PA",  help="Symbole boursier (ex: MC.PA, TTE.PA)")
    parser.add_argument("--epochs",  type=int, default=50, help="Nombre d'époques")
    parser.add_argument("--seq-len", type=int, default=60, help="Longueur de la fenêtre glissante")
    parser.add_argument("--hidden",  type=int, default=64, help="Taille des couches cachées LSTM")
    parser.add_argument("--lr",      type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train(
        ticker=args.ticker,
        epochs=args.epochs,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        lr=args.lr,
    )
