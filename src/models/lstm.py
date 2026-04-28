import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from src.preprocessing.constants import IDX_TO_ACTIVITY, ACTIVITY_MAP
from src.models.utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
    save_metrics_report,
    save_training_history,
)
from src.models.utils.pytorch_dataloader import get_dataloaders


class CNNBranch(nn.Module):
    """Single CNN branch applied time-distributed across sub-sequences."""

    def __init__(self, n_channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(64, 32, kernel_size, padding=pad)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch * n_seq, n_channels, n_steps)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)  # (batch*n_seq, 32, n_steps//2)
        return x.flatten(1)  # (batch*n_seq, 32 * 16 = 512)


class MultibranchCNNBiLSTM(nn.Module):
    """
    Challa et al. (2021) architecture.
    Input shape: (batch, n_seq=4, n_steps=32, n_channels=37)
    """

    def __init__(
        self, n_channels: int, n_classes: int, n_seq: int = 4, n_steps: int = 32
    ):
        super().__init__()
        self.n_seq = n_seq
        self.n_steps = n_steps

        self.branch3 = CNNBranch(n_channels, kernel_size=3)
        self.branch7 = CNNBranch(n_channels, kernel_size=7)
        self.branch11 = CNNBranch(n_channels, kernel_size=11)

        feat_per_branch = 32 * (n_steps // 2)  # 32 * 16 = 512
        lstm_input_size = 3 * feat_per_branch  # 1536

        # BiLSTM: output size = 2 × hidden_size (bidirectional)
        self.bilstm1 = nn.LSTM(
            lstm_input_size, 64, batch_first=True, bidirectional=True
        )
        self.bilstm2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(0.5)
        self.dense = nn.Linear(64, 128)
        self.bn = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)

        # Flatten sub-sequence and batch dims for TimeDistributed Conv1D
        # (batch, n_seq, n_steps, n_channels) → (batch*n_seq, n_channels, n_steps)
        x = x.view(batch * self.n_seq, self.n_steps, -1)
        x = x.permute(0, 2, 1).contiguous()

        b3 = self.branch3(x)
        b7 = self.branch7(x)
        b11 = self.branch11(x)

        out = torch.cat([b3, b7, b11], dim=1)  # (batch*n_seq, 1536)
        out = out.view(batch, self.n_seq, -1)  # (batch, n_seq, 1536)

        out, _ = self.bilstm1(out)  # (batch, n_seq, 128)
        out, _ = self.bilstm2(out)  # (batch, n_seq, 64)
        out = out[:, -1, :]  # last timestep (batch, 64)
        out = self.dropout_lstm(out)

        out = F.relu(self.bn(self.dense(out)))  # (batch, 128)
        return self.classifier(out)  # (batch, n_classes)


class PyTorchLSTM:
    def __init__(
        self, n_channels, n_classes, lr=0.001, epochs=100, batch_size=400, weights=None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultibranchCNNBiLSTM(n_channels, n_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=7, factor=0.5, min_lr=1e-5
        )

        if weights is not None:
            weights = weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = n_classes

    def fit(self, train_loader, val_loader=None, patience=15) -> list[dict]:
        best_val_acc = 0.0
        epochs_no_imp = 0
        best_state = None
        stopped_epoch = self.epochs
        history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item()

            if val_loader is not None:
                val_acc = self._eval_accuracy(val_loader)
                self.scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_imp = 0
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    epochs_no_imp += 1
                    if epochs_no_imp >= patience:
                        stopped_epoch = epoch
                        print(f"  Early stopping at epoch {epoch}")
                        break

            if epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:3d} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}"
                )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running_loss / len(train_loader),
                    "val_acc": val_acc if val_loader is not None else None,
                }
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return history

    def _eval_accuracy(self, loader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                preds = self.model(batch_X).argmax(1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        return correct / total

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        # X might be (N, 128, 37) or (N, 4, 32, 37)
        # PAMAP2Dataset handles the reshape if needed, but here we might get raw numpy
        if len(X.shape) == 3 and X.shape[1] == 128:
            n, t, c = X.shape
            X = X.reshape(n, 4, 32, c)

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            all_probs = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i : i + self.batch_size]
                probs = torch.softmax(self.model(batch_X), dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def stream_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        self.model.eval()
        if len(X.shape) == 3 and X.shape[1] == 128:
            n, t, c = X.shape
            X = X.reshape(n, 4, 32, c)

        n = len(X)
        y_pred = np.empty(n, dtype=np.int64)
        y_prob = np.empty((n, self.num_classes), dtype=np.float32)
        latencies_ms = []

        with torch.no_grad():
            for i in range(n):
                window = torch.from_numpy(X[i : i + 1]).float().to(self.device)
                t0 = time.perf_counter()
                out = self.model(window)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                t1 = time.perf_counter()
                y_pred[i] = np.argmax(prob)
                y_prob[i] = prob[0]
                latencies_ms.append((t1 - t0) * 1000.0)

        total_ms = sum(latencies_ms)
        return (
            y_pred,
            y_prob,
            {"avg_ms": total_ms / n, "total_ms": total_ms, "n_samples": n},
        )


def run_lstm(data_dir: Path, output_dir: Path):
    """Fixed-split training for LSTM using windowed data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [ACTIVITY_MAP[IDX_TO_ACTIVITY[i]] for i in range(len(ACTIVITY_MAP))]

    print(f" Loading windowed data from {data_dir} ")
    # Set reshape_dl=True for the new architecture
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=400, is_windows=True, reshape_dl=True
    )

    first_X, _ = next(iter(train_loader))
    # Shape is (batch, 4, 32, 37), so n_channels is at index 3
    n_channels = first_X.shape[3]
    num_classes = len(ACTIVITY_MAP)

    y_train = train_loader.dataset.y.numpy()
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.FloatTensor(weights)

    clf = PyTorchLSTM(n_channels, num_classes, weights=weights_tensor)

    print(" Training PyTorch Multibranch CNN-BiLSTM ")
    history = clf.fit(train_loader, val_loader=val_loader, patience=15)
    save_training_history(history, output_dir)

    all_metrics = []
    for name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        X_eval = loader.dataset.X.numpy()
        y_eval = loader.dataset.y.numpy()
        y_pred = clf.predict(X_eval)
        y_prob = clf.predict_proba(X_eval)
        m = calculate_metrics(y_eval, y_pred, y_prob, name)
        all_metrics.append(m)
        plot_confusion_matrix(y_eval, y_pred, name, output_dir, class_names)
        plot_pr_curves(y_eval, y_prob, name, output_dir, class_names)
        print(f"  {name.upper()}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

    print(" Streaming test predictions (1 window at a time) ")
    X_test = test_loader.dataset.X.numpy()
    _, _, timing = clf.stream_predict(X_test)
    print(
        f"  Streaming: avg={timing['avg_ms']:.4f}ms/window, "
        f"total={timing['total_ms']:.2f}ms over {timing['n_samples']} windows"
    )
    save_metrics_report(
        all_metrics, output_dir, "CNN-BiLSTM (Challa et al.)", timing=timing
    )


def main():
    processed_base = Path("PAMAP2_Dataset/processed")
    results_base = Path("results/lstm")
    for subdir in ["normal", "feature_selection"]:
        run_lstm(processed_base / subdir, results_base / subdir)


if __name__ == "__main__":
    main()
