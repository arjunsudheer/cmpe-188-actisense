import time
import torch
import torch.nn as nn
import torch.optim as optim
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
from src.models.utils.pytorch_dataloader import PAMAP2Dataset
from torch.utils.data import DataLoader


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class PyTorchLogReg:
    def __init__(
        self,
        input_dim,
        num_classes,
        lr=0.01,
        epochs=100,
        batch_size=400,
        weights=None,
        weight_decay=1e-4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LogisticRegressionModel(input_dim, num_classes).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if weights is not None:
            weights = weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes

    def fit(self, train_loader, val_loader=None, patience=10) -> list[dict]:
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        stopped_epoch = self.epochs
        history = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if val_loader is not None:
                val_loss = self._eval_loss(val_loader)
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        stopped_epoch = epoch + 1
                        break

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": running_loss / len(train_loader),
                    "val_loss": val_loss if val_loader is not None else None,
                }
            )

            if (epoch + 1) % 20 == 0:
                avg_train_loss = running_loss / len(train_loader)
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def _eval_loss(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                total_loss += self.criterion(outputs, batch_y).item()
        return total_loss / len(loader)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def stream_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        self.model.eval()
        n = len(X)
        y_prob = self.predict_proba(X)
        y_pred = np.argmax(y_prob, axis=1)

        n_time = min(n, 1000)
        latencies_ms = []
        with torch.no_grad():
            for i in range(n_time):
                sample = torch.from_numpy(X[i : i + 1]).float().to(self.device)
                t0 = time.perf_counter()
                out = self.model(sample)
                _ = torch.softmax(out, dim=1).cpu().numpy()
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        avg_ms = sum(latencies_ms) / n_time
        total_ms = avg_ms * n
        return y_pred, y_prob, {"avg_ms": avg_ms, "total_ms": total_ms, "n_samples": n}


def run_logreg(data_dir: Path, output_dir: Path):
    """Fixed-split training for Logistic Regression using ML features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [ACTIVITY_MAP[IDX_TO_ACTIVITY[i]] for i in range(len(ACTIVITY_MAP))]

    print(f" Loading ML features from {data_dir} ")
    train_ds = PAMAP2Dataset(data_dir / "train_ml_X.npy", data_dir / "train_y.npy")
    val_ds = PAMAP2Dataset(data_dir / "val_ml_X.npy", data_dir / "val_y.npy")
    test_ds = PAMAP2Dataset(data_dir / "test_ml_X.npy", data_dir / "test_y.npy")

    train_loader = DataLoader(train_ds, batch_size=400, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=400, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=400, shuffle=False)

    input_dim = train_ds.X.shape[1]
    num_classes = len(ACTIVITY_MAP)

    y_train = train_ds.y.numpy()
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.FloatTensor(weights)

    clf = PyTorchLogReg(input_dim, num_classes, weights=weights_tensor)

    print(" Training PyTorch Logistic Regression ")
    history = clf.fit(train_loader, val_loader=val_loader, patience=10)
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
    X_test = test_ds.X.numpy()
    _, _, timing = clf.stream_predict(X_test)
    print(
        f"  Streaming: avg={timing['avg_ms']:.4f}ms/window, "
        f"total={timing['total_ms']:.2f}ms over {timing['n_samples']} windows"
    )
    save_metrics_report(
        all_metrics, output_dir, "Logistic Regression (PyTorch)", timing=timing
    )


def main():
    processed_base = Path("PAMAP2_Dataset/processed")
    results_base = Path("results/logreg")
    for subdir in ["normal", "feature_selection"]:
        run_logreg(processed_base / subdir, results_base / subdir)


if __name__ == "__main__":
    main()
