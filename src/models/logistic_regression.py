import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from src.preprocessing.constants import IDX_TO_ACTIVITY
from src.models.utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
    save_metrics_report,
)
from src.models.utils.pytorch_dataloader import get_dataloaders


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class PyTorchLogReg:
    def __init__(self, input_dim, num_classes, lr=0.01, epochs=30, batch_size=1024, weights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LogisticRegressionModel(input_dim, num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Use provided weights for imbalanced datasets
        if weights is not None:
            weights = weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(train_loader):.4f}")

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def run_logreg(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [IDX_TO_ACTIVITY[i] for i in range(len(IDX_TO_ACTIVITY))]

    print(f" Loading data for LogReg from {data_dir} ...")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=2048)

    # Get dimensions
    first_X, _ = next(iter(train_loader))
    input_dim = first_X.shape[1]
    num_classes = len(IDX_TO_ACTIVITY)

    # Calculate balanced class weights to handle imbalance
    y_train = train_loader.dataset.y.numpy()
    counts = np.bincount(y_train, minlength=num_classes)
    weights = np.ones(num_classes)
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = len(y_train) / (num_classes * counts[i])
    weights_tensor = torch.tensor(weights, dtype=torch.float)

    clf = PyTorchLogReg(input_dim, num_classes, weights=weights_tensor)

    print(" Training PyTorch Logistic Regression (Weighted) ...")
    clf.fit(train_loader)

    all_metrics = []
    # Use the original Dataset objects to get full numpy arrays for scikit-learn metrics
    # This avoids rewriting metrics.py to be batch-based
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

    save_metrics_report(all_metrics, output_dir, "Logistic Regression (PyTorch)")


def main():
    processed_base = Path("PAMAP2_Dataset/processed")
    results_base = Path("results/logreg")
    for subdir in ["normal", "feature_selection"]:
        run_logreg(processed_base / subdir, results_base / subdir)


if __name__ == "__main__":
    main()
