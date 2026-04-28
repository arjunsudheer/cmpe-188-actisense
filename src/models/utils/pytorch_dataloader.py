import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path


class PAMAP2Dataset(Dataset):
    def __init__(self, data_path: Path, labels_path: Path, reshape_dl: bool = False):
        X_np = np.load(data_path)
        if reshape_dl and len(X_np.shape) == 3 and X_np.shape[1] == 128:
            # (N, 128, C) -> (N, 4, 32, C)
            n, t, c = X_np.shape
            X_np = X_np.reshape(n, 4, 32, c)

        self.X = torch.from_numpy(X_np).float()
        self.y = torch.from_numpy(np.load(labels_path)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 400,
    is_windows: bool = False,
    is_ml: bool = False,
    reshape_dl: bool = False,
):
    if is_ml:
        suffix = "_ml"
    elif is_windows:
        suffix = "_windows"
    else:
        suffix = ""

    train_ds = PAMAP2Dataset(
        data_dir / f"train{suffix}_X.npy",
        data_dir / f"train_y.npy",
        reshape_dl=reshape_dl,
    )
    val_ds = PAMAP2Dataset(
        data_dir / f"val{suffix}_X.npy", data_dir / f"val_y.npy", reshape_dl=reshape_dl
    )
    test_ds = PAMAP2Dataset(
        data_dir / f"test{suffix}_X.npy",
        data_dir / f"test_y.npy",
        reshape_dl=reshape_dl,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
