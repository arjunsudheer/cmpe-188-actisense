import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

class PAMAP2Dataset(Dataset):
    """
    Generic Dataset for PAMAP2 .npy data.
    Works for both 2D tabular data (N, F) and 3D windowed data (N, T, F).
    """
    def __init__(self, data_path: Path, labels_path: Path):
        self.X = torch.from_numpy(np.load(data_path)).float()
        self.y = torch.from_numpy(np.load(labels_path)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(data_dir: Path, batch_size: int = 1024, is_windows: bool = False):
    """
    Creates train, val, and test DataLoaders for a given directory.
    
    Args:
        data_dir: Path to the directory containing .npy files.
        batch_size: Number of samples per batch.
        is_windows: If True, loads windowed data (e.g. 'train_windows_X.npy').
                   If False, loads tabular data (e.g. 'train_X.npy').
    """
    suffix = "_windows" if is_windows else ""
    
    train_ds = PAMAP2Dataset(data_dir / f"train{suffix}_X.npy", data_dir / f"train{suffix}_y.npy")
    val_ds = PAMAP2Dataset(data_dir / f"val{suffix}_X.npy", data_dir / f"val{suffix}_y.npy")
    test_ds = PAMAP2Dataset(data_dir / f"test{suffix}_X.npy", data_dir / f"test{suffix}_y.npy")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
