import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import os


def create(args):
    """Create train and test dataloaders with 80/20 split from channel_io.npz."""
    path = os.path.join(args.dataset_dir, 'channel_io.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found: {path}')

    data = np.load(path)
    Z = data['Z']      # (N, 64) complex
    Z_hat = data['Z_hat']  # (N, 64) complex

    # Keep as complex; convert to torch (float32 for real/imag)
    z_t = torch.from_numpy(Z.astype(np.complex64))   # (N, 64)
    z_hat_t = torch.from_numpy(Z_hat.astype(np.complex64))

    dataset = TensorDataset(z_t, z_hat_t)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_test = n - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(getattr(args, 'seed', 0)))

    batch_size = getattr(args, 'batch_size', 64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, test_loader
