"""
prepare_data.py
================
Combines the three datasets, splits into train/test, scales every
feature, and wraps everything into PyTorch DataLoaders ready for
training. See lessons/03_preparing_the_data.md for the full explanation.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES


def prepare_data(X_bb, y_bb, X_cc, y_cc, X_qcd, y_qcd, batch_size=256, test_size=0.2, random_state=42):
    """
    X_*, y_* : outputs of features.extract_features() for each dataset.
    Returns: train_loader, test_loader, X_test_scaled, y_test, scaler
             (the last two are handy for evaluate.py)
    """
    # Combine everything into one big pile
    X = np.concatenate([X_bb, X_cc, X_qcd], axis=0)
    y = np.concatenate([y_bb, y_cc, y_qcd], axis=0)

    # Hold out 20% of the data that the model never trains on,
    # keeping class proportions equal on both sides (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Put every feature on the same "average 0, spread 1" footing.
    # Flatten (n_events, 2, 10) -> (n_events*2, 10) to fit the scaler,
    # then reshape back.
    scaler = StandardScaler()
    n_features = len(FEATURE_NAMES)

    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)

    # Fit the scaler on TRAINING data only, then apply it to both sets --
    # never fit on test data, or you leak information about it.
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(-1, 2, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(-1, 2, n_features)

    # Wrap into PyTorch tensors and batch DataLoaders
    train_data = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_data = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test_scaled, y_test, scaler
