import pickle
import torch
import numpy as np
import torch.nn.functional as F
import os
from scipy.signal import resample
from sympy.physics.units import percent
from torch.utils.data import DataLoader, Dataset




class TUABLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


if __name__ == '__main__':

    root = "C:/Dataset/raw/tuh_eeg_abnormal/v3.0.1/edf/processed/"
    sampling_rate = 200
    train_files = os.listdir(os.path.join(root, "train"))

    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, sampling_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=6,  # More than 5 workers
        pin_memory=True,  # Optimized for GPU if available
        persistent_workers= True
    )

    # loop the train loader

    for epoch in range(2):  # Train for 2 epochs
        print(f"Epoch {epoch + 1} starting...")
        for batch_idx, (X, Y) in enumerate(train_loader):
            # Move data to the appropriate device
            inputs = X
            labels = Y
            # Example operation (no model here, just showcasing the loop)
            print(f"Batch {batch_idx + 1} processed with inputs: {inputs[:3]}")
            print(batch_idx)