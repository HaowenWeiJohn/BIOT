import torch
from torch.utils.data import DataLoader, Dataset
import time

# Step 1: Create a simple custom dataset
class SimpleDataset(Dataset):
    def __init__(self, size):
        # Create a synthetic dataset with integers from 0 to size-1
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Simulate some data processing delay
        time.sleep(0.01)
        return {"input": self.data[idx], "label": self.data[idx] % 2}

if __name__ == '__main__':

    # Step 2: Initialize the dataset
    dataset = SimpleDataset(size=1000)

    # Step 3: Create the DataLoader with more than 5 workers
    data_loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=6,  # More than 5 workers
        pin_memory=True  # Optimized for GPU if available

    )

    # Step 4: Loop through the DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(2):  # Train for 2 epochs
        print(f"Epoch {epoch + 1} starting...")
        for batch_idx, batch in enumerate(data_loader):
            # Move data to the appropriate device
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            # Example operation (no model here, just showcasing the loop)
            print(f"Batch {batch_idx + 1} processed with inputs: {inputs[:3]}")

    print("Data loading and looping completed!")
