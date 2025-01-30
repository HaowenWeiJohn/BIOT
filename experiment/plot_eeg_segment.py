import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from datasets.IIIC.process import train_mask

# load IIIC dataset

dataset_root = "C:/Dataset/raw/IIIC/processed/"
# dataset_root, "train_X.npy"
train_x_path = os.path.join(dataset_root, "train_X.npy")
train_y_path = os.path.join(dataset_root, "train_Y.npy")

train_x = np.load(train_x_path)
train_y = np.load(train_y_path)

print(train_x.shape)
print(train_y.shape)

# plot EEG segment



