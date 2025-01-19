from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score
)

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch
import numpy as np
import torch.nn.functional as F
import os
from scipy.signal import resample


class TUABLoader(torch.utils.data.Dataset):
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
        # if self.sampling_rate != self.default_rate:
        #     X = resample(X, 10 * self.sampling_rate, axis=-1)
        # X = X / (
        #     np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
        #     + 1e-8
        # )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y

class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path, sampling_rate=200):
        # x is numpy array of shape (n_samples, n_channels, n_timesteps)
        # y is 1-D numpy array of shape (n_samples,)

        self.x = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = self.y[index]
        return x, y



class IIICLoader(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path, sampling_rate=200):
        # x is numpy array of shape (n_samples, n_channels, n_timesteps)
        # y is 1-D numpy array of shape (n_samples,)

        self.x = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = self.y[index]
        return x, y





def compute_metrics(targets, predictions, n_classes):
    targets = targets.cpu().numpy()
    predictions_prob = torch.softmax(predictions, dim=1).detach().cpu().numpy()
    predictions_classes = predictions_prob.argmax(axis=1)

    # Calculate AUC for each class
    auc_scores = []
    for i in range(n_classes):
        try:
            auc_scores.append(roc_auc_score((targets == i).astype(int), predictions_prob[:, i]))
        except ValueError:
            auc_scores.append(float('nan'))

    # Compute confusion matrix for sensitivity and specificity
    cm = confusion_matrix(targets, predictions_classes, labels=list(range(n_classes)))
    sensitivity = []
    specificity = []

    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        specificity.append(tn / (tn + fp) if tn + fp > 0 else 0.0)

    # Calculate F1 score
    f1 = f1_score(targets, predictions_classes, average='macro')

    # Calculate balanced accuracy
    balanced_accuracy = balanced_accuracy_score(targets, predictions_classes)

    # Average metrics
    avg_auc = np.nanmean(auc_scores)  # Handle cases where AUC is not computable
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)



    return avg_auc, avg_sensitivity, avg_specificity, f1, balanced_accuracy



def train_one_epoch(model, loader, criterion, optimizer, device, n_classes):
    model.train()
    running_loss = 0.0
    targets_all = []
    predictions_all = []

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * inputs.size(0)
        targets_all.append(targets)
        predictions_all.append(outputs)

        progress_bar.set_description(f"Training (Batch Loss: {loss.item():.4f})")

    targets_all = torch.cat(targets_all)
    predictions_all = torch.cat(predictions_all)

    auroc, sensitivity, specificity, f1, balanced_acc = compute_metrics(targets_all, predictions_all, n_classes)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, auroc, sensitivity, specificity, f1, balanced_acc

def evaluate(model, loader, criterion, device, n_classes):
    model.eval()
    running_loss = 0.0
    targets_all = []
    predictions_all = []

    progress_bar = tqdm(loader, desc="Evaluate", leave=False)

    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Metrics
            running_loss += loss.item() * inputs.size(0)
            targets_all.append(targets)
            predictions_all.append(outputs)

            progress_bar.set_description(f"Evaluate (Batch Loss: {loss.item():.4f})")

    targets_all = torch.cat(targets_all)
    predictions_all = torch.cat(predictions_all)

    auroc, sensitivity, specificity, f1, balanced_acc = compute_metrics(targets_all, predictions_all, n_classes)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, auroc, sensitivity, specificity, f1, balanced_acc


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


