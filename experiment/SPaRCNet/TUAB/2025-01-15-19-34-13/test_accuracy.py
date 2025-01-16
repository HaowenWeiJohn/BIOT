import torch.nn as nn
import os

import numpy as np
import torch
import torch.nn as nn

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
from model import SPaRCNet
from utils import TUABLoader
from sklearn.preprocessing import label_binarize
import time


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
        inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)

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

            progress_bar.set_description(f"Validation (Batch Loss: {loss.item():.4f})")

    targets_all = torch.cat(targets_all)
    predictions_all = torch.cat(predictions_all)

    auroc, sensitivity, specificity, f1, balanced_acc = compute_metrics(targets_all, predictions_all, n_classes)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, auroc, sensitivity, specificity, f1, balanced_acc



# def test_model(model, loader, device):
#     model.eval()
#     predictions = []
#     targets_list = []
#
#     with torch.no_grad():
#         for inputs, targets in tqdm(loader, desc="Testing", leave=False):
#             inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
#
#             # Forward pass
#             outputs = model(inputs)
#             predictions.append(outputs.cpu())
#             targets_list.append(targets)
#
#     predictions = torch.cat(predictions, dim=0)
#     targets_list = torch.cat(targets_list, dim=0)
#
#     # Compute metrics
#     avg_auc, avg_sensitivity, avg_specificity, test_f1, test_balanced_accuracy = compute_metrics(targets_list, predictions, n_classes)
#
#     return predictions, targets_list


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':

    # torch.multiprocessing.set_start_method('spawn', force=True)
    # check if GPU is available
    print(torch.cuda.is_available())

    sampling_rate = 200
    sample_length = 10
    batch_size = 512
    num_workers = 32
    in_channels = 16
    n_classes = 2
    num_epochs = 10
    lr = 1e-3
    weight_decay = 1e-5

    model_name = "SPaRCNet"
    dataset_name = "TUAB"
    dataset_root = "C:/Dataset/raw/tuh_eeg_abnormal/v3.0.1/edf/processed/"


    test_files = os.listdir(os.path.join(dataset_root, "test"))


    test_dataset = TUABLoader(os.path.join(dataset_root, "test"), test_files, sampling_rate)


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )



    model = SPaRCNet(
        in_channels = in_channels,
        sample_length=int(sampling_rate * sample_length),
        n_classes=n_classes,
        block_layers=4,
        growth_rate=16,
        bn_size=16,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    )

    # Cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Testing
    # model.load_state_dict(torch.load(os.path.join("best_model.pth")))
    test_loss, test_auroc, test_sensitivity, test_specificity, test_f1, test_balanced_accuracy = evaluate(
        model, test_loader, criterion, device, n_classes
    )



    # change to dict
    test_result = {
        'test_loss': test_loss,
        'test_auroc': test_auroc,
        'test_sensitivity': test_sensitivity,
        'test_specificity': test_specificity,
        'test_f1': test_f1,
        'test_balanced_accuracy': test_balanced_accuracy,
    }

    # print the test results
    print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auroc:.4f} | Test Sensitivity: {test_sensitivity:.4f} | Test Specificity: {test_specificity:.4f} | Test F1: {test_f1:.4f} | Test Balanced Accuracy: {test_balanced_accuracy:.4f}" )


    # save the test results
    # np.savez(os.path.join("test_results.npz"), **test_result)


    print("Testing completed and results saved.")







