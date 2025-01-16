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

            progress_bar.set_description(f"Evaluate (Batch Loss: {loss.item():.4f})")

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
    num_epochs = 100
    lr = 1e-3
    weight_decay = 1e-5

    model_name = "SPaRCNet"
    dataset_name = "TUAB"
    dataset_root = "C:/Dataset/raw/tuh_eeg_abnormal/v3.0.1/edf/processed/"

    # log dir should be names as model, dataset, and the time of the experiment
    log_dir = os.path.join(model_name, dataset_name, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)


    # load the train, val, test files
    train_files = os.listdir(os.path.join(dataset_root, "train"))
    val_files = os.listdir(os.path.join(dataset_root, "val"))
    test_files = os.listdir(os.path.join(dataset_root, "test"))


    train_dataset = TUABLoader(os.path.join(dataset_root, "train"), train_files, sampling_rate)
    val_dataset = TUABLoader(os.path.join(dataset_root, "val"), val_files, sampling_rate)
    test_dataset = TUABLoader(os.path.join(dataset_root, "test"), test_files, sampling_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # the total number of samples:
    print('train samples:', len(train_dataset))
    print('val samples:', len(val_dataset))
    print('test samples:', len(test_dataset))
    print('total:', len(train_dataset) + len(val_dataset) + len(test_dataset))


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    set_seed(42)

    # Cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # # Scheduler (optional for learning rate decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Scheduler (optional for learning rate decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)



    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_auroc': [],
        'train_sensitivity': [],
        'train_specificity': [],
        'train_f1': [],
        'train_balanced_accuracy': [],

        'val_loss': [],
        'val_auroc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_f1': [],
        'val_balanced_accuracy': [],
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_auroc, train_sensitivity, train_specificity, train_f1, train_balanced_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device, n_classes
        )
        val_loss, val_auroc, val_sensitivity, val_specificity, val_f1, val_balanced_accuracy = evaluate(
            model, val_loader, criterion, device, n_classes
        )

        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_auroc'].append(train_auroc)
        history['train_sensitivity'].append(train_sensitivity)
        history['train_specificity'].append(train_specificity)
        history['train_f1'].append(train_f1)
        history['train_balanced_accuracy'].append(train_balanced_accuracy)


        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)
        history['val_sensitivity'].append(val_sensitivity)
        history['val_specificity'].append(val_specificity)
        history['val_f1'].append(val_f1)
        history['val_balanced_accuracy'].append(val_balanced_accuracy)


        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), "best_model.pth")
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print("Model saved!")

        # Scheduler step
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auroc:.4f} | Train Sensitivity: {train_sensitivity:.4f} | Train Specificity: {train_specificity:.4f} | Train F1: {train_f1:.4f} | Train Balanced Accuracy: {train_balanced_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auroc:.4f} | Val Sensitivity: {val_sensitivity:.4f} | Val Specificity: {val_specificity:.4f} | Val F1: {val_f1:.4f} | Val Balanced Accuracy: {val_balanced_accuracy:.4f}")

    # Save training history
    np.savez("training_history.npz", **history)

    del train_loader
    del val_loader

    # Testing
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pth")))
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
    np.savez(os.path.join(log_dir, "test_results.npz"), **test_result)


    print("Testing completed and results saved.")


