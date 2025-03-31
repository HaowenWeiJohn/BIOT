import argparse
import time
import torch.nn as nn
from experiment.train_utils import *
from model import SPaRCNet, CNNTransformer, FFCL, ContraWR, STTransformer, BIOTClassifier, BIOTEncoder
from model.biot import ClassificationHead
from itertools import zip_longest
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class SupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth)
        # Removed chb-mit classifier, keep the others:
        self.classifier_iiic_seizure = ClassificationHead(emb_size, 6)
        self.classifier_tuab = ClassificationHead(emb_size, 2)
        self.classifier_tuev = ClassificationHead(emb_size, 6)

    def forward(self, x, task="iiic-seizure"):
        x = self.biot(x)
        if task == "iiic-seizure":
            x = self.classifier_iiic_seizure(x)
        elif task == "tuab":
            x = self.classifier_tuab(x)
        elif task == "tuev":
            x = self.classifier_tuev(x)
        else:
            raise NotImplementedError(f"Unknown task: {task}")
        return x


def train_multitask_epoch(
    model,
    iiic_loader,
    tuab_loader,
    tuev_loader,
    optimizer,
    criterion,
    device,
    # each dataset might have a different number of classes:
    n_classes_iiic=6,
    n_classes_tuab=2,
    n_classes_tuev=6
):
    """
    Perform one training epoch in a round-robin way across the three DataLoaders.
    Returns a dict of metrics for each task, e.g.:
        {
           "iiic":  (loss, auroc, sensitivity, specificity, f1, balanced_accuracy),
           "tuab":  (...),
           "tuev":  (...),
        }
    or whatever you decide to return.
    """

    model.train()

    # We’ll keep track of the aggregated loss and predictions per task
    # so we can compute your usual metrics at the end of the epoch.
    iiic_running_loss = 0.0
    iiic_targets = []
    iiic_outputs = []
    iiic_num_samples = 0

    tuab_running_loss = 0.0
    tuab_targets = []
    tuab_outputs = []
    tuab_num_samples = 0

    tuev_running_loss = 0.0
    tuev_targets = []
    tuev_outputs = []
    tuev_num_samples = 0

    min_length = min(len(iiic_loader), len(tuab_loader), len(tuev_loader))

    # Round-robin through the loaders
    # for batch1, batch2, batch3 in zip(iiic_loader, tuab_loader, tuev_loader): #, fillvalue=None):
    for (batch1, batch2, batch3) in tqdm(zip(iiic_loader, tuab_loader, tuev_loader),
                                         total=min_length,
                                         desc="RoundRobin (IIIC, TUAB, TUEV)"):

        # == IIIC batch ==
        if batch1 is not None:
            x1, y1 = batch1
            x1, y1 = x1.to(device), y1.to(device)

            optimizer.zero_grad()
            out1 = model(x1, task="iiic-seizure")   # forward for iiic
            loss1 = criterion(out1, y1)
            loss1.backward()
            optimizer.step()

            # track metrics
            iiic_running_loss += loss1.item() * x1.size(0)
            iiic_targets.append(y1)
            iiic_outputs.append(out1)
            iiic_num_samples += x1.size(0)

        # == TUAB batch ==
        if batch2 is not None:
            x2, y2 = batch2
            x2, y2 = x2.to(device), y2.to(device)

            optimizer.zero_grad()
            out2 = model(x2, task="tuab")          # forward for tuab
            loss2 = criterion(out2, y2)
            loss2.backward()
            optimizer.step()

            # track metrics
            tuab_running_loss += loss2.item() * x2.size(0)
            tuab_targets.append(y2)
            tuab_outputs.append(out2)
            tuab_num_samples += x2.size(0)

        # == TUEV batch ==
        if batch3 is not None:
            x3, y3 = batch3
            x3, y3 = x3.to(device), y3.to(device)

            optimizer.zero_grad()
            out3 = model(x3, task="tuev")          # forward for tuev
            loss3 = criterion(out3, y3)
            loss3.backward()
            optimizer.step()

            # track metrics
            tuev_running_loss += loss3.item() * x3.size(0)
            tuev_targets.append(y3)
            tuev_outputs.append(out3)
            tuev_num_samples += x3.size(0)

    # Now compute final “epoch-level” metrics for each task
    results = {}

    # IIIC metrics
    if iiic_num_samples > 0:
        all_targets = torch.cat(iiic_targets, dim=0)
        all_outputs = torch.cat(iiic_outputs, dim=0)
        epoch_loss = iiic_running_loss / iiic_num_samples
        (auroc_val,
         sensitivity_val,
         specificity_val,
         f1_val,
         balanced_acc_val) = compute_metrics(all_targets, all_outputs, n_classes_iiic)

        results["iiic"] = (epoch_loss, auroc_val, sensitivity_val, specificity_val, f1_val, balanced_acc_val)

    # TUAB metrics
    if tuab_num_samples > 0:
        all_targets = torch.cat(tuab_targets, dim=0)
        all_outputs = torch.cat(tuab_outputs, dim=0)
        epoch_loss = tuab_running_loss / tuab_num_samples
        (auroc_val,
         sensitivity_val,
         specificity_val,
         f1_val,
         balanced_acc_val) = compute_metrics(all_targets, all_outputs, n_classes_tuab)

        results["tuab"] = (epoch_loss, auroc_val, sensitivity_val, specificity_val, f1_val, balanced_acc_val)

    # TUEV metrics
    if tuev_num_samples > 0:
        all_targets = torch.cat(tuev_targets, dim=0)
        all_outputs = torch.cat(tuev_outputs, dim=0)
        epoch_loss = tuev_running_loss / tuev_num_samples
        (auroc_val,
         sensitivity_val,
         specificity_val,
         f1_val,
         balanced_acc_val) = compute_metrics(all_targets, all_outputs, n_classes_tuev)

        results["tuev"] = (epoch_loss, auroc_val, sensitivity_val, specificity_val, f1_val, balanced_acc_val)

    return results


def evaluate_task(model, loader, device, criterion, n_classes, task_name):
    """
    Evaluate the model on a single dataset/loader with the specified task_name.
    Returns (loss, auroc, sensitivity, specificity, f1, balanced_accuracy).
    """
    model.eval()
    running_loss = 0.0
    targets_all = []
    outputs_all = []
    n_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"Evaluating {task_name}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, task=task_name)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)
            targets_all.append(targets)
            outputs_all.append(logits)

    if n_samples == 0:
        return (0, 0, 0, 0, 0, 0)

    # Concatenate
    all_targets = torch.cat(targets_all, dim=0)
    all_outputs = torch.cat(outputs_all, dim=0)

    epoch_loss = running_loss / n_samples
    auroc_val, sens_val, spec_val, f1_val, balanced_acc_val = compute_metrics(all_targets, all_outputs, n_classes)
    return (epoch_loss, auroc_val, sens_val, spec_val, f1_val, balanced_acc_val)


if __name__ == '__main__':

    # Access the arguments
    model_name = "BIOT-supervised_multitask_TUAB_IIIC_TUEV"
    dataset_name = "TUAB+IIIC+TUEV"

    print(torch.cuda.is_available())

    sampling_rate = 200
    # sample_length = 10
    batch_size = 512
    # num_workers = 4
    in_channels = 16
    # n_classes = 6
    num_epochs = 30
    lr = 1e-3
    weight_decay = 1e-5
    class_weight = None
    dataset_root = "C:/Dataset/raw/IIIC/processed/"

    # TUAB dataset parameters
    tuab_root = "C:/Dataset/raw/tuh_eeg_abnormal/v3.0.1/edf/processed/"
    tuab_class_weight = None
    tuab_n_classes = 2
    tuab_num_workers = 16
    tuab_sample_length = 10

    # IIIC dataset parameters
    iiic_root = "C:/Dataset/raw/IIIC/processed/"
    iiic_class_weight = [0.1181606, 0.10036655, 0.18678813, 0.20368562, 0.19775413, 0.19324496]
    iiic_n_classes = 6
    iiic_num_workers = 0
    iiic_sample_length = 10

    # TUEV dataset parameters
    tuev_root = "C:/Dataset/raw/tuh_eeg_events/v2.0.1/edf/processed/"
    tuev_class_weight = [0.5711400335475981, 0.02574656125416239, 0.06524567482806858, 0.3027807097490057,
                         0.02898413631785354, 0.006102884303311568]
    tuev_n_classes = 6
    tuev_num_workers = 0
    tuev_sample_length = 5

    # Instantiate your model
    model = SupervisedPretrain(emb_size=256, heads=8, depth=4)

    # Create TUAB dataset
    tuab_train_files = os.listdir(os.path.join(tuab_root, "train"))
    tuab_val_files = os.listdir(os.path.join(tuab_root, "val"))
    tuab_test_files = os.listdir(os.path.join(tuab_root, "test"))

    tuab_train_dataset = TUABLoader(os.path.join(tuab_root, "train"), tuab_train_files, sampling_rate)
    tuab_val_dataset = TUABLoader(os.path.join(tuab_root, "val"), tuab_val_files, sampling_rate)
    tuab_test_dataset = TUABLoader(os.path.join(tuab_root, "test"), tuab_test_files, sampling_rate)

    # Create IIIC dataset
    iiic_train_dataset = IIICLoader(os.path.join(iiic_root, "train_X.npy"), os.path.join(iiic_root, "train_Y.npy"),
                                    sampling_rate)
    iiic_val_dataset = IIICLoader(os.path.join(iiic_root, "val_X.npy"), os.path.join(iiic_root, "val_Y.npy"),
                                  sampling_rate)
    iiic_test_dataset = IIICLoader(os.path.join(iiic_root, "test_X.npy"), os.path.join(iiic_root, "test_Y.npy"),
                                   sampling_rate)

    # Create TUEV dataset
    tuev_train_dataset = TUEVLoader(os.path.join(tuev_root, "train_X.npy"), os.path.join(tuev_root, "train_Y.npy"),
                                    sampling_rate)
    tuev_val_dataset = TUEVLoader(os.path.join(tuev_root, "val_X.npy"), os.path.join(tuev_root, "val_Y.npy"),
                                  sampling_rate)
    tuev_test_dataset = TUEVLoader(os.path.join(tuev_root, "test_X.npy"), os.path.join(tuev_root, "test_Y.npy"),
                                   sampling_rate)

    # TUAB DataLoaders
    tuab_train_loader = DataLoader(
        tuab_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=tuab_num_workers,
        pin_memory=True,
        persistent_workers=tuab_num_workers > 0,
    )
    tuab_val_loader = DataLoader(
        tuab_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=tuab_num_workers,
        pin_memory=True,
        persistent_workers=tuab_num_workers > 0,
    )
    tuab_test_loader = DataLoader(
        tuab_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=tuab_num_workers,
        pin_memory=True,
        persistent_workers=tuab_num_workers > 0,
    )

    # IIIC DataLoaders
    iiic_train_loader = DataLoader(
        iiic_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=iiic_num_workers,
        pin_memory=True,
        persistent_workers=iiic_num_workers > 0,
    )
    iiic_val_loader = DataLoader(
        iiic_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=iiic_num_workers,
        pin_memory=True,
        persistent_workers=iiic_num_workers > 0,
    )
    iiic_test_loader = DataLoader(
        iiic_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=iiic_num_workers,
        pin_memory=True,
        persistent_workers=iiic_num_workers > 0,
    )

    # TUEV DataLoaders
    tuev_train_loader = DataLoader(
        tuev_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=tuev_num_workers,
        pin_memory=True,
        persistent_workers=tuev_num_workers > 0,
    )
    tuev_val_loader = DataLoader(
        tuev_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=tuev_num_workers,
        pin_memory=True,
        persistent_workers=tuev_num_workers > 0,
    )
    tuev_test_loader = DataLoader(
        tuev_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=tuev_num_workers,
        pin_memory=True,
        persistent_workers=tuev_num_workers > 0,
    )

    # log dir should be names as model, dataset, and the time of the experiment
    log_dir = os.path.join(model_name, dataset_name, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    # create a txt log file to save the results
    log_file = open(os.path.join(log_dir, "log.txt"), "w")

    # write the model, dataset name, time of the experiment, and all the hyperparameters
    log_file.write(f"Model: {model_name}\n")
    log_file.write(f"Dataset: {dataset_name}\n")
    log_file.write(f"Time: {time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
    log_file.write(f"Sampling Rate: {sampling_rate}\n")
    # log_file.write(f"Sample Length: {sample_length}\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    # log_file.write(f"Number of Workers: {num_workers}\n")
    # log_file.write(f"Number of Classes: {n_classes}\n")
    log_file.write(f"Number of Epochs: {num_epochs}\n")
    log_file.write(f"Learning Rate: {lr}\n")
    log_file.write(f"Weight Decay: {weight_decay}\n")
    log_file.write(f"Class Weight: {class_weight}\n")
    log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    set_seed(42)

    criterion = nn.CrossEntropyLoss()  # if you want a single global loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    best_val_loss = float('inf')  # or some criterion if you want to track the best epoch

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        log_file.write(f"Epoch {epoch + 1}/{num_epochs}\n")

        # --- Train (round-robin) ---
        train_metrics = train_multitask_epoch(
            model,
            iiic_train_loader,
            tuab_train_loader,
            tuev_train_loader,
            optimizer,
            criterion,
            device,
            n_classes_iiic=6,
            n_classes_tuab=2,
            n_classes_tuev=6
        )

        # train_metrics is a dict like:
        #   {
        #     "iiic": (loss, auroc, sens, spec, f1, balanced_acc),
        #     "tuab": (...),
        #     "tuev": (...),
        #   }

        # Log or print train metrics for each dataset
        if "iiic" in train_metrics:
            tr_loss, tr_auroc, tr_sens, tr_spec, tr_f1, tr_bal_acc = train_metrics["iiic"]
            msg = (f"   [IIIC]   Train Loss: {tr_loss:.4f} | AUC: {tr_auroc:.4f} | "
                   f"Sens: {tr_sens:.4f} | Spec: {tr_spec:.4f} | F1: {tr_f1:.4f} | Bal Acc: {tr_bal_acc:.4f}")
            print(msg)
            log_file.write(msg + "\n")

        if "tuab" in train_metrics:
            tr_loss, tr_auroc, tr_sens, tr_spec, tr_f1, tr_bal_acc = train_metrics["tuab"]
            msg = (f"   [TUAB]   Train Loss: {tr_loss:.4f} | AUC: {tr_auroc:.4f} | "
                   f"Sens: {tr_sens:.4f} | Spec: {tr_spec:.4f} | F1: {tr_f1:.4f} | Bal Acc: {tr_bal_acc:.4f}")
            print(msg)
            log_file.write(msg + "\n")

        if "tuev" in train_metrics:
            tr_loss, tr_auroc, tr_sens, tr_spec, tr_f1, tr_bal_acc = train_metrics["tuev"]
            msg = (f"   [TUEV]   Train Loss: {tr_loss:.4f} | AUC: {tr_auroc:.4f} | "
                   f"Sens: {tr_sens:.4f} | Spec: {tr_spec:.4f} | F1: {tr_f1:.4f} | Bal Acc: {tr_bal_acc:.4f}")
            print(msg)
            log_file.write(msg + "\n")

        # --- Validation ---
        # Evaluate each dataset separately
        iiic_val_loss, iiic_val_auroc, iiic_val_sens, iiic_val_spec, iiic_val_f1, iiic_val_bacc = evaluate_task(
            model, iiic_val_loader, device, criterion, 6, "iiic-seizure"
        )
        tuab_val_loss, tuab_val_auroc, tuab_val_sens, tuab_val_spec, tuab_val_f1, tuab_val_bacc = evaluate_task(
            model, tuab_val_loader, device, criterion, 2, "tuab"
        )
        tuev_val_loss, tuev_val_auroc, tuev_val_sens, tuev_val_spec, tuev_val_f1, tuev_val_bacc = evaluate_task(
            model, tuev_val_loader, device, criterion, 6, "tuev"
        )

        # Print & log validation metrics
        iiic_msg = (f"   [IIIC]   Val Loss: {iiic_val_loss:.4f} | AUC: {iiic_val_auroc:.4f} | "
                    f"Sens: {iiic_val_sens:.4f} | Spec: {iiic_val_spec:.4f} | F1: {iiic_val_f1:.4f} | Bal Acc: {iiic_val_bacc:.4f}")
        print(iiic_msg)
        log_file.write(iiic_msg + "\n")

        tuab_msg = (f"   [TUAB]   Val Loss: {tuab_val_loss:.4f} | AUC: {tuab_val_auroc:.4f} | "
                    f"Sens: {tuab_val_sens:.4f} | Spec: {tuab_val_spec:.4f} | F1: {tuab_val_f1:.4f} | Bal Acc: {tuab_val_bacc:.4f}")
        print(tuab_msg)
        log_file.write(tuab_msg + "\n")

        tuev_msg = (f"   [TUEV]   Val Loss: {tuev_val_loss:.4f} | AUC: {tuev_val_auroc:.4f} | "
                    f"Sens: {tuev_val_sens:.4f} | Spec: {tuev_val_spec:.4f} | F1: {tuev_val_f1:.4f} | Bal Acc: {tuev_val_bacc:.4f}")
        print(tuev_msg)
        log_file.write(tuev_msg + "\n")

        # Decide how you want to measure “best” – e.g., sum of val losses or average
        current_total_val_loss = iiic_val_loss + tuab_val_loss + tuev_val_loss
        if current_total_val_loss < best_val_loss:
            best_val_loss = current_total_val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            info_msg = "   [Info] New best model saved!"
            print(info_msg)
            log_file.write(info_msg + "\n")

        scheduler.step()  # typical usage
        log_file.flush()

    # --- Testing ---
    # Load best model and evaluate on test sets
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pth")))

    iiic_test_loss, iiic_test_auroc, iiic_test_sens, iiic_test_spec, iiic_test_f1, iiic_test_bacc = evaluate_task(
        model, iiic_test_loader, device, criterion, 6, "iiic-seizure"
    )
    tuab_test_loss, tuab_test_auroc, tuab_test_sens, tuab_test_spec, tuab_test_f1, tuab_test_bacc = evaluate_task(
        model, tuab_test_loader, device, criterion, 2, "tuab"
    )
    tuev_test_loss, tuev_test_auroc, tuev_test_sens, tuev_test_spec, tuev_test_f1, tuev_test_bacc = evaluate_task(
        model, tuev_test_loader, device, criterion, 6, "tuev"
    )

    final_msg = "\n** FINAL TEST RESULTS **"
    print(final_msg)
    log_file.write(final_msg + "\n")

    msg_iiic = (f"[IIIC ] Loss={iiic_test_loss:.4f}, AUC={iiic_test_auroc:.4f}, "
                f"Sens={iiic_test_sens:.4f}, Spec={iiic_test_spec:.4f}, F1={iiic_test_f1:.4f}, BAcc={iiic_test_bacc:.4f}")
    print(msg_iiic)
    log_file.write(msg_iiic + "\n")

    msg_tuab = (f"[TUAB ] Loss={tuab_test_loss:.4f}, AUC={tuab_test_auroc:.4f}, "
                f"Sens={tuab_test_sens:.4f}, Spec={tuab_test_spec:.4f}, F1={tuab_test_f1:.4f}, BAcc={tuab_test_bacc:.4f}")
    print(msg_tuab)
    log_file.write(msg_tuab + "\n")

    msg_tuev = (f"[TUEV ] Loss={tuev_test_loss:.4f}, AUC={tuev_test_auroc:.4f}, "
                f"Sens={tuev_test_sens:.4f}, Spec={tuev_test_spec:.4f}, F1={tuev_test_f1:.4f}, BAcc={tuev_test_bacc:.4f}")
    print(msg_tuev)
    log_file.write(msg_tuev + "\n")

    log_file.flush()
    log_file.close()

    print("Done! Logs and best_model.pth are saved.")
