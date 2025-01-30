# We load the pretrained model and the unsupervised pretrain model
from sympy.codegen import Print
import torch
from experiment.train_utils import *
from model import UnsupervisedPretrain, BIOTClassifier
from tqdm import tqdm
import pickle
import torch
import numpy as np
import time

def contrastive_loss(emb, pred_emb, T):
    emb = F.normalize(emb, p=2, dim=1)
    pred_emb = F.normalize(pred_emb, p=2, dim=1)
    sim = torch.mm(emb, pred_emb.t())
    sim /= T
    sim = torch.exp(sim)
    sim /= torch.sum(sim, dim=1, keepdim=True)
    loss = -torch.log(torch.diag(sim))
    return loss.mean()


def contrastive_loss(emb, pred_emb, temperature=0.1):
    """
    Computes the contrastive loss (InfoNCE loss) for a batch of embeddings.

    Args:
        emb (torch.Tensor): Batch of embeddings from one view, shape (batch_size, vector_size).
        pred_emb (torch.Tensor): Batch of embeddings from another view, shape (batch_size, vector_size).
        temperature (float): Temperature scaling parameter.

    Returns:
        torch.Tensor: Contrastive loss (scalar).
    """
    # Normalize the embeddings to unit vectors
    emb = F.normalize(emb, p=2, dim=1)  # Shape: (batch_size, vector_size)
    pred_emb = F.normalize(pred_emb, p=2, dim=1)  # Shape: (batch_size, vector_size)

    # Compute similarity matrix (batch_size x batch_size)
    logits = torch.mm(emb, pred_emb.t()) / temperature  # Scaled cosine similarity

    # Labels: Diagonal elements should be the positives (index match)
    batch_size = emb.shape[0]
    labels = torch.arange(batch_size).to(logits.device)  # Shape: (batch_size,)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


if __name__ == '__main__':

    print("Start training")
    print("BIOT-pretrain-SHHS+PREST")
    pretrained_model_path = "../pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"


    sampling_rate = 200
    sample_length = 10
    batch_size = 512
    num_workers = 2
    # in_channels = 16
    # n_classes = 6
    num_epochs = 20
    lr = 1e-3
    weight_decay = 1e-5
    T = 0.2
    scheduler_step_size = 100
    scheduler_gamma = 0.95

    save_step = 20

    # load this to the unsupervised pretrain model
    model = UnsupervisedPretrain(
        emb_size=256, heads=8, depth=4, n_channels=18, n_fft=200, hop_length=100
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # load the pretrained model
    model.biot.load_state_dict(torch.load(pretrained_model_path))

    Print("Model loaded")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )

    iiic_dataset_root = "C:/Dataset/raw/IIIC/processed/"
    # load the train, val, test files
    iiic_train_dataset = IIICLoader(os.path.join(iiic_dataset_root, "train_X.npy"), os.path.join(iiic_dataset_root, "train_Y.npy"),
                               sampling_rate)
    iiic_val_dataset = IIICLoader(os.path.join(iiic_dataset_root, "val_X.npy"), os.path.join(iiic_dataset_root, "val_Y.npy"),
                             sampling_rate)
    iiic_test_dataset = IIICLoader(os.path.join(iiic_dataset_root, "test_X.npy"), os.path.join(iiic_dataset_root, "test_Y.npy"),
                              sampling_rate)

    tuab_dataset_root = "C:/Dataset/raw/tuh_eeg_abnormal/v3.0.1/edf/processed/"

    tuab_train_files = os.listdir(os.path.join(tuab_dataset_root, "train"))
    tuab_val_files = os.listdir(os.path.join(tuab_dataset_root, "val"))
    tuab_test_files = os.listdir(os.path.join(tuab_dataset_root, "test"))

    tuab_train_dataset = TUABLoader(os.path.join(tuab_dataset_root, "train"), tuab_train_files, sampling_rate)
    tuab_val_dataset = TUABLoader(os.path.join(tuab_dataset_root, "val"), tuab_val_files, sampling_rate)
    tuab_test_dataset = TUABLoader(os.path.join(tuab_dataset_root, "test"), tuab_test_files, sampling_rate)


    # concatenate all the datasets
    dataset = torch.utils.data.ConcatDataset([iiic_train_dataset, iiic_val_dataset, iiic_test_dataset, tuab_train_dataset, tuab_val_dataset, tuab_test_dataset])

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # # create some dummpy data
    # data = torch.randn(4000, 16, 2000)
    #
    # class DummyDataset(torch.utils.data.Dataset):
    #     def __init__(self, data):
    #         self.data = data
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    #     def __getitem__(self, idx):
    #         return torch.FloatTensor(self.data[idx])
    #
    #
    # dataset = DummyDataset(data)
    #
    # loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )





    log_dir = os.path.join("BIOT-pretrain-IIIC+TUAB", time.strftime("%Y-%m-%d-%H-%M-%S_pretrain"))
    os.makedirs(log_dir, exist_ok=True)
    # create a txt log file to save the results
    log_file = open(os.path.join(log_dir, "log.txt"), "w")

    log_file.write(f"Time: {time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
    log_file.write(f"Learning Rate: {lr}\n")
    log_file.write(f"Weight Decay: {weight_decay}\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.write(f"Temperature: {T}\n")
    log_file.write(f"Scheduler Step Size: {scheduler_step_size}\n")
    log_file.write(f"Scheduler Gamma: {scheduler_gamma}\n")
    log_file.flush()

    set_seed(42)

    global_step = 0


    for epoch in range(num_epochs):
        with tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}") as pbar:
            for i, (data, labels) in pbar:
                optimizer.zero_grad()
                data = data.to(device)
                emb, pred_emb = model(data, n_channel_offset=0)
                loss = contrastive_loss(emb, pred_emb, T)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Update tqdm progress bar description with current loss
                pbar.set_postfix({'loss': loss.item()})

                log_file.write(f"Epoch {epoch} step {i} loss {loss.item()}\n")
                log_file.flush()

                global_step += 1

                if global_step % save_step == 0:
                    # Save the model with the global step name
                    torch.save(model.biot.state_dict(),
                               os.path.join(log_dir, f"EEG-IIIC+TUAB-18-channels_Step-{global_step}.ckpt"))
                    print(f"Model saved at step {global_step}")





                # for epoch in range(num_epochs):
                #     for i, (data, labels) in enumerate(loader):
                #         optimizer.zero_grad()
                #         data = data.to(device)
                #         emb, pred_emb = model(data, n_channel_offset=0)
                #         loss = contrastive_loss(emb, pred_emb, T)
                #
                #         loss.backward()
                #         optimizer.step()
                #         scheduler.step()
                #         print(f"Epoch {epoch} step {i} loss {loss.item()}")
                #         log_file.write(f"Epoch {epoch} step {i} loss {loss.item()}\n")
                #         log_file.flush()
                #
                #         global_step += 1
                #
                #         if global_step % save_step == 0:
                #             # save the model with the global step name +
                #             torch.save(model.biot.state_dict(), os.path.join(log_dir, "EEG-IIIC+TUAB-18-channels_" + f"Step-{global_step}.ckpt"))
                #             print(f"Model saved at step {global_step}")
                #

                # save the encoder
                # create a new directory with the global step
                # os.makedirs(os.path.join(log_dir, f"{global_step}"), exist_ok=True)
                # # save the model
                # torch.save(model.biot.state_dict(), os.path.join(log_dir, f"{global_step}", "model.ckpt"))
                # print(f"Model saved at step {global_step}")
                # # create a new log file with the global step
                # evaluate_log = open(os.path.join(log_dir, f"{global_step}", "evaluate_log.txt"), "w")
                # eval_model = BIOTClassifier(
                #     emb_size=256,
                #     heads=8,
                #     depth=4,
                #     n_classes=n_classes,
                #     n_fft=200,
                #     hop_length=100
                # )
                #
                # eval_model.biot.load_state_dict(model.biot.state_dict())
                # eval_model = eval_model.to(device)
                #









