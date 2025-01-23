# We load the pretrained model and the unsupervised pretrain model
from sympy.codegen import Print
import torch
from experiment.train_utils import *
from model import UnsupervisedPretrain
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
    num_workers = 0
    in_channels = 16
    n_classes = 6
    num_epochs = 20
    lr = 1e-3
    weight_decay = 1e-5
    T = 0.2

    # load this to the unsupervised pretrain model
    model = UnsupervisedPretrain(
        emb_size=256, heads=8, depth=4, n_channels=18, n_fft=200, hop_length=100
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load the pretrained model
    model.biot.load_state_dict(torch.load(pretrained_model_path))

    Print("Model loaded")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.95
    )


    # create some dummpy data
    data = torch.randn(520*100, 16, 2000)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.FloatTensor(self.data[idx])


    dataset = DummyDataset(data)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    log_dir = os.path.join("BIOT-pretrain-TUEV-IIIC", time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    # create a txt log file to save the results
    log_file = open(os.path.join(log_dir, "log.txt"), "w")

    log_file.write(f"Time: {time.strftime('%Y-%m-%d-%H-%M-%S')}\n")
    log_file.write(f"Learning Rate: {lr}\n")
    log_file.write(f"Weight Decay: {weight_decay}\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.flush()

    set_seed(42)


    for epoch in range(num_epochs):
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            emb, pred_emb = model(batch, n_channel_offset=0)
            loss = contrastive_loss(emb, pred_emb, T)

            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch} step {i} loss {loss.item()}")


