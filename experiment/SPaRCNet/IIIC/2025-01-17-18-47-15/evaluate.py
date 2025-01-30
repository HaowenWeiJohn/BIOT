



import time

import torch.nn as nn

from experiment.train_utils import *
from model import SPaRCNet, CNNTransformer, FFCL, ContraWR, STTransformer, BIOTClassifier


if __name__ == '__main__':

    model_name = "SPaRCNet"
    # model_name = "CNNTransformer"
    # model_name = "FFCL"
    # model_name = "ContraWR"
    # model_name = "STTransformer"
    # model_name = "BIOT"
    # model_name = "BIOT-pretrain-PREST"
    # model_name = "BIOT-pretrain-SHHS+PREST"
    # model_name = "BIOT-pretrain-six-datasets"
    # model_name = "BIOT-pretrain-IIIC+TUAB"




    dataset_name = "IIIC"

    print(torch.cuda.is_available())

    sampling_rate = 200
    sample_length = 10
    batch_size = 512
    num_workers = 4
    in_channels = 16
    n_classes = 6
    num_epochs = 20
    lr = 1e-4
    weight_decay = 1e-5
    class_weight = None
    dataset_root = "C:/Dataset/raw/IIIC/processed/"

    # get dataset
    if dataset_name == "TUAB":
        dataset_root = "C:/Dataset/raw/tuh_eeg_abnormal/v3.0.1/edf/processed/"
        class_weight = None
        n_classes = 2

        num_workers = 32
        in_channels = 16
        sampling_rate = 200
        sample_length = 10

    elif dataset_name == "IIIC":
        dataset_root = "C:/Dataset/raw/IIIC/processed/"
        class_weight = [0.1181606 , 0.10036655, 0.18678813, 0.20368562, 0.19775413, 0.19324496]
        n_classes = 6

        num_workers = 0
        in_channels = 16
        sampling_rate = 200
        sample_length = 10

    elif dataset_name == "TUEV":
        dataset_root = "C:/Dataset/raw/tuh_eeg_events/v2.0.1/edf/processed/"
        class_weight = [0.5711400335475981, 0.02574656125416239, 0.06524567482806858, 0.3027807097490057, 0.02898413631785354, 0.006102884303311568]
        n_classes = 6

        num_workers = 0
        in_channels = 16
        sampling_rate = 200
        sample_length = 5

    else:
        # stop the program
        exit()

    if num_workers == 0:
        persistent_workers = False
    else:
        persistent_workers = True


    model = None
    if model_name == "SPaRCNet":
        print("SPaRCNet")
        model = SPaRCNet(
            in_channels=in_channels,
            sample_length=int(sampling_rate * sample_length),
            n_classes=n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif model_name == "CNNTransformer":
        print("CNNTransformer")
        model = CNNTransformer(
        in_channels=in_channels,
        n_classes=n_classes,
        fft=200,
        steps=20,
        dropout=0.2,
        nhead=4,
        emb_size=256,
        )

    elif model_name == "FFCL":
        print("FFCL")
        model = FFCL(
            in_channels=in_channels,
            n_classes=n_classes,
            fft=200,
            steps=20,
            sample_length=int(sampling_rate * sample_length),
            shrink_steps=20,
        )

    elif model_name == "ContraWR":
        print("ContraWR")
        model = ContraWR(in_channels=in_channels,
                         n_classes=n_classes,
                         fft=200,
                         steps=20)

    elif model_name == "STTransformer":
        print("STTransformer")
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=n_classes,
            channel_length=int(sampling_rate * sample_length)
        )

    elif model_name == "BIOT":
        print("BIOT")
        model = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=n_classes,
            n_fft=200,
            hop_length=100
        )

    elif model_name == "BIOT-pretrain-PREST":
        print("BIOT-pretrain-PREST")
        pretrained_model_path = "../pretrained-models/EEG-PREST-16-channels.ckpt"
        model = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=n_classes,
            n_fft=200,
            hop_length=100,
            n_channels=16,  # here is 16
        )
        model.biot.load_state_dict(torch.load(pretrained_model_path))

    elif model_name == "BIOT-pretrain-SHHS+PREST":
        print("BIOT-pretrain-SHHS+PREST")
        pretrained_model_path = "../pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"
        model = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=n_classes,
            n_fft=200,
            hop_length=100,
            n_channels=18,  # here is 18
        )
        model.biot.load_state_dict(torch.load(pretrained_model_path))


    elif model_name == "BIOT-pretrain-six-datasets":
        print("BIOT-pretrain-six-datasets")
        pretrained_model_path = "../pretrained-models/EEG-six-datasets-18-channels.ckpt"
        model = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=n_classes,
            n_fft=200,
            hop_length=100,
            n_channels=18,  # here is 18
        )
        model.biot.load_state_dict(torch.load(pretrained_model_path))

    elif model_name == "BIOT-pretrain-IIIC+TUAB":
        print("BIOT-pretrain-IIIC+TUAB")
        pretrained_model_path = "../pretrained-models/EEG-IIIC+TUAB-18-channels.ckpt"
        model = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=n_classes,
            n_fft=200,
            hop_length=100,
            n_channels=18,  # here is 18
        )
        model.biot.load_state_dict(torch.load(pretrained_model_path))

    else:
        # stop the program
        exit()



    # create data loaders
    if dataset_name == "TUAB":

        # load the train, val, test files

        train_files = os.listdir(os.path.join(dataset_root, "train"))
        val_files = os.listdir(os.path.join(dataset_root, "val"))
        test_files = os.listdir(os.path.join(dataset_root, "test"))

        train_dataset = TUABLoader(os.path.join(dataset_root, "train"), train_files, sampling_rate)
        val_dataset = TUABLoader(os.path.join(dataset_root, "val"), val_files, sampling_rate)
        test_dataset = TUABLoader(os.path.join(dataset_root, "test"), test_files, sampling_rate)


    elif dataset_name == "IIIC":

        # load the train, val, test files
        train_dataset = IIICLoader(os.path.join(dataset_root, "train_X.npy"), os.path.join(dataset_root, "train_Y.npy"), sampling_rate)
        val_dataset = IIICLoader(os.path.join(dataset_root, "val_X.npy"), os.path.join(dataset_root, "val_Y.npy"), sampling_rate)
        test_dataset = IIICLoader(os.path.join(dataset_root, "test_X.npy"), os.path.join(dataset_root, "test_Y.npy"), sampling_rate)


    elif dataset_name == "TUEV":
        # load the train, val, test files
        train_dataset = TUEVLoader(os.path.join(dataset_root, "train_X.npy"), os.path.join(dataset_root, "train_Y.npy"), sampling_rate)
        val_dataset = TUEVLoader(os.path.join(dataset_root, "val_X.npy"), os.path.join(dataset_root, "val_Y.npy"), sampling_rate)
        test_dataset = TUEVLoader(os.path.join(dataset_root, "test_X.npy"), os.path.join(dataset_root, "test_Y.npy"), sampling_rate)


    else:
        # stop the program
        print("Dataset not found")
        exit()


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    # for i, (X, y) in enumerate(val_loader):
    #     print(X.shape, y.shape)
    #     if i == 0:
    #         break
    #
    # # try to loop the train_loader
    # for i, (X, y) in enumerate(train_loader):
    #     print(X.shape, y.shape)
    #     if i == 0:
    #         break





    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))

    set_seed(42)

    criterion = nn.CrossEntropyLoss()
    if class_weight is not None:
        class_weight = torch.tensor(class_weight).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)


    del train_loader
    del val_loader

    # Testing
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_auroc, test_sensitivity, test_specificity, test_f1, test_balanced_accuracy = evaluate(
        model, test_loader, criterion, device, n_classes
    )

    print(
        f"Test Loss: {test_loss:.4f} | Test AUC: {test_auroc:.4f} | Test Sensitivity: {test_sensitivity:.4f} | Test Specificity: {test_specificity:.4f} | Test F1: {test_f1:.4f} | Test Balanced Accuracy: {test_balanced_accuracy:.4f}")








