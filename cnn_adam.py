import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torch.nn.utils.rnn import pad_sequence
import time
import copy
import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import timeit
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class SleepDataset(Dataset):
    def __init__(self, file_path, length, window_size=5):
        
        self.file_path = file_path
        self.length = length
        self.data = torch.load(self.file_path, weights_only=False, mmap=True)
#         self.features = torch.from_numpy(np.transpose(data["X"], axes=(0, 2, 1))).float()
#         self.labels = torch.from_numpy(data["y"]).long()


    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        
        # return self.features[idx], self.labels[idx]
        feature = torch.from_numpy(self.data["X"][idx]).float()
        label = torch.tensor(self.data["y"][idx]).long()
        
        # Apply transpose here if needed based on original shape
        feature = feature.transpose(1, 0)
        return feature, label
    

def custom_collate(batch):
#     print(f"Length of batch: {len(batch)}")
#     print(f"First item in batch's first element: {batch[0][0].shape}")
#     print(f"First item in batch's second element: {batch[0][1].shape}")
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # Pad sequences to the length of the longest in the batch
#     print(f"Data list: {data}")
#     print(f"Labels list: {labels}" )
    padded_data = pad_sequence(data, batch_first=True)
    labels = torch.tensor(labels)
    return padded_data, labels

class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        ### TODO: BEGIN SOLUTION ###
        self.model_sequential = nn.Sequential(
            nn.Conv1d(in_channels=14, out_channels= 32, kernel_size=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv1d(in_channels=64, out_channels = 128, kernel_size=3),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels = 256, kernel_size=3),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=256, out_channels = 64, kernel_size=7),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels = 32, kernel_size=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Flatten(),
            
            nn.Linear(5824, 1028),
            nn.ReLU(),
            nn.Linear(1028, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        ### END SOLUTION ###

    def forward(self, x):
        outs = None
        ### TODO: BEGIN SOLUTION ###
        outs = self.model_sequential(x)
        ### END SOLUTION ###
        return outs
    


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct.item() / batch_size

    return acc


def train(epoch, data_loader, model, optimizer, criterion, scheduler):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        data = data.to(device)
        data = torch.transpose(data, 1, 2)
        # print(f"data shape: {data.shape}")
        target = target.to(device)

        # calculate model predictions, training loss, and update model parameters
        ### TODO: BEGIN SOLUTION ###
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        ### END SOLUTION ###

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % len(data_loader) == 0:
            print(
                (
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                ).format(
                    epoch,
                    idx,
                    len(data_loader),
                    iter_time=iter_time,
                    loss=losses,
                    top1=acc,
                )
            )


def validate(epoch, val_loader, model, criterion):
    """
    Hint: make sure to use torch.no_grad() to disable gradient computation. This will help reduce memory usage and speed up computation.
    """
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 6
    cm = torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        data = data.to(device)
        data = torch.transpose(data, 1, 2)
        target = target.to(device)

        # calculate model predictions and validation loss
        ### TODO: BEGIN SOLUTION ###
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

            
        ### END SOLUTION ###

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % len(val_loader) == 0:
            print(
                (
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                ).format(
                    epoch,
                    idx,
                    len(val_loader),
                    iter_time=iter_time,
                    loss=losses,
                    top1=acc,
                )
            )
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm


    

if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    assert device == "cuda"


    data_dir = 'anphy_sleep_data/patient_records/clean'       
    file_paths = glob.glob(os.path.join(data_dir, '*.pt'))

    metadata = pd.read_csv("recording_epoch_nums.csv", header=None)
    metadata.sort_values(by=0, inplace=True)
    metadata = metadata.reset_index(drop=True)

    datasets = [SleepDataset(data_dir + "/" + metadata.loc[i, 0], metadata.loc[i, 1]) for i in range(len(metadata))]
    combined_dataset = ConcatDataset(datasets)

    

    lr = 0.001
    momentum = 0.9
    weight_decay = 0.001
    epochs = 1000
    warmup = 0
    steps = [6, 8]
    
    model = MyModel()
    model = model.to(device)
    
    n = len(combined_dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(
    combined_dataset, 
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42) # Use a generator for reproducible splits
    )

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        
    best = 0.0
    best_cm = None
    best_model = model
    for epoch in range(epochs):

        # train loop
        train(epoch, train_loader, model, optimizer, criterion, scheduler)

        # validation loop
        acc, cm = validate(epoch, val_loader, model, criterion)

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)

    print("Best Prec @1 Acccuracy: {:.4f}".format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))
    
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    torch.save(
            best_model.state_dict(), "./checkpoints/" + "best_model_adam" + "_with_eval" + ".pth"
        )

    best_model.eval()
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 6
    cm = torch.zeros(num_class, num_class)

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            start = time.time()

            data = data.to(device)
            data = torch.transpose(data, 1, 2)
            target = target.to(device)

            out = model(data)
            loss = criterion(out, target)


            batch_acc = accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss, out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if idx % len(val_loader) == 0:
                print(
                    (
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                    ).format(
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Test Accuracy of Class {}: {:.4f}".format(i, acc_i))

        print("*Test Prec @1: {top1.avg:.4f}".format(top1=acc))
