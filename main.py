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
import argparse
import yaml
from torchmetrics.classification import MulticlassF1Score
from sklearn.utils import class_weight
from models import MyModel, MyLSTMModel

NUM_CLASS = 6


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
device = get_device()
print(f"Device: {device}")
# For the purposes of keeping PACE on track
assert device == "cuda"

parser = argparse.ArgumentParser(description="CS 7643 Sleep Stage Classification Project")
parser.add_argument("--config", default="./configs/config_mymodel.yaml")


def concat_samples(features, context_size=2):
    # print(f"Features shape: {features.shape}")
    N, D = features.shape
    padded = torch.nn.functional.pad(features, (0, 0, context_size, context_size), mode='constant', value=0)

    windows = []
    for i in range(N):
        # Slice the padded tensor to get the segment + context
        window = padded[i : i + (2 * context_size + 1), :]
        windows.append(window)

    stacked = torch.stack(windows, dim=0) # (Segments, Window_Size, Features)
    new_features = stacked.view(N, -1)
    # print(f"Final feature shape: {new_features.shape}")
    
    return new_features


class SleepDataset(Dataset):
    def __init__(self, file_path, length, window_size):
        
        self.file_path = file_path
        self.length = length
        self.data = torch.load(self.file_path, weights_only=False, mmap=True)
        self.window_size = window_size
#         self.features = torch.from_numpy(np.transpose(data["X"], axes=(0, 2, 1))).float()
#         self.labels = torch.from_numpy(data["y"]).long()


    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        
        # return self.features[idx], self.labels[idx]
        feature = torch.from_numpy(self.data["X"][idx]).to(torch.float32)
        label = torch.tensor(self.data["y"][idx]).to(torch.int)
        
        # Apply transpose here if needed based on original shape
        feature = feature.transpose(1, 0)
        # feature = concat_samples(features, self.window_size)
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


def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    print("Train loop initiated!")

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        data = data.to(device)
        data = torch.transpose(data, 1, 2)
        # print(f"data shape: {data.shape}")
        target = target.to(device)
        print(f"Target shape: {target.shape}")

        print("Calculating model predictions")

        # calculate model predictions, training loss, and update model parameters
        ### TODO: BEGIN SOLUTION ###
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        ### END SOLUTION ###
        print("Model predictions and gradients calculated")

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
    print("Train loop ended!")


def validate(epoch, val_loader, model, criterion):
    """
    Hint: make sure to use torch.no_grad() to disable gradient computation. This will help reduce memory usage and speed up computation.
    """
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    
    cm = torch.zeros(NUM_CLASS, NUM_CLASS)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASS, average="macro").to(device)
    # evaluation loop
    print("Val loop initiated!")
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        data = data.to(device)
        data = torch.transpose(data, 1, 2)
        target = target.to(device)

        # calculate model predictions and validation loss
        ### TODO: BEGIN SOLUTION ###
        print("Calculating val predictions")

        model.eval()
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

            
        ### END SOLUTION ###
        print("Val predictions finished")

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])
        f1_metric.update(preds.to(device), target.to(device))

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
    print("Val loop ended!")
    final_macro_f1 = f1_metric.compute()

    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    print(f"Macro F1 Score on Validation Set: {final_macro_f1.item()}")
    model.train()

    return acc.avg, cm


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_balanced_weights(labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes).float()
    
    weights = labels.size(0) / (num_classes * counts)
    
    weights[torch.isinf(weights)] = 0.0
    
    return weights


def main():
    # lr = 0.001
    # momentum = 0.9
    # weight_decay = 0.001
    # epochs = 1000
    # warmup = 0
    # steps = [6, 8]

    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    


    data_dir = 'anphy_sleep_data/patient_records/clean'       
    
    metadata = pd.read_csv("recording_epoch_nums.csv", header=None)
    metadata.sort_values(by=0, inplace=True)
    metadata = metadata.reset_index(drop=True)

    datasets = [SleepDataset(data_dir + "/" + metadata.loc[i, 0], metadata.loc[i, 1], 2) for i in range(len(metadata))]
    combined_dataset = ConcatDataset(datasets)
    

    
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

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)
    
    print("No memory issues with data loaders")
    # print("Before getting y train")
    # y_train = torch.cat([label for _, label in train_loader])
    # assert(y_train.size(0) == n_train)
    # print("Successfully retrieved y train")
    # class_weights = get_balanced_weights(y_train, 6).to(device)
    # print("Got class weights successfully")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum = args.momentum,
        weight_decay = args.reg,
    )
        
    best = 0.0
    best_cm = None
    best_model = model
    peak_val_accuracy_epoch = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_loader, model, optimizer, criterion)

        # validation loop
        acc, cm = validate(epoch, val_loader, model, criterion)

        if acc > best:
            best = acc
            peak_val_accuracy_epoch = epoch
            best_cm = cm
            best_model = copy.deepcopy(model)

    print("Best Prec @1 Acccuracy: {:.4f}".format(best))
    print(f"Epoch where best accuracy reached: {peak_val_accuracy_epoch}")
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))
    
    if args.save_best:
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        torch.save(
                best_model.state_dict(), "./checkpoints/" + "lstm_model" + ".pth"
            )

    best_model.eval()
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    
    cm = torch.zeros(NUM_CLASS, NUM_CLASS)

    f1_metric = MulticlassF1Score(num_classes=NUM_CLASS, average="macro").to(device)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            start = time.time()

            data = data.to(device)
            data = torch.transpose(data, 1, 2)
            target = target.to(device)

            out = best_model(data)
            all_preds.append(out.detach().cpu())
            all_targets.append(target.detach().cpu())
            loss = criterion(out, target)


            batch_acc = accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss, out.shape[0])
            acc.update(batch_acc, out.shape[0])
            f1_metric.update(preds.to(device), target.to(device))

            iter_time.update(time.time() - start)
            if idx % len(test_loader) == 0:
                print(
                    (
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                    ).format(
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )
        final_macro_f1 = f1_metric.compute()
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Test Accuracy of Class {}: {:.4f}".format(i, acc_i))
        print(f"Macro F1 Score on Test Set: {final_macro_f1.item()}")
        print(f"Final Confusion Matrix: {cm}")
    final_preds = torch.cat(all_preds, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    df = pd.DataFrame({
    'Predicted': final_preds.numpy(),
    'Actual': final_targets.numpy()
        })
    df.to_parquet('results.parquet', index=False)


if __name__ == "__main__":
    main()
    

        