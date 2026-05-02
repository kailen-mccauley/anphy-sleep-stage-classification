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
from models import MyModel, MyLSTMModel, CNNContext, CNNContextOnlyBody, LSTMContext, LSTMContextOnlyBody, LSTMOnlyBody, CNNOnlyBody
import pyarrow as pa
import pyarrow.parquet as pq
import sys

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
parser.add_argument("--config", default="./configs/LSTM_base.yaml")



class SleepDataset(Dataset):
    def __init__(self, file_path, length, window_size=None):
        
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
        label = torch.tensor(self.data["y"][idx]).to(torch.long)
        
        # Apply transpose here if needed based on original shape
        feature = feature.transpose(1, 0)
        # feature = concat_samples(features, self.window_size)
        return feature, label
    
class SleepDatasetWithContext(Dataset):
    def __init__(self, file_path, length, window_size=2):
        self.file_path = file_path
        self.length = length
        self.data = torch.load(self.file_path, weights_only=False, mmap=True)
        self.window_size = window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # Handle boundaries for when index = 0 or end of data list
        prev_idx = max(idx - 1, 0)
        next_idx = min(idx + 1, self.length - 1)

        prev_feat = torch.from_numpy(self.data["X"][prev_idx]).to(torch.float32)
        curr_feat = torch.from_numpy(self.data["X"][idx]).to(torch.float32)
        next_feat = torch.from_numpy(self.data["X"][next_idx]).to(torch.float32)

        # transpose because Emily did and I trust her
        prev_feat = prev_feat.transpose(1, 0)
        curr_feat = curr_feat.transpose(1, 0)
        next_feat = next_feat.transpose(1, 0)

        # Concatenate (which dim tho?? i forgor)
        feature = torch.cat([prev_feat, curr_feat, next_feat], dim=1)

        # Label is the middle one
        label = torch.tensor(self.data["y"][idx]).to(torch.long)

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

    # max_len = max(x.shape[0] for x in batch)
    # print("Max length in batch:", max_len)
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


def train(epoch, data_loader, model, optimizer, scheduler, criterion, metric_collector):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # print("Train loop initiated!")

    model.train()

    total_correct = 0

    end_time = time.time()
    data_time = AverageMeter()
    for idx, (data, target) in enumerate(data_loader):
        # print("Data time: ", time.time() - start_data_time)
        data_time.update(time.time() - end_time)

        start = time.time()

        data = data.to(device)
        data = torch.transpose(data, 1, 2)
        # print(f"data shape: {data.shape}")
        target = target.to(device)
        # print(f"Target shape: {target.shape}")

        # print("Calculating model predictions")

        cm = torch.zeros(NUM_CLASS, NUM_CLASS)
        f1_metric = MulticlassF1Score(num_classes=NUM_CLASS, average="macro").to(device)

        # calculate model predictions, training loss, and update model parameters
        ### TODO: BEGIN SOLUTION ###
        # start_compute = time.time()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        # print("Compute time: ", time.time() - start_compute)

        if scheduler is not None:
            scheduler.step(epoch + idx / len(data_loader))
        
        ### END SOLUTION ###
        # print("Model predictions and gradients calculated")

        batch_acc = accuracy(out, target)

        _, preds = torch.max(out, 1)
        total_correct += preds.eq(target).sum() * 1.0
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1


        losses.update(loss.item(), out.shape[0])
        acc.update(batch_acc, out.shape[0])
        f1_metric.update(preds.to(device), target.to(device))

        iter_time.update(time.time() - start)
        if idx % len(data_loader) == 0:
            print(
                (
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Iter Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                    "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                ).format(
                    epoch,
                    idx,
                    len(data_loader),
                    iter_time=iter_time,
                    data_time=data_time,
                    loss=losses,
                    top1=acc,
                )
            )

        end_time = time.time()

    final_macro_f1 = f1_metric.compute()

    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    overall_accuracy = total_correct / len(data_loader.dataset)
    for i, acc_i in enumerate(per_cls_acc):
        print("Training Accuracy of Class {}: {:.4f}".format(i, acc_i))
    print(f"Total Overall Training Accuracy: {overall_accuracy}")
    print(f"Average Training Accuracy across Batches: {acc.avg}")
    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    print(f"Macro F1 Score on Training Set: {final_macro_f1.item()}")
    # print("Train loop ended!")
    metric_collector["accuracy"].append(acc.avg)
    metric_collector["loss"].append(losses.avg)
    metric_collector["f1"].append(final_macro_f1.item())


def validate(epoch, val_loader, model, criterion, metric_collector):
    """
    Hint: make sure to use torch.no_grad() to disable gradient computation. This will help reduce memory usage and speed up computation.
    """
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    data_time = AverageMeter()

    model.eval()

    
    cm = torch.zeros(NUM_CLASS, NUM_CLASS)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASS, average="macro").to(device)
    # evaluation loop
    # print("Val loop initiated!")

    total_correct = 0
    end_time = time.time()

    for idx, (data, target) in enumerate(val_loader):
        data_time.update(time.time() - end_time)

        start = time.time()

        data = data.to(device)
        data = torch.transpose(data, 1, 2)
        target = target.to(device)

        # calculate model predictions and validation loss
        ### TODO: BEGIN SOLUTION ###
        # print("Calculating val predictions")

        
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

            
        ### END SOLUTION ###
        # print("Val predictions finished")

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        total_correct += preds.eq(target).sum() * 1.0
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss.item(), out.shape[0])
        acc.update(batch_acc, out.shape[0])
        f1_metric.update(preds.to(device), target.to(device))

        iter_time.update(time.time() - start)
        if idx % len(val_loader) == 0:
        # if idx % 1 == 0:
            print(
                (
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Iter Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                    "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                ).format(
                    epoch,
                    idx,
                    len(val_loader),
                    iter_time=iter_time,
                    data_time = data_time,
                    loss=losses,
                    top1=acc,
                )
            )

        end_time = time.time()
    # print("Val loop ended!")
    final_macro_f1 = f1_metric.compute()

    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    overall_accuracy = total_correct / len(val_loader.dataset)
    for i, acc_i in enumerate(per_cls_acc):
        print("Validation Accuracy of Class {}: {:.4f}".format(i, acc_i))
    print(f"Total Overall Validation Accuracy: {overall_accuracy}")
    print(f"Average Val Accuracy across Batches: {acc.avg}")
    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    print(f"Macro F1 Score on Validation Set: {final_macro_f1.item()}")

    metric_collector["accuracy"].append(acc.avg)
    metric_collector["loss"].append(losses.avg)
    metric_collector["f1"].append(final_macro_f1.item())
    

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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, smaller_better=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.smaller_better = smaller_better

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        else:
            if self.smaller_better:
                if val_loss > self.best_loss - self.min_delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_loss = val_loss
                    self.counter = 0
            else:
                if val_loss < self.best_loss - self.min_delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_loss = val_loss
                    self.counter = 0


def get_balanced_weights(labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes).float()
    
    weights = labels.size(0) / (num_classes * counts)
    
    weights[torch.isinf(weights)] = 0.0
    
    return weights


def main():

    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
            print(f"{k}: {v}")
    

    generator = torch.Generator().manual_seed(int(args.seed))


    
    if args.body_only:
        data_dir = 'anphy_sleep_data/patient_records/only_body'
        metadata = pd.read_csv("recording_epoch_nums_body_only.csv", header=None)
    else:
        data_dir = 'anphy_sleep_data/patient_records/clean'
        metadata = pd.read_csv("recording_epoch_nums.csv", header=None)       
    
    
    metadata.sort_values(by=0, inplace=True)
    metadata = metadata.reset_index(drop=True)

    if args.context:
        datasets = [SleepDatasetWithContext(data_dir + "/" + metadata.loc[i, 0], metadata.loc[i, 1], 1) for i in range(len(metadata))]
    else:
        datasets = [SleepDataset(data_dir + "/" + metadata.loc[i, 0], metadata.loc[i, 1], 2) for i in range(len(metadata))]
    combined_dataset = ConcatDataset(datasets)
    # combined_dataset = GlobalSleepDataset(metadata, data_dir)
    print("Dataset created")
    
    model_name = f"{args.model}_"

    if args.context:
        model_name += "WITH_CONTEXT_"
    model_name += f"{args.optimizer}_optim"
    if args.scheduler:
        model_name += "_with_scheduler"
    else:
        model_name +="_noscheduler"
    if args.body_only:
        model_name += "_body_only"
    if args.class_weighting:
        model_name += "_with_class_weights"
    if args.early_stopping:
        model_name += "_earlystop"
    model_name += f"_seed_{args.seed}"

 
    match args.context:
        case False if args.model == "CNN" and args.body_only:
            model = CNNOnlyBody()
        case False if args.model == "CNN-LSTM" and args.body_only:
            model = LSTMOnlyBody()
        case False if args.model == "CNN":
            model = MyModel()
        case False if args.model == "CNN-LSTM":
            model = MyLSTMModel()
        case True if args.model == "CNN" and args.body_only:
            model = CNNContextOnlyBody()
        case True if args.model == "CNN-LSTM" and args.body_only:
            model = LSTMContextOnlyBody()
        case True if args.model == "CNN":
            model = CNNContext()
        case True if args.model == "CNN-LSTM":
            model = LSTMContext()
        case _:
            raise ValueError("No valid model specified")
    model = model.to(device)
    
    n = len(combined_dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    train_dataset, val_dataset, test_dataset = random_split(
    combined_dataset, 
    [n_train, n_val, n_test],
    generator=generator) # Use a generator for reproducible splits
    print("Dataset split complete")

    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=1, collate_fn=custom_collate)  
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, prefetch_factor=1, collate_fn=custom_collate)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, prefetch_factor=1, collate_fn=custom_collate)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, collate_fn=custom_collate)


    print("Data loaders initiated")
    print(f"Length of test loader: {len(test_loader)}")
    assert len(test_loader) > 0

    criterion = None
    if args.class_weighting:

        # print("Before getting y train")
        y_train = torch.cat([label for _, label in train_loader])
        assert(y_train.size(0) == n_train)
        # print("Successfully retrieved y train")
        class_weights = get_balanced_weights(y_train, NUM_CLASS).to(device)
        print("Got class weights successfully")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = None
    if args.optimizer == "base":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum = args.momentum,
            weight_decay = args.reg,
        )
    elif args.optimizer == "ADAM":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = args.learning_rate,
            weight_decay = args.reg
        )
    else:
        raise ValueError("No valid optimizer specified")

    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10, 
            T_mult=2, 
            eta_min=1e-6
        )


    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopping(args.patience, args.min_delta, smaller_better=False)
   
        
    best = 0.0
    best_cm = None
    best_model = model
    peak_val_accuracy_epoch = 0

    train_metrics_collector = {"accuracy": [], "loss": [], "f1": []}
    val_metrics_collector = {"accuracy": [], "loss": [], "f1": []}
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        if args.optimizer == "base":
            adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train(epoch, train_loader, model, optimizer, scheduler, criterion, train_metrics_collector)

        # validation loop
        acc, cm = validate(epoch, val_loader, model, criterion, val_metrics_collector)

        if early_stopper is not None:
            early_stopper(acc)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if acc > best:
            best = acc
            peak_val_accuracy_epoch = epoch
            best_cm = cm
            best_model = copy.deepcopy(model)
        
        

    print("Best Prec @1 Acccuracy: {:.4f}".format(best))
    print(f"Epoch where best accuracy reached: {peak_val_accuracy_epoch}")
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Best Validation Accuracy of Class {}: {:.4f}".format(i, acc_i))
    
    torch.cuda.synchronize()
    if args.save_best:
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        torch.save(
                best_model.state_dict(), "./checkpoints/" + model_name + "_best_model" + ".pth"
            )
        
    # print("Preparing for testing...")

    best_model.eval()
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    data_time = AverageMeter()
    total_correct = 0

    
    cm = torch.zeros(NUM_CLASS, NUM_CLASS)

    f1_metric = MulticlassF1Score(num_classes=NUM_CLASS, average="macro").to(device)

    # output_dir = "results_files"
    # os.makedirs(output_dir, exist_ok=True)

    output_dir = Path(f"results_files/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    end_time = time.time()

    all_preds = []
    all_targets = []

    # print("Testing initiated!")

    test_metrics_collector = {"accuracy": [], "loss": [], "f1": []}

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if idx == 0:
                print("Test loader successfully iterating")

            data_time.update(time.time() - end_time)

            start = time.time()

            data = data.to(device)
            data = torch.transpose(data, 1, 2)
            target = target.to(device)

            out = best_model(data)
            
            loss = criterion(out, target)


            batch_acc = accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            total_correct += preds.eq(target).sum() * 1.0
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])
            f1_metric.update(preds.to(device), target.to(device))

            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

            # batch_df = pd.DataFrame({
            #     "Predicted": preds.cpu().numpy(),
            #     "Actual": target.cpu().numpy()
            # })

            # batch_df.to_parquet(f"{output_dir}/{model_name}_batch_{idx}.parquet", index=False)

            iter_time.update(time.time() - start)
            if idx % len(test_loader) == 0:
                print(
                    (
                        "Iter Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                        "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    ).format(
                        iter_time=iter_time,
                        data_time = data_time,
                        loss=losses,
                        top1=acc,
                    )
                )
            end_time = time.time()
    final_macro_f1 = f1_metric.compute()
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    overall_accuracy = total_correct / len(test_loader.dataset)
    for i, acc_i in enumerate(per_cls_acc):
        print("Test Accuracy of Class {}: {:.4f}".format(i, acc_i))
    print(f"Macro F1 Score on Test Set: {final_macro_f1.item()}")
    print(f"Total Overall Test Accuracy: {overall_accuracy}")
    print(f"Average Test Accuracy Across Batches: {acc.avg}")
    print(f"Final Confusion Matrix: {cm}")

    test_metrics_collector["accuracy"].append(acc.avg)
    test_metrics_collector["loss"].append(losses.avg)
    test_metrics_collector["f1"].append(final_macro_f1.item())

    if args.save_loss:
        print("Saving losses")
        train_loss_df = pd.DataFrame(train_metrics_collector)
        val_loss_df = pd.DataFrame(val_metrics_collector)
        test_loss_df = pd.DataFrame(test_metrics_collector)

        train_loss_df.to_csv(f"{output_dir}/seed_{args.seed}_train_loss.csv", index_label="epoch")
        val_loss_df.to_csv(f"{output_dir}/seed_{args.seed}_val_loss.csv", index_label="epoch")
        test_loss_df.to_csv(f"{output_dir}/seed_{args.seed}_test_loss.csv", index_label="epoch")

    if args.save_preds:
        print("Saving results")
        

        writer = None
        for i in range(len(test_loader)):
            chunk = pd.DataFrame({
                "Predicted": all_preds[i].numpy(),
                "Actual": all_targets[i].numpy()
            })
            if i == 0:
                print(chunk.head())
            
            chunk = pa.Table.from_pandas(chunk)

            if writer is None:
                writer = pq.ParquetWriter(f"{output_dir}/results.parquet", chunk.schema)
            
            writer.write_table(chunk)
        
        if writer:
            writer.close()

            
        


    
   

if __name__ == "__main__":
    main()
    sys.exit(0)

        