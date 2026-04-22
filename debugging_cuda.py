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

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")