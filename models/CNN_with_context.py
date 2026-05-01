import torch
import torch.nn as nn

class CNNContext(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(CNNContext, self).__init__()
        ### TODO: BEGIN SOLUTION ###
        self.model_sequential = nn.Sequential(
            nn.Conv1d(in_channels=42, out_channels= 32, kernel_size=3),
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
    
