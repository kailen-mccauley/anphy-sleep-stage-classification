import torch
import torch.nn as nn

class LSTMContextOnlyBody(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(LSTMContextOnlyBody, self).__init__()
        ### TODO: BEGIN SOLUTION ###
        self.model_sequential = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels= 32, kernel_size=3),
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

            # nn.Flatten(),
            
            # nn.Linear(5824, 1028),
            # nn.ReLU(),
            # nn.Linear(1028, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 6),
        )
        

        self.lstm = nn.LSTM(32, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64 * 2, 6)

    def forward(self, x):
        outs = None
        ### TODO: BEGIN SOLUTION ###
        # print(f"dimensions of x: {x.shape}")

        outs = self.model_sequential(x)
        # print(f"dimensions of outs after cnn: {outs.shape}")
        outs = outs.permute(0, 2, 1)
        self.lstm.flatten_parameters()
        outs, (hn, cn) = self.lstm(outs)
        final_hn_forward_state = hn[0]
        final_hn_backward_state = hn[1]
        hidden_cat = torch.cat((final_hn_forward_state, final_hn_backward_state), dim=1)

        outs = self.fc(hidden_cat)
        ### END SOLUTION ###
        return outs
    
