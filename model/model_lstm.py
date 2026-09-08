# import torch
# import torch.nn as nn

# class LSTMModel(nn.Module):
#     def __init__(self, input_size=42, hidden_size=128, output_size=10, num_layers=2):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # LSTM layer
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         # Fully connected layer
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x shape: (batch, seq_len, input_size)
#         out, (hn, cn) = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # take last time step
#         return out


import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, output_size=10):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        output, (h, c) = self.lstm(x)
        out = self.fc(h[-1])  # last hidden state
        return out
