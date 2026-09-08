"""
Small Transformer-based classifier for sequences.
"""

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x (batch, T, d_model)
        L = x.size(1)
        return x + self.pe[:, :L, :].to(x.device)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=42, d_model=128, nhead=8, num_layers=3, num_classes=10, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, D, T)
        x = x.permute(0,2,1)  # (batch, T, D)
        x = self.fc_in(x)  # (batch, T, d_model)
        x = self.pos(x)
        out = self.encoder(x)  # (batch, T, d_model)
        # pool across time
        out = out.permute(0,2,1)  # (batch, d_model, T)
        out = self.pool(out).squeeze(-1)  # (batch, d_model)
        logits = self.fc_out(out)
        return logits
