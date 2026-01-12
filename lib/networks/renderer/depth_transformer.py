import torch
import torch.nn as nn
import math

class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Q, d_model)
        Q = x.size(1)
        return x + self.pe[:Q].unsqueeze(0)

class DepthConfidenceTransformer(nn.Module):
    """
    Input:  var_scalar (B, Q, 1)
    Output: conf_score (B, Q, 1)  (higher = more confident)
    """
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 64,
        dropout: float = 0.0,
        max_len: int = 512,
    ):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos = SinusoidalPosEnc(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output a confidence logit per query point
        self.out = nn.Linear(d_model, 1)

    def forward(self, var_scalar: torch.Tensor) -> torch.Tensor:
        # var_scalar: (B, Q, 1)
        x = self.in_proj(var_scalar)      # (B, Q, d_model)
        x = self.pos(x)                   # (B, Q, d_model)
        x = self.enc(x)                   # (B, Q, d_model)
        conf = self.out(x)                # (B, Q, 1)
        return conf
