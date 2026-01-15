# depth_transformer/model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthTransformer(nn.Module):
    """
    Inputs:
      x: [B*N, Q, Din]  per-ray sequence over query depths
      z: [B*N, Q]       absolute camera-z samples (optional feature)
      t: [B*N, Q]       ray t samples (optional feature)
      valid_q: [B*N, Q] bool query-valid mask (optional)

    Outputs:
      logits: [B*N, Q] confidence per query point
      z_soft: [B*N]    refined depth via soft-argmax over logits (in camera-z)
      z_hard: [B*N]    refined depth via argmax
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,          # [BN,Q,Din]
        z_samples: torch.Tensor,  # [BN,Q]
        valid_q: torch.Tensor | None = None,  # [BN,Q] bool
        temperature: float = 1.0,
    ):
        h = self.proj(x)  # [BN,Q,d_model]

        # attention mask expects True for "mask out"
        attn_mask = None
        if valid_q is not None:
            attn_mask = ~valid_q  # [BN,Q]

        h = self.enc(h, src_key_padding_mask=attn_mask)  # [BN,Q,d_model]
        logits = self.head(h).squeeze(-1)                # [BN,Q]

        if valid_q is not None:
            #logits = logits.masked_fill(~valid_q, -1e9)
            neg = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~valid_q, neg)

        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)  # [BN,Q]
        z_soft = (probs * z_samples).sum(dim=-1)                    # [BN]
        k_hard = probs.argmax(dim=-1)                               # [BN]
        z_hard = z_samples.gather(1, k_hard[:, None]).squeeze(1)    # [BN]

        return {
            "logits": logits,
            "probs": probs,
            "z_soft": z_soft,
            "z_hard": z_hard,
            "k_hard": k_hard,
        }
