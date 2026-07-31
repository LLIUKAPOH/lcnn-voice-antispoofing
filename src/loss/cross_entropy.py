import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()

        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)

        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits, labels, **batch):
        return {"loss": self.loss(logits, labels)}
