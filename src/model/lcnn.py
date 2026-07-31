import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    """Select the largest value from each pair of feature maps."""

    def forward(self, x):
        first, second = torch.chunk(x, 2, dim=1)
        return torch.maximum(first, second)


class MFMConv2d(nn.Module):
    """(Conv2d + MaxFeatureMap) """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding),
            MaxFeatureMap(),
        )

    def forward(self, x):
        return self.layers(x)


class LCNN(nn.Module):


    input_shape = (863, 600)
    final_feature_shape = (32, 53, 37)

    def __init__(self, n_classes=2, dropout=0.75):
        super().__init__()

        # The model copies the STC LCNN from the paper
        self.features = nn.Sequential(
            MFMConv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            MFMConv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            MFMConv2d(32, 48, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(48),
            MFMConv2d(48, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            MFMConv2d(48, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            MFMConv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            MFMConv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            MFMConv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            MFMConv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
        )

        feature_dim = 32 * 53 * 37
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 160),
            MaxFeatureMap(),
        )
        #dropout is before the final BatchNorm. as required :-)
        self.dropout = nn.Dropout(dropout)
        self.final_batch_norm = nn.BatchNorm1d(80)
        self.classifier = nn.Linear(80, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """init. conv., linear, and batch norm. layers using kaiming"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, data_object, **batch):
        if data_object.ndim != 4 or tuple(data_object.shape[-2:]) != self.input_shape:
            raise ValueError(
                "LCNN expects data_object with shape (batch, 1, 863, 600), "
                f"got {tuple(data_object.shape)}."
            )

        x = self.features(data_object)
        expected_shape = self.final_feature_shape
        if tuple(x.shape[1:]) != expected_shape:
            raise RuntimeError(
                f"Unexpected STC-LCNN feature shape {tuple(x.shape[1:])}; "
                f"expected {expected_shape}."
            )
        x = x.flatten(start_dim=1)
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.final_batch_norm(x)
        return {"logits": self.classifier(x)}

    def __str__(self):
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        result_info = super().__str__()
        return (
            f"{result_info}\nAll parameters: {all_parameters}"
            f"\nTrainable parameters: {trainable_parameters}"
        )
