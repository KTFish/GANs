import torch
import config
from torch import nn

# Strided Convolutions
# Paper: "Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator)."
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html


class GeneratorBlock:
    """Block of generator network."""

    def __init__(
        self, in_channels: int, out_channels: int, final_layer: bool = False
    ) -> None:
        # Paper: "Use ReLU activation in generator for all layers except for the output, which uses Tanh."
        # Paper: "Directly applying batchnorm to all layer however, resulted in sample oscillation and model instability.
        # This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer."
        if final_layer:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels),
                nn.Tanh(), # Output [-1, 1]
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    def __init__(
        self, in_channels: int = 1, out_channels: int = 10, hidden_units: int = 10
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(in_features=)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return 
