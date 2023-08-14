import torch
from torch import nn
from typing import List


class CnnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        """Block of CNN
        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            stride (int, optional): Stride. Defaults to 2.
        """
        super(CnnBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features: List[int] = [64, 128, 256, 512]):
        super().__init__()

        # First Layer
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # Intermediate Layers
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            stride = 1 if feature == features[-1] else 2
            layers.append(
                CnnBlock(in_channels, feature, stride=stride),
            )
            in_channels = feature

        # Final Layer
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )

        # Combine it all together
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator model.

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): Real or Fake sample.

        Returns:
            torch.Tensor: _description_
        """
        # Concatenate x and y along the channel dimension.
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    predictions = model(x, y)
    # assert
    print(predictions.shape)


if __name__ == "__main__":
    test()
