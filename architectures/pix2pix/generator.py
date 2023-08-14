import torch
from torch import nn


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample=False,
        activation="relu",
        use_dropout=False,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            if upsample
            else nn.Conv2d(
                in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(
            hidden_channels,
            hidden_channels * 2,
            upsample=False,
            activation="leakyReLU",
            use_dropout=False,
        )
        self.down2 = Block(
            hidden_channels * 2,
            hidden_channels * 4,
            upsample=False,
            activation="leakyReLU",
            use_dropout=False,
        )
        self.down3 = Block(
            hidden_channels * 4,
            hidden_channels * 8,
            upsample=False,
            activation="leakyReLU",
            use_dropout=False,
        )
        self.down4 = Block(
            hidden_channels * 8,
            hidden_channels * 8,
            upsample=False,
            activation="leakyReLU",
            use_dropout=False,
        )
        self.down5 = Block(
            hidden_channels * 8,
            hidden_channels * 8,
            upsample=False,
            activation="leakyReLU",
            use_dropout=False,
        )
        self.down6 = Block(
            hidden_channels * 8,
            hidden_channels * 8,
            upsample=False,
            activation="leakyReLU",
            use_dropout=False,
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 8,
                hidden_channels * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
        )
        self.up1 = Block(
            hidden_channels * 8, hidden_channels * 8, upsample=True, use_dropout=True
        )
        self.up2 = Block(
            hidden_channels * 8 * 2,
            hidden_channels * 8,
            upsample=True,
            use_dropout=True,
        )
        self.up3 = Block(
            hidden_channels * 8 * 2,
            hidden_channels * 8,
            upsample=True,
            use_dropout=True,
        )
        self.up4 = Block(
            hidden_channels * 8 * 2,
            hidden_channels * 8,
            upsample=True,
            use_dropout=False,
        )
        self.up5 = Block(
            hidden_channels * 8 * 2,
            hidden_channels * 4,
            upsample=True,
            use_dropout=False,
        )
        self.up6 = Block(
            hidden_channels * 4 * 2,
            hidden_channels * 2,
            upsample=True,
            use_dropout=False,
        )
        self.up7 = Block(
            hidden_channels * 2 * 2, hidden_channels, upsample=True, use_dropout=False
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, in_channels, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        return self.final(torch.cat([up7, d1], dim=1))


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, hidden_channels=64)
    predictions = model(x)
    assert (
        predictions.shape == x.shape
    ), f"Wrong shape. {predictions.shape} shoudl be {x.shape}"


if __name__ == "__main__":
    test()
