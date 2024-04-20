import torch.nn as nn
import torch
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        is_down_sampling: bool = True,
        is_use_activation: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            (
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    padding_mode="reflect",
                    **kwargs,
                )
                if is_down_sampling
                else nn.ConvTranspose2d(
                    in_channels=input_channel, out_channels=output_channel, **kwargs
                )
            ),
            nn.InstanceNorm2d(output_channel),
            nn.ReLU(inplace=True) if is_use_activation else nn.Identity(),
        )

    def forward(self, x) -> Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1, stride=1),
            ConvBlock(
                channels,
                channels,
                is_use_activation=False,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

    def forward(self, x) -> Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self, image_channels: int = 3, num_features: int = 64, num_residuals: int = 9
    ) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                # by default downsample is true with stride of 2
                ConvBlock(
                    input_channel=num_features,
                    output_channel=num_features * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ConvBlock(
                    input_channel=num_features * 2,
                    output_channel=num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    input_channel=num_features * 4,
                    output_channel=num_features * 2,
                    is_down_sampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    input_channel=num_features * 2,
                    output_channel=num_features,
                    is_down_sampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )
        self.last = nn.Conv2d(
            in_channels=num_features,
            out_channels=image_channels,
            kernel_size=7,
            stride=2,
            padding=1,
            padding_mode="reflect",
        )

    def forward(self, x) -> Tensor:
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.last(x)
        return torch.tanh(x)


def test():
    x = torch.randn(1, 3, 126, 126)
    model = Generator()
    prediction = model.forward(x)
    print(f"Generator input shape: {x.shape}")
    print(f"Generator output shape: {prediction.shape}")


if __name__ == "__main__":
    test()
