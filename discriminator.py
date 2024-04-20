from typing import List
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules import InstanceNorm2d


class Block(nn.Module):
    def __init__(
        self, *, input_channels: int, output_channels: int, stride: int
    ) -> None:
        """
        This class takes input image and performs convolution on that image and returns the output of convolution

        params:
          input_channels (int): The input channel of image (red,green,blue) will make it 3
          output_channels (int): The output channel of image after convolution operation

        returns:
          The output of convolution operation after an input is passed through self.conv block
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=(4, 4),
                stride=stride,
                padding=1,
                padding_mode="reflect",
            ),
            InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x) -> Tensor:
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(
        self, input_channels: int = 3, features: List[int] = [64, 128, 256, 512]
    ) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        input_channels = features[0]
        # because nn.initial changes the input to 64 channels
        for feature in features[1:]:
            layers.append(
                Block(
                    input_channels=input_channels,
                    output_channels=feature,
                    stride=1 if feature == features[-1] else 2,
                )
            )  # if the feature is last then use stride = 1 else use stride = 2
            input_channels = feature

        # At this point the last input is of sie 512 and now since this is generator we want to output the probability
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=3,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.Sigmoid(),
            )
        )

        # Since we have blocks in our layers list we gonna past them through Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.randn(1, 3, 256, 256)
    model = Discriminator()
    prediction = model.forward(x)
    print(f"Discriminator input shape: {x.shape}")
    print(f"Discriminator output shape: {prediction.shape}")


if __name__ == "__main__":
    test()
