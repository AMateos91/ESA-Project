from __future__ import annotations

import torch
import torch.nn as nn



class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )


    def forward(
        self,
        x
    ):

        return self.block(x)



class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )


    def forward(
        self,
        x
    ):

        x = self.pool(x)

        return self.conv(x)



class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):

        super().__init__()


        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )


        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )


    def forward(
        self,
        x,
        skip
    ):


        x = self.up(x)


        diff_y = (
            skip.size()[2]
            -
            x.size()[2]
        )


        diff_x = (
            skip.size()[3]
            -
            x.size()[3]
        )


        if diff_x != 0 or diff_y != 0:

            x = nn.functional.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ]
            )


        x = torch.cat(
            [
                skip,
                x
            ],
            dim=1
        )


        return self.conv(x)



class FireUNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        features=(64, 128, 256, 512),
    ):

        super().__init__()


        self.encoder1 = DoubleConv(
            in_channels,
            features[0]
        )


        self.encoder2 = DownBlock(
            features[0],
            features[1]
        )


        self.encoder3 = DownBlock(
            features[1],
            features[2]
        )


        self.encoder4 = DownBlock(
            features[2],
            features[3]
        )


        self.bottleneck = DownBlock(
            features[3],
            features[3] * 2
        )


        self.decoder4 = UpBlock(
            features[3] * 2,
            features[3],
            features[3]
        )


        self.decoder3 = UpBlock(
            features[3],
            features[2],
            features[2]
        )


        self.decoder2 = UpBlock(
            features[2],
            features[1],
            features[1]
        )


        self.decoder1 = UpBlock(
            features[1],
            features[0],
            features[0]
        )


        self.output = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=1
        )



    def forward(
        self,
        x
    ):


        e1 = self.encoder1(
            x
        )


        e2 = self.encoder2(
            e1
        )


        e3 = self.encoder3(
            e2
        )


        e4 = self.encoder4(
            e3
        )


        b = self.bottleneck(
            e4
        )


        d4 = self.decoder4(
            b,
            e4
        )


        d3 = self.decoder3(
            d4,
            e3
        )


        d2 = self.decoder2(
            d3,
            e2
        )


        d1 = self.decoder1(
            d2,
            e1
        )


        return torch.sigmoid(
            self.output(d1)
        )



if __name__ == "__main__":


    model = FireUNet(
        in_channels=7,
        out_channels=1
    )


    x = torch.randn(
        2,
        7,
        128,
        128
    )


    y = model(
        x
    )


    print(
        y.shape
    )
