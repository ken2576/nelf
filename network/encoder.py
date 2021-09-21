import torch
import numpy as np

from .base_conv import NormAct, NormActConv, ResConvBlock


class ImageEncoderWithLight(torch.nn.Module):
    def __init__(self, input_channel=3, light_size=(8, 16)):
        super().__init__()

        self.light_size = light_size
        self.light_pixel_num = np.prod(light_size)
        self.channel_num = [input_channel, 32, 64, 128, 256]

        # Encoder
        self.resblocks = torch.nn.ModuleList([
            ResConvBlock(
                input_channel=self.channel_num[i],
                hidden_channel=self.channel_num[i+1],
                output_channel=self.channel_num[i+1],
                stride=2
            )
            for i in range(len(self.channel_num) - 1)
        ])
        if self.light_size is not None:
            self.lightpred = NormActConv(
                input_channel=self.channel_num[-1],
                output_channel=self.light_pixel_num * 4,
                kernel_size=3,
                stride=1
            )
            self.lightpred_final = NormAct(
                channel_num=self.light_pixel_num * 4,
                act='softplus'
            )

        # Decoder
        self.upconvs = torch.nn.ModuleList([
            NormActConv(
                input_channel=self.channel_num[i+1],
                output_channel=self.channel_num[i],
                kernel_size=3,
                stride=1
            )
            for i in range(len(self.channel_num) - 2, 0, -1)
        ])
        self.catconvs = torch.nn.ModuleList([
            NormActConv(
                input_channel=self.channel_num[i] * 2,
                output_channel=self.channel_num[i] * (1 if i != 1 else 2),
                kernel_size=(3 if i != 1 else 1),
                stride=1
            )
            for i in range(len(self.channel_num) - 2, 0, -1)
        ])


    def get_feature_channel_num(self):
        return self.channel_num[1]


    def forward(self, x):
        # Encoder
        encoder = []
        for i in range(len(self.resblocks)):
            x = self.resblocks[i](x)
            encoder.append(x)

        if self.light_size is not None:
            light_feature = self.lightpred_final(self.lightpred(x))

        # Decoder
        upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        for i in range(len(self.upconvs)):
            cat = torch.cat([
                self.upconvs[i](upsample(x)),
                encoder[-(2+i)]
            ], axis=1)
            x = self.catconvs[i](cat)
        latent = x

        if self.light_size is None:
            return latent
        else:
            return latent, light_feature

