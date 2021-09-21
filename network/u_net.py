import torch
import numpy as np
from .base_conv import ConvNormAct

class PortraitRelightingNet(torch.nn.Module):
    def __init__(self, light_size):
        super().__init__()
        self.light_size = light_size

        self.channel_nums = [3, 32, 64, 64, 128, 128, 256, 256, 512, 512, 512]
        
        self.image_encoders = torch.nn.ModuleList([
            ConvNormAct(
                self.channel_nums[i],
                self.channel_nums[i+1] - (self.channel_nums[i] if i == 0 else 0),
                7 if i == 0 else 3,
                2 if i in [1, 3, 5, 7] else 1
            )
            for i in range(len(self.channel_nums) - 1)
        ])
        self.light_decoders = torch.nn.ModuleList([
            ConvNormAct(512, 512, 3, 1),
            ConvNormAct(512, 4 * np.prod(light_size), 3, 1, act='softplus')
        ])
        self.light_encoders = torch.nn.ModuleList([
            ConvNormAct(3 * np.prod(light_size), 512, 1, 1), ConvNormAct(512, 256, 1, 1)
        ])
        self.image_decoders = torch.nn.ModuleList([
            ConvNormAct(
                self.channel_nums[i] + (self.channel_nums[i] if i < len(self.channel_nums) - 1 else 256),
                self.channel_nums[i-1],
                3,
                1,
                'prelu' if i != 1 else 'sigmoid'
            )
            for i in range(len(self.channel_nums) - 1, 0, -1)
        ])
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        


    def forward(self, source_image, target_light):
        x = source_image

        encoders_features = []
        for i, block in enumerate(self.image_encoders):
            x = block(x) if i != 0 else torch.cat([x, block(x)], axis=1)
            encoders_features.append(x)

        for block in self.light_decoders:
            x = block(x)
        light = x[:, :np.prod(self.light_size) * 3, :, :].view(1, np.prod(self.light_size), 3, -1)
        confidence = x[:, np.prod(self.light_size) * 3:, :, :].view(1, np.prod(self.light_size), 1, -1)
        source_light = torch.sum(light * confidence, axis=-1) / torch.sum(confidence, axis=-1)

        if target_light.ndim <= 2:
            x = torch.roll(
                source_light.view(*self.light_size, 3),
                -target_light.squeeze().int().item(),
                dims=1
            ).view(1, -1, 1, 1)
        else:
            x = target_light.reshape(1, -1, 1, 1)

        for block in self.light_encoders:
            x = block(x)

        x = x.expand(-1, -1, *encoders_features[-1].shape[2:])
        for i, block in enumerate(self.image_decoders):
            x = torch.cat([x, encoders_features[-i-1]], axis=1)
            if i in [2, 4, 6, 8]:
                x = self.upsample(x)
            x = block(x)


        return x, source_light.view(1, *self.light_size, 3)
        
