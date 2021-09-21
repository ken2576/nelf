import torch
import numpy as np
from .base_mlp import MLP, ResBlocks, PosEncodeResnet


class NelfNet(torch.nn.Module):
    def __init__(self, latent_size, light_size):
        super().__init__()
        self.light_size = light_size

        self.dirc_mlp = MLP(feature_nums=[3 + 3, 16, (3 + latent_size)])
        self.geometry_mlp = ResBlocks(
            input_size=3 * (3 + latent_size),
            hidden_size=64, block_num=2, output_size=64 + 1
        )
        self.density_mlp = MLP(feature_nums=[128, 32, 1])

        self.lt_mlp = ResBlocks(
            input_size=3 + latent_size + 3 + 64,
            hidden_size=128, block_num=2, output_size=3 * np.prod(light_size)
        )
        self.blend_mlp = MLP(feature_nums=[3 + 3 + 64, 32, 16, 1])
        self.softplus = torch.nn.Softplus()



    def forward(self, source_rgb, latent, source_dirc, target_dirc, target_irradiance):
        # rgb: (batch_size, view_num, 3)
        # latent: (batch_size, view_num, latent_size)
        # source_dirc: (batch_size, view_num, 3)
        # target_dirc: (batch_size, 3)
        # target_irradiance: (1, 3, light_size[0], light_size[1])
        view_num = latent.shape[1]
        dircs = torch.cat([
            source_dirc, target_dirc.unsqueeze(1).expand(-1, view_num, -1)
        ], axis=-1)

        input_feature = self.dirc_mlp(dircs) + torch.cat([source_rgb, latent], axis=-1)
        output = self.geometry_mlp(torch.cat([
            input_feature,
            torch.mean(input_feature, axis=1, keepdim=True).expand(-1, view_num, -1),
            torch.var(input_feature, axis=1, keepdim=True).expand(-1, view_num, -1)
        ], axis=-1))
        feature, weight = output[..., :-1], torch.sigmoid(output[..., -1:])

        weight_mean = torch.sum(weight * feature, axis=1)
        weight_var = torch.sum(weight * feature ** 2, axis=1) - weight_mean ** 2
        density = self.softplus(self.density_mlp(torch.cat([weight_mean, weight_var], axis=-1)))

        lt_scale = self.softplus(self.lt_mlp(torch.cat([
            source_rgb, latent, source_dirc, feature,
        ], axis=-1)))
        rgb_perview = source_rgb * torch.sum(
            lt_scale.view(-1, view_num, 3, np.prod(self.light_size))
            * target_irradiance.view(1, 1, 3, np.prod(self.light_size)),
            axis=-1
        )

        blend_weight = self.blend_mlp(torch.cat([dircs, feature], axis=-1))
        rgb = torch.sum(torch.softmax(blend_weight, axis=1) * rgb_perview, axis=1)

        return density, rgb

