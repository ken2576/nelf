import torch
from .base_mlp import MLP

class IBRNet(torch.nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.dirc_mlp = MLP(feature_nums=[3, 16, (3 + latent_size)])
        self.feature_mlp0 = MLP(feature_nums=[3 * (3 + latent_size), 64, 32])
        self.feature_mlp1 = MLP(feature_nums=[32, 32, 32])
        self.weight_mlp0 = MLP(feature_nums=[32, 32, 1])
        self.weight_mlp1 = MLP(feature_nums=[32, 32, 1])
        self.density_mlp = MLP(feature_nums=[64, 64, 1])
        self.blend_mlp = MLP(feature_nums=[35, 16, 8, 1])
        self.softplus = torch.nn.Softplus()


    def forward(self, source_rgb, latent, source_dirc, target_dirc):
        # source_rgb: (batch_size, view_num, 3)
        # latent: (batch_size, view_num, latent_size)
        # source_dirc: (batch_size, view_num, 3)
        # target_dirc: (batch_size, 3)
        view_num = latent.shape[1]
        d_dirc = source_dirc - target_dirc.unsqueeze(1)

        feature = self.dirc_mlp(d_dirc) + torch.cat([source_rgb, latent], axis=-1)
        feature0 = self.feature_mlp0(torch.cat([
            feature,
            torch.mean(feature, axis=1, keepdim=True).expand(-1, view_num, -1),
            torch.var(feature, axis=1, keepdim=True).expand(-1, view_num, -1),
        ], axis=-1))
        feature1 = self.feature_mlp1(feature0)
        weight0 = torch.sigmoid(self.weight_mlp0(feature0))
        weight1 = torch.sigmoid(self.weight_mlp0(weight0 * feature1))
        weight1 = weight1 / torch.sum(weight1, axis=1, keepdim=True)
        
        weight_mean = torch.sum(weight1 * feature1, axis=1)
        weight_var = torch.sum(weight1 * feature1 ** 2, axis=1) - weight_mean ** 2
        density = self.softplus(self.density_mlp(torch.cat([weight_mean, weight_var], axis=-1)))

        blend_weight = self.blend_mlp(torch.cat([feature1, d_dirc], axis=-1))
        target_rgb = torch.sum(torch.softmax(blend_weight, axis=1) * source_rgb, axis=1)

        return density, target_rgb
