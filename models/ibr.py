import numpy as np
import torch

from network.encoder import ImageEncoderWithLight
from network.ibrnet import IBRNet
from utils.camera import *
from utils.ray import *


class IBR(torch.nn.Module):
    def __init__(
        self,
        coarse_sample_num, fine_sample_num,
        train_ray_per_gpu, test_ray_per_gpu,
        train_mask_ratio, head_diameter
    ):
        super().__init__()

        self.coarse_sample_num = coarse_sample_num
        self.fine_sample_num = fine_sample_num
        self.train_total_ray = train_ray_per_gpu * torch.cuda.device_count()
        self.test_total_ray = test_ray_per_gpu * torch.cuda.device_count()
        self.train_mask_ratio = train_mask_ratio

        self.head_diameter = head_diameter

        self.image_encoder = ImageEncoderWithLight(light_size=None)
        self.latent_size = self.image_encoder.get_feature_channel_num()
        self.coarse_appearance_encoder = IBRNet(latent_size=self.latent_size)
        self.fine_appearance_encoder = IBRNet(latent_size=self.latent_size)


        self.register_buffer('source_intrinsic', None, persistent=False)
        self.register_buffer('source_extrinsic', None, persistent=False)
        self.register_buffer('source_cam_pos', None, persistent=False)

        self.register_buffer('source_shape', None, persistent=False)
        self.register_buffer('source_mask', None, persistent=False)
        self.register_buffer('source_image', None, persistent=False)
        self.register_buffer('source_feature', None, persistent=False)

        self.register_buffer('head_center', None, persistent=False)

    
    ############ Training Loss  ############
    def train_loss(self, output):
        rgb_loss = (
            torch.mean(torch.abs(output['coarse_rgb'] - output['target_rgb']))
            + torch.mean(torch.abs(output['fine_rgb'] - output['target_rgb']))
        )
        return {
            'rgb_loss': rgb_loss,
        }
    
    def val_loss(self, output, **data):
        rgb_loss = torch.mean(torch.abs(output['rgb'] - data['target_image']))
        return {
            'rgb_loss': rgb_loss,
        }


    ############ Class Entrance ############
    def render(self, train, model, **data):
        '''
        Given the image from the source_camera,
        render the image from the target_camera.
        '''
        self.train(train)
        with torch.set_grad_enabled(train):
            self.encode(
                data['source_shape'], data['source_intrinsic'], data['source_extrinsic'],
                data['source_image'], data['source_mask']
            )

        head_depth = torch.norm(
            self.head_center
            + torch.mv(torch.t(data['target_extrinsic'][:, :3]), data['target_extrinsic'][:, 3]),
            p=2
        ).item()

        if train:
            pixels_front, pixels_back = get_frontback_pix(data['target_mask']) # (pixel_num, XY)
            sample_num_front = int(self.train_total_ray * self.train_mask_ratio)
            sample_pixels = torch.cat([
                pixels_front[torch.randint(
                    high=pixels_front.shape[0], size=(sample_num_front, )
                ), :],
                pixels_back[torch.randint(
                    high=pixels_back.shape[0],
                    size=(self.train_total_ray - sample_num_front, )
                ), :]
            ], dim=0)                                                          # (sample_pixel_num, XY)
            rays = pix_to_ray(
                data['target_intrinsic'], data['target_extrinsic'], sample_pixels,
                head_depth - self.head_diameter, head_depth + self.head_diameter
            )                                                                  # (sample_pixel_num, 3 + 3 + 2)

            (coarse_rgb, coarse_depth, coarse_alpha,
             fine_rgb, fine_depth, fine_alpha
            ) = model(rays)
            target_rgb = data['target_image'][
                sample_pixels[:, 1].long(), sample_pixels[:, 0].long(), :
            ]
            target_depth = data['target_depth'][
                sample_pixels[:, 1].long(), sample_pixels[:, 0].long()
            ]
            target_alpha = data['target_mask'][
                sample_pixels[:, 1].long(), sample_pixels[:, 0].long()
            ]
            return {
                'coarse_rgb': coarse_rgb, 'fine_rgb': fine_rgb, 'target_rgb': target_rgb,
                'coarse_depth': coarse_depth, 'fine_depth': fine_depth, 'target_depth': target_depth,
                'coarse_alpha': coarse_alpha, 'fine_alpha': fine_alpha, 'target_alpha': target_alpha,
            }
        else:
            rgb, depth, alpha = [], [], []

            pixels = get_image_pix(shape=data['target_shape'])                  # (pixel_num, XY)
            rays = pix_to_ray(
                data['target_intrinsic'], data['target_extrinsic'], pixels,
                head_depth - self.head_diameter, head_depth + self.head_diameter
            )
            with torch.no_grad():
                for ray in torch.split(rays, self.test_total_ray):
                    _, _, _, rgb_chunk, depth_chunk, alpha_chunk = model(ray)
                    rgb.append(rgb_chunk)
                    depth.append(depth_chunk)
                    alpha.append(alpha_chunk)
                rgb = torch.cat(rgb, 0).view(data['target_shape'][0].item(), data['target_shape'][1].item(), 3)
                depth = torch.cat(depth, 0).view(data['target_shape'][0].item(), data['target_shape'][1].item())
                alpha = torch.cat(alpha, 0).view(data['target_shape'][0].item(), data['target_shape'][1].item())

            return {
                'rgb': rgb,
                'depth': depth,
                'alpha': alpha,
            }


    ############ Highlevel functions ############
    def encode(
            self,
            shape, intrinsic, extrinsic,
            image, mask
        ):
        self.source_intrinsic = intrinsic
        self.source_extrinsic = extrinsic
        self.source_cam_pos = -torch.matmul(
            extrinsic[..., :3].transpose(-2, -1), extrinsic[..., 3:]
        ).squeeze()

        self.source_shape = shape
        self.source_mask = torch.nn.MaxPool2d(kernel_size=7, stride=1, padding=3)(mask.unsqueeze(1))
        self.source_image = image.permute(0, 3, 1, 2)
        self.source_feature = self.image_encoder(self.source_image)

        self.head_center = self.source_intrinsic.new_zeros((3, ))


    def forward(self, ray):
        ray_num = ray.shape[0]
        
        ### Coarse ### 
        coarse_point_depth = sample_ray_depth(ray, self.coarse_sample_num)     # (ray_num, coarse_sample_num)
        coarse_pos, coarse_dirc, coarse_pix = self.sample_point(
            ray, coarse_point_depth
        ) 
        coarse_point_density, coarse_point_rgb = self.compute_color(
            coarse_pos, coarse_dirc, coarse_pix, 'coarse'
        )                                                                      # (ray_num * coarse_sample_num, 3/None)
        coarse_weight, coarse_rgb, coarse_depth, coarse_alpha = composite(
            coarse_point_depth, coarse_point_density, coarse_point_rgb
        )

        ### Fine ### 
        fine_point_depth = sample_ray_depth(
            ray, self.fine_sample_num, weight=coarse_weight
        )
        full_sample_num = self.coarse_sample_num + self.fine_sample_num
        full_point_depth, _ = torch.sort(
            torch.cat([coarse_point_depth, fine_point_depth], -1), -1
        )                                                                       # (ray_num, full_sample_num)
        full_pos, full_dirc, full_pix = self.sample_point(
            ray, full_point_depth
        ) 
        full_point_density, full_point_rgb = self.compute_color(
            full_pos, full_dirc, full_pix, 'fine'
        ) 
        _, full_rgb, full_depth, full_alpha = composite(
            full_point_depth, full_point_density, full_point_rgb
        )

        return coarse_rgb, coarse_depth, coarse_alpha, full_rgb, full_depth, full_alpha


    ############ Helpers ############
    def sample_point(self, ray, depth):
        pos, dirc = march_ray(ray, depth)                                      # (ray_num, sample_num, 3)

        # Project to self.camera
        pix = torch.matmul(
            torch.cat([
                pos, torch.ones_like(pos[..., :1])
            ], -1).unsqueeze(2).unsqueeze(2),                                  # (ray_num, sample_num, 1, 1, 4)
            torch.matmul(
                self.source_intrinsic, self.source_extrinsic
            ).permute(0, 2, 1)                                                 # (view_num, 4, 3)
        ).squeeze()                                                            # (ray_num, sample_num, view_num, 3)
        pix = pix[..., :2] / (pix[..., 2:] + 1e-6)
        pix /= self.source_shape[:, [1, 0]]
        pix = 2 * pix - 1                                                      # (ray_num, sample_num, view_num, XY)

        return pos, -dirc, pix


    def compute_color(self, pos, dirc, pix, mode):
        # Check mode
        if mode == 'coarse':
            mlp = self.coarse_appearance_encoder
            feature_start, feature_end = 0, self.latent_size
        elif mode == 'fine':
            mlp = self.fine_appearance_encoder
            feature_start, feature_end = self.latent_size, 2 * self.latent_size

        # Query visual hull
        visual_hull_mask = torch.sum(torch.nn.functional.grid_sample(
            self.source_mask, pix.permute(2, 0, 1, 3),
            align_corners=False
        )[:, 0, :, :], axis=0) == self.source_mask.shape[0]

        # Pass through Nerf
        point_density = pos.new_zeros(visual_hull_mask.shape)
        point_rgb = pos.new_zeros((*visual_hull_mask.shape, 3))
        if visual_hull_mask.sum() > 0:
            raw_rgb = torch.nn.functional.grid_sample(
                self.source_image,
                pix[visual_hull_mask].permute(1, 0, 2).unsqueeze(1),
                align_corners=False
            )[:, :, 0, :].permute(2, 0, 1)
            image_feature = torch.nn.functional.grid_sample(
                self.source_feature,
                pix[visual_hull_mask].permute(1, 0, 2).unsqueeze(1),
                align_corners=False
            )[:, feature_start:feature_end, 0, :].permute(2, 0, 1) 

            source_dirc = normalize(
                self.source_cam_pos.unsqueeze(0) - pos[visual_hull_mask].unsqueeze(1)
            )
            point_density_mask, point_rgb_mask = mlp(
                raw_rgb, image_feature, source_dirc, dirc[visual_hull_mask],
            )

            point_density[visual_hull_mask] = point_density_mask[..., 0]
            point_rgb[visual_hull_mask] = point_rgb_mask

        return point_density, point_rgb

