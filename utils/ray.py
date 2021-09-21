import torch

def march_ray(ray, depth):
    ray_num = ray.shape[0]
    sample_num = depth.shape[-1]

    ray_expand = ray.unsqueeze(1)
    return (
        ray_expand[..., :3] + depth.unsqueeze(-1) * ray_expand[..., 3:6],
        ray_expand[..., 3:6].expand(ray_num, sample_num, 3)
    )


def sample_ray_depth(ray, sample_num, weight=None, depth=None, depth_std=None):
    device = ray.device
    ray_num = ray.shape[0]
    near, far = ray[:, -2:-1], ray[:, -1:] 

    if weight is None and depth is None:
        step = 1.0 / sample_num
        z_steps = torch.linspace(0, 1 - step, sample_num, device=device) 
        z_steps = z_steps.repeat(ray_num, 1)   # (ray_num, sample_num)
        z_steps += torch.rand_like(z_steps) * step

        return near * (1 - z_steps) + far * z_steps  # (ray_num, sample_num)

    elif weight is not None:
        bin_num = weight.shape[-1]
        weight = weight.detach() + 1e-5  # Prevent division by zero
        pdf = weight / torch.sum(weight, -1, keepdim=True)  # (ray_num, bin_num)
        cdf = torch.cumsum(pdf, -1)  
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (ray_num, bin_num + 1)

        u = torch.rand(
            ray_num, sample_num, dtype=torch.float32, device=device
        )  # (ray_num, sample_num)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (ray_num, sample_num)
        inds = torch.clamp_min(inds, 0.0)
        z_steps = (inds + torch.rand_like(inds)) / bin_num  # (ray_num, sample_num)

        return near * (1 - z_steps) + far * z_steps  # (ray_num, sample_num)

    elif depth is not None:
        z_samp = depth.unsqueeze(1).repeat(1, sample_num)
        z_samp += torch.randn_like(z_samp) * depth_std # (ray_num, sample_num)
        return torch.max(torch.min(z_samp, far), near) # (ray_num, sample_num)


def composite(depth, density, rgb):
    delta = torch.cat([
        depth[:, 1:2] - depth[:, :1],
        (depth[:, 2:] - depth[:, :-2]) / 2,
        depth[:, -1:] - depth[:, -2:-1]], -1)

    attenuation = delta * density
    alpha = 1 - torch.exp(-attenuation) 
    transmittance = torch.exp(-torch.cumsum(
        torch.cat([torch.zeros_like(attenuation[:, :1]), attenuation], axis=-1),
        axis=-1))
    # transmittance = torch.cumprod(
    #     torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], axis=-1),
    #     axis=-1)
            
    weights = alpha * transmittance[:, :-1]

    return (
        weights,                                                               # Final weights
        torch.sum(weights.unsqueeze(-1) * rgb, axis=-2),                       # Final RGB
        torch.sum(weights * depth, -1) / (1 - transmittance[:, -1] + 1e-6),    # Final depth
        1 - transmittance[:, -1]                                               # Final alpha
    )


def normalize(x, axis=-1):
    return x / torch.norm(x, p=2, dim=axis, keepdim=True)



if __name__ == '__main__':
    
    ray = torch.rand(256, 64).cuda()
    rgb = torch.rand(256, 64, 3).cuda()
    density = torch.zeros(256, 64).cuda()
    a, b, c, d = composite(ray, rgb, density)