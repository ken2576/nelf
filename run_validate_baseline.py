import torch, cv2, os
import numpy as np

import arg
from learning.validation import compute_metric
from learning.visualization import visualize_image
from models import IBR, SIPR
from data import PairDataset, PortraitDataset


# Data Loader
light_size = arg.sipr_args['light_size']
dataset = PairDataset(
    PortraitDataset(arg.data_path, arg.val_data_names, arg.data_categories, arg.light_ext),
    shuffle=False, light_size=light_size, rotate_ratio=arg.rotate_ratio
)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=None,
    num_workers=2, prefetch_factor=1,
    pin_memory=True
)


# Load model
ibr = torch.nn.DataParallel(IBR(**arg.ibr_args)).cuda()
ibr.module.load_state_dict(torch.load(f'{arg.ibr_ckpt_path}/{arg.ibr_step}.pth')['model_state_dict'])

sipr = torch.nn.DataParallel(SIPR(**arg.sipr_args)).cuda()
sipr.module.load_state_dict(torch.load(f'{arg.sipr_ckpt_path}/{arg.sipr_step}.pth')['model_state_dict'])


# Precomputation on light
thetav, phiv = torch.meshgrid(
    torch.arange(light_size[0]).cuda(),
    torch.arange(light_size[1]).cuda()
)
thetav = (thetav.float() + 0.5) / light_size[0] * np.pi
phiv = (phiv.float() + 0.5) / light_size[1] * 2 * np.pi
light_coord = torch.stack([
    -torch.sin(thetav) * torch.sin(phiv),
    -torch.cos(thetav),
    torch.sin(thetav) * torch.cos(phiv)
], axis=-1) # Align with camera coordinate

def rot_light(light, source_extrinsic, target_extrinsic):
    light_coord_rotate = torch.matmul(
        torch.matmul(source_extrinsic[:, :3], torch.t(target_extrinsic[:, :3])).unsqueeze(0).unsqueeze(0),
        light_coord.unsqueeze(-1)
    ).squeeze()
    light_coord_sample = torch.stack([
        torch.atan2(light_coord_rotate[..., 0], -light_coord_rotate[..., 2]) / np.pi,
        2 * torch.acos(-light_coord_rotate[..., 1]) / np.pi - 1
    ], axis=-1) # Range: [-1, 1]
    # print(light_coord_sample)

    light_rot = torch.nn.functional.grid_sample(
        light.permute(2, 0, 1).unsqueeze(0),
        light_coord_sample.unsqueeze(0),
        align_corners=False
    )
    return light_rot[0].permute(1, 2, 0)



# Run validation
relighting_first = int(os.sys.argv[2]) == 1
for val_num, data in enumerate(data_loader):
    data_cuda = {k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}
    if relighting_first:
        for i in range(data_cuda['source_image'].shape[0]):
            relight_cuda = {
                'source_image': data_cuda['source_image'][i],
                'source_mask': data_cuda['source_mask'][i],
                'target_light': rot_light(data_cuda['target_light'], data_cuda['source_extrinsic'][0], data_cuda['source_extrinsic'][i])
            }
            relight = sipr.module.render(False, sipr, **relight_cuda)
            data_cuda['source_image'][i] = relight['rgb']
        output = ibr.module.render(False, ibr, **data_cuda)
    else:
        view_syn = ibr.module.render(False, ibr, **data_cuda)
        relight_cuda = {
            'source_image': view_syn['rgb'],
            'source_mask': view_syn['alpha'] > 0,
            'target_light': rot_light(data_cuda['target_light'], data_cuda['source_extrinsic'][0], data_cuda['target_extrinsic'])
        }
        output = sipr.module.render(False, sipr, **relight_cuda)

    image_name = f'{data["data_name"]}_{data["source_image_id"]}_{data["target_image_id"]}'
    cv2.imwrite(
        f'{arg.val_path}/{image_name}.jpg',
        visualize_image(output, **data_cuda)
    )


# Validation
psnr, ssim = compute_metric(arg.val_path)
print(arg.exp_name)
print(f'PSNR: {psnr:.4f}')
print(f'SSIM: {ssim:.6f}')