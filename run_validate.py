import torch, cv2, os
import numpy as np

import arg, models
from learning.validation import compute_metric
from learning.visualization import visualize_image
from data import PairDataset, PortraitDataset


# Data Loader
light_size = arg.model_args['light_size']
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
model = torch.nn.DataParallel(
    models.get_model(arg.model_name)(**arg.model_args)
).cuda()
model.module.load_state_dict(torch.load(f'{arg.ckpt_path}/{350000}.pth')['model_state_dict'])


# Run validation
for val_num, data in enumerate(data_loader):
    data_cuda = {k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}
    output = model.module.render(False, model, **data_cuda)
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