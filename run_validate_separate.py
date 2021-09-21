import torch, cv2, os
import numpy as np

import arg, models
from learning.validation import compute_metric
from learning.visualization import visualize_image
from data import PairDataset, PortraitDataset


# Data Loader
mode = os.sys.argv[2]
if mode == 'view':
    light_size = None
else:
    light_size = arg.model_args['light_size']

dataset = PairDataset(
    PortraitDataset(
        f'{arg.base_path}/data/blender_{mode}', arg.val_data_names,
        ['source_image', 'target_image'], arg.light_ext
    ), shuffle=False, light_size=light_size, rotate_ratio=arg.rotate_ratio
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
model.module.load_state_dict(torch.load(f'{arg.ckpt_path}/{arg.train_step}.pth')['model_state_dict'])


# Run validation
val_path = f'{arg.val_path}_{mode}'
os.makedirs(val_path, exist_ok=True)
for val_num, data in enumerate(data_loader):
    if mode == 'view':
        data['target_light'] = torch.Tensor([0.0])
    if mode == 'relight' and os.sys.argv[1] == 'sipr':
        data['source_shape'] = data['source_shape'][0]
        data['source_image'] = data['source_image'][0]
        data['source_mask'] = data['source_mask'][0]

    data_cuda = {k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}
    output = model.module.render(False, model, **data_cuda)

    if mode == 'view':
        del output['light']


    image_name = f'{data["data_name"]}_{data["source_image_id"]}_{data["target_image_id"]}'
    cv2.imwrite(
        f'{val_path}/{image_name}.jpg',
        visualize_image(output, **data_cuda)
    )


# Validation
psnr, ssim = compute_metric(val_path)
print(arg.exp_name)
print(f'PSNR: {psnr:.4f}')
print(f'SSIM: {ssim:.6f}')