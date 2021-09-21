import torch

import arg, models, learning
from data import PairDataset, PortraitDataset


# Data
light_size = arg.model_args['light_size'] if 'light_size' in arg.model_args else None
batch_size = None if arg.batch_per_gpu is None else arg.batch_per_gpu * torch.cuda.device_count()

train_dataset = PairDataset(
    PortraitDataset(arg.data_path, arg.train_data_names, arg.data_categories, arg.light_ext),
    shuffle=True, light_size=light_size, rotate_ratio=arg.rotate_ratio
)
val_dataset = PairDataset(
    PortraitDataset(arg.data_path, arg.val_data_names, arg.data_categories, arg.light_ext),
    shuffle=False, light_size=light_size, rotate_ratio=arg.rotate_ratio
)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=8, prefetch_factor=1,
    persistent_workers=True, pin_memory=True
)
val_data_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size,
    num_workers=2, prefetch_factor=1,
    pin_memory=True
)

# Model
model = torch.nn.DataParallel(
    models.get_model(arg.model_name)(**arg.model_args)
).cuda()
optimizer = torch.optim.Adam([
    {'params': [param for name, param in model.named_parameters() if 'image_encoder' in name], 'lr': arg.image_encoder_lr},
    {'params': [param for name, param in model.named_parameters() if 'image_encoder' not in name]}
], lr=arg.mlp_lr)


# Training
learning.train(
    model, optimizer, train_data_loader, val_data_loader, val_dataset,
    arg.train_step, arg.prompt_interval,
    arg.vis_interval, arg.val_interval, arg.ckpt_interval,
    arg.vis_path, arg.val_path, arg.ckpt_path,
    arg.vis_data_id, arg.vis_source_image_id, arg.vis_target_image_ids,
    arg.use_pretrain, arg.pretrain_step
)


# Validation
psnr, ssim = learning.compute_metric(arg.val_path)
print(arg.exp_name)
print(f'PSNR: {psnr:.4f}')
print(f'SSIM: {ssim:.6f}')