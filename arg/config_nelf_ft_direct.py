
# Experiment
exp_name = 'nelf_ft_direct'
exp_path = f'data_results/{exp_name}'

vis_path = f'{exp_path}/vis'
ckpt_path = f'{exp_path}/ckpt'
val_path = f'{exp_path}/validation'


# Visualization
vis_data_id = 0
vis_source_image_id = [0, 1, 2]
vis_target_image_ids = [0, 1, 2]


# Model
model_name = 'nelf_ft_direct'
model_args = {
    'coarse_sample_num': 64,
    'fine_sample_num': 128,
    'train_ray_per_gpu': 256,
    'test_ray_per_gpu': 640,

    'train_mask_ratio': 0.75,
    'head_diameter': 200, # mm
    'light_size': (8, 16),
}


# Dataset
data_path = 'data/blender_both'
bad_data_names = ['sub148']
val_data_names = ['sub122', 'sub212', 'sub340', 'sub344']
train_data_names = [
    f'sub{i:03d}'
    for i in range(1, 360) 
    if f'sub{i:03d}' not in val_data_names and f'sub{i:03d}' not in bad_data_names
]
# val_data_names = ['sub001']
# train_data_names = [f'sub{i:03d}' for i in range(2, 10) ]
data_categories = ['source_image', 'target_image', 'target_image_rotation']
light_ext = ''

rotate_ratio = 0.3
batch_per_gpu = None



# Training 
mlp_lr = 1e-4
image_encoder_lr = 2e-4
train_step = 500000
prompt_interval = 10
vis_interval = 10000
val_interval = 20000
ckpt_interval = 50000


# Pretrain
use_pretrain = False
pretrain_step = 0
