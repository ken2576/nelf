
# Experiment
exp_name = 'sipr'
exp_path = f'data_results/{exp_name}'

vis_path = f'{exp_path}/vis'
ckpt_path = f'{exp_path}/ckpt'
val_path = f'{exp_path}/validation'


# Visualization
vis_data_id = 0
vis_source_image_id = [0, 1, 2]
vis_target_image_ids = [0, 1, 2]


# Model
model_name = 'sipr'
model_args = {
    'light_size': (8, 16),
}


# Dataset
data_path = 'data/blender_relight'
bad_data_names = ['sub148']

'''Uncomment to train on full dataset
val_data_names = ['sub122', 'sub212', 'sub340', 'sub344']
train_data_names = [
    f'sub{i:03d}'
    for i in range(1, 360) 
    if f'sub{i:03d}' not in val_data_names and f'sub{i:03d}' not in bad_data_names
]
'''

# Uncomment the following 2 lines to use small dataset as a preview
val_data_names = ['sub212']
train_data_names = ['sub122', 'sub212', 'sub340', 'sub344']


data_categories = ['source_image', 'target_image', 'target_image_rotation']
light_ext = 'cam'

rotate_ratio = 0.2
batch_per_gpu = None


# Training 
mlp_lr = 1e-4
image_encoder_lr = 2e-4
train_step = 1000000
prompt_interval = 1
vis_interval = 50000
val_interval = 100000
ckpt_interval = 1000000


# Pretrain
use_pretrain = False
pretrain_step = 0
