import os

# Experiment
relighting_first = int(os.sys.argv[2]) == 1
print(relighting_first)
if relighting_first:
    exp_name = 'validate_sipr_ibr'
else:
    exp_name = 'validate_ibr_sipr'
exp_path = f'data_results/{exp_name}'

val_path = f'{exp_path}/validation'
ibr_ckpt_path = 'data_results/ibr/ckpt'
sipr_ckpt_path = 'data_results/sipr/ckpt'


# Model
ibr_args = {
    'coarse_sample_num': 64,
    'fine_sample_num': 128,
    'train_ray_per_gpu': 512,
    'test_ray_per_gpu': 512,

    'train_mask_ratio': 0.75,
    'head_diameter': 200, # mm
}
sipr_args = {
    'light_size': (8, 16),
}


# Dataset
data_path = 'data/blender_both'
val_data_names = ['sub122', 'sub212', 'sub340', 'sub344']
data_categories = ['source_image', 'target_image']
light_ext = ''

rotate_ratio = 0.0


# Training 
ibr_step = 300000
sipr_step = 1000000
