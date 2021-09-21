import torch, cv2, os
import numpy as np

import arg
from models import IBR, SIPR
from test import tester_baseline


# Model
ibr = torch.nn.DataParallel(IBR(**arg.ibr_args)).cuda()
ibr.module.load_state_dict(torch.load(f'{arg.ibr_ckpt_path}/{150000}.pth')['model_state_dict'])

sipr = torch.nn.DataParallel(SIPR(**arg.sipr_args)).cuda()
sipr.module.load_state_dict(torch.load(f'{arg.sipr_ckpt_path}/{arg.sipr_step}.pth')['model_state_dict'])


# Light
dataset_color = np.load(f'{arg.base_path}/data/dataset_color.npy')
light_size = (8, 16)

filename = 'sipr_ibr' if int(os.sys.argv[2]) else 'ibr_sipr'

##### Test #####

if os.sys.argv[3].startswith('validate'):
    light = cv2.imread(f'{arg.base_path}/data_test/{os.sys.argv[3]}/source_image_masked/target_light.hdr', cv2.IMREAD_UNCHANGED)
    light = cv2.resize(
        cv2.cvtColor(light, cv2.COLOR_BGR2RGB), tuple(light_size[::-1]), interpolation=cv2.INTER_AREA
    )
    cam_poses = [(15, 22.5*i) for i in range(16)] + [(15, 0) for i in range(32)]
    lights = [0 for i in range(16)] + [15-i for i in range(16)] + [np.roll(light, i, axis=1) for i in range(16)]
    filenames = [f'{filename}_{i}' for i in range(48)]
    tester_baseline(
        data_name=os.sys.argv[3], image_ids=[0, 1, 2, 3, 4],
        dataset_color=None, extra_color_scale=1.0,
        cam_poses=cam_poses,
        lights=lights,
        filenames=filenames,
        sipr=sipr, ibr=ibr, relight_first=int(os.sys.argv[2]), test_path=f'{arg.base_path}/data_test', folder='masked'
    )