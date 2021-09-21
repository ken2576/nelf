import torch, cv2, os
import numpy as np

import arg, models
from test import tester

print(os.sys.argv)

# Model
model = torch.nn.DataParallel(
    models.get_model(arg.model_name)(**arg.model_args)
).cuda()
ckpt = torch.load(f'{arg.ckpt_path}/{int(os.sys.argv[3])}.pth')
model.module.load_state_dict(ckpt['model_state_dict'])


# Light
dataset_color = np.load(f'{arg.base_path}/data/dataset_color.npy')
light_size = (8, 16)


##### Test #####
if os.sys.argv[2].startswith('validate'):
    light = cv2.imread(f'{arg.base_path}/data_test/{os.sys.argv[2]}/source_image_masked/target_light.hdr', cv2.IMREAD_UNCHANGED)
    light = cv2.resize(
        cv2.cvtColor(light, cv2.COLOR_BGR2RGB), tuple(light_size[::-1]), interpolation=cv2.INTER_AREA
    )
    cam_poses = [(15, 22.5*i) for i in range(16)] + [(15, 0) for i in range(32)]
    lights = [0 for i in range(16)] + [15-i for i in range(16)] + [np.roll(light, i, axis=1) for i in range(16)]
    filenames = [f'{i}' for i in range(48)]
    tester(
        data_name=os.sys.argv[2], image_ids=[0, 1, 2, 3, 4],
        dataset_color=None, extra_color_scale=1.0,
        cam_poses=cam_poses,
        lights=lights,
        filenames=filenames,
        model=model, test_path=f'{arg.base_path}/data_test', folder='masked'
    )