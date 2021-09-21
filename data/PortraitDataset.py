import numpy as np
import os, cv2


class PortraitDataset:
    def __init__(
        self, data_path, data_names=None,
        data_categories=['source_image', 'target_image', 'target_image_rotation'],
        light_ext=''
    ):
        self.data_path = data_path
        if data_names is None:
            self.data_names = sorted([
                f for f in os.listdir(data_path)
                  if os.path.isdir(f'{data_path}/{f}')
            ])
        else:
            self.data_names = sorted(data_names)
        self.data_categories = data_categories
        self.light_ext = light_ext

        self.data_params = {}
        for data_name in self.data_names:
            self.data_params[data_name] = {}
            for data_category in self.data_categories:
                param = np.load(f'{data_path}/{data_name}/{data_category}/cameras.npz')
                self.data_params[data_name][data_category] = {
                    'shapes': param['shapes'],
                    'intrinsics': param['intrinsics'],
                    'extrinsics': param['extrinsics'],
                }


    def __len__(self):
        return len(self.data_names)


    def get(self, property, data_id, data_category, image_id=None):
        data_name = self.data_names[data_id]
        multiview = self.data_params[data_name][data_category]['shapes'].ndim > 2

        if property == 'image_num':
            return self.data_params[data_name][data_category]['shapes'].shape[0]

        elif property in ['shape', 'intrinsic', 'extrinsic']:
            return self.data_params[data_name][data_category][f'{property}s'][image_id]

        elif property in ['image', 'mask', 'depth']:
            ext = {'image': 'jpg', 'mask': 'jpg', 'depth': 'exr'}
            mode = {'image': cv2.IMREAD_COLOR, 'mask': cv2.IMREAD_GRAYSCALE, 'depth': cv2.IMREAD_UNCHANGED}

            if data_category == 'source_image' and multiview:
                return np.stack([
                    cv2.imread(
                        f'{self.data_path}/{data_name}/{data_category}/{image_id:03d}_{view_id:03d}_{property}.{ext[property]}',
                        mode[property]
                    ) 
                    for view_id in range(self.data_params[data_name][data_category]['shapes'].shape[1])
                ], axis=0)
            else:
                return cv2.imread(
                    f'{self.data_path}/{data_name}/{data_category}/{image_id:03d}_{property}.{ext[property]}',
                    mode[property]
                )

        elif property == 'light':
            if data_category == 'target_image_rotation':
                with open(f'{self.data_path}/{data_name}/{data_category}/{image_id:03d}_{property}.txt') as f:
                    return float(f.read())
            elif data_category == 'source_image' and multiview:
                return cv2.imread(
                    f'{self.data_path}/{data_name}/{data_category}/{image_id:03d}_000_{property}.hdr',
                    cv2.IMREAD_UNCHANGED
                )
            else:
                return cv2.imread(
                    f'{self.data_path}/{data_name}/{data_category}/{image_id:03d}_{property}{self.light_ext}.hdr',
                    cv2.IMREAD_UNCHANGED
                )
        else:
            raise NotImplementedError

