import os, cv2, json
import numpy as np
import torch

from utils.color import srgb2linear


# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(matrix_world):
    def decompose(mat):
        loc = mat[:3, -1]
        rot = mat[:3, :3]
        return loc, rot
    # bcam stands for blender camera
    R_bcam2cv = np.array(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = decompose(matrix_world)
    # print(matrix_world)
    R_world2bcam = rotation.T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # print(R_world2bcam)
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv[:, None]], 1)
    return RT


class PairDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, scale=1.0):
        super().__init__()
        self.dataset = dataset

        self.scale = scale
        self.scale_matrix = np.array(
            [[scale, 0,     (scale - 1) / 2], 
             [0,     scale, (scale - 1) / 2],
             [0,     0,     1]])


    def get_data(self, data_id, camera_id):
        data_name = self.dataset.data_names[data_id]
        data_param = self.dataset.data_params[data_name]

        output = {}
        shape = np.rint(self.scale * data_param["shapes"][camera_id, :2]).astype(int)
        output['shape'] = torch.IntTensor(shape)
        output['intrinsic'] = torch.Tensor(
            self.scale_matrix.dot(data_param["intrinsics"][camera_id])
        )
        output['extrinsic'] = torch.Tensor(
            data_param["extrinsics"][camera_id]
        )

        rgb_path = os.path.join(self.dataset.data_path, data_name, 'rgb', f'{camera_id:03d}.exr')
        image = cv2.resize(
            cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED),
            tuple(shape[::-1])
        )[:, :, ::-1] # BGR to RGB

        mask_path = os.path.join(self.dataset.data_path, data_name, 'mask', f'{camera_id:03d}_0001.exr')
        mask = cv2.resize(
            cv2.imread(mask_path, cv2.IMREAD_UNCHANGED),
            tuple(shape[::-1])
        )[:, :, 0]

        # env_path = os.path.join(self.dataset.data_path, data_name, 'envmap', f'{camera_id:03d}.hdr')
        # envmap = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]] # BGR to RGB

        image = image * mask[:, :, np.newaxis]
        output['image'] = torch.Tensor(image)
        output['mask'] = torch.Tensor(mask)
        # output['envmap'] = torch.Tensor(envmap)

        return output


    def __iter__(self):
        while True:
            data_id = torch.randint(len(self.dataset), (1, )).item()
            data_name = self.dataset.data_names[data_id]
            data_param = self.dataset.data_params[data_name]

            output = {'data_name': data_name}
            for prefix in ['source', 'target']:
                camera_id = torch.randint(data_param['shapes'].shape[0], (1, )).item()
                if prefix == 'source':
                    camera_id = 0
                output[f'{prefix}_camera_id'] = camera_id

                camera_data = self.get_data(data_id, camera_id)
                output[f'{prefix}_shape'] = camera_data['shape']
                output[f'{prefix}_intrinsic'] = camera_data['intrinsic']
                output[f'{prefix}_extrinsic'] = camera_data['extrinsic']
                output[f'{prefix}_image'] = camera_data['image']
                output[f'{prefix}_mask'] = camera_data['mask']
                # output[f'{prefix}_envmap'] = camera_data['envmap']

            yield output


class FacescapeBlenderDataset(torch.utils.data.Dataset):
    """
    Dataset from Facescape (H. Yang et al. 2020)
    https://facescape.nju.edu.cn/
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_names = sorted(os.listdir(data_path))
        self.data_params = {}
        for data_name in self.data_names:
            intrinsics = np.load(
                os.path.join(
                    data_path,
                    data_name,
                    'cam_ints.npy'
                )
            )
            extrinsics = np.load(
                os.path.join(
                    data_path,
                    data_name,
                    'cam_exts.npy'
                )
            )
            extrinsics = [get_3x4_RT_matrix_from_blender(x) for x in extrinsics]
            extrinsics = np.array(extrinsics)
            # extrinsics = np.concatenate([extrinsics,
            #     np.zeros_like(extrinsics[..., 0:1, :])], -2)
            # extrinsics[:, 3, 3] = 1

            file_name = os.path.join(
                data_path,
                data_name,
                'commandline_args.txt'
            )
            with open(file_name, 'r') as f:
                data = json.load(f)

            img_shape = np.array(data[0]['img_wh'])[::-1][None, :]
            img_shape = np.tile(img_shape, [len(extrinsics), 1])

            self.data_params[data_name] = {
                'shapes': img_shape,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
            }

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        return self.data_params[self.data_names[index]]

if __name__ == '__main__':
    base_path = '/home/kingo/Documents/projects/PortraitView'

    dataset = PairDataset(FacescapeBlenderDataset(f'{base_path}/data/facescape_render'))
    it = iter(dataset)
    sample = next(it)
    for key in sample:
        print(key, sample[key].shape if hasattr(sample[key], 'shape') else sample[key])