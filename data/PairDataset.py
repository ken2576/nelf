import numpy as np
import torch, cv2, os
from utils.color import srgb2linear


class PairDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, shuffle=True, 
                 light_size=(8, 16), rotate_ratio=0.2):
        super().__init__()
        self.dataset = dataset

        self.shuffle = shuffle
        self.light_size = light_size
        self.rotate_ratio = rotate_ratio

        # Hack the celeba dataset in
        self.celeba_path = f'{dataset.data_path}/../CelebAMask'
        self.celeba_num = len(os.listdir(self.celeba_path))

    def get_data(self, data_id, data_category, image_id):
        output = {}
        shape = np.rint(
            self.dataset.get('shape', data_id, data_category, image_id)
        ).astype(int)
        output['shape'] = torch.IntTensor(shape)
        output['intrinsic'] = torch.Tensor(
            self.dataset.get('intrinsic', data_id, data_category, image_id)
        )
        output['extrinsic'] = torch.Tensor(
            self.dataset.get('extrinsic', data_id, data_category, image_id)
        )

        depth = self.dataset.get('depth', data_id, data_category, image_id)[..., 0]
        depth[depth > 10000] = 0
        output['depth'] = torch.Tensor(depth)

        image = self.dataset.get('image', data_id, data_category, image_id)[..., ::-1] / 255.0
        mask = self.dataset.get('mask', data_id, data_category, image_id) / 255.0
        output['image'] = torch.Tensor(srgb2linear(image * (depth > 0)[..., np.newaxis]))
        output['mask'] = torch.Tensor(mask * (depth > 0))

        if self.light_size is not None:
            light = self.dataset.get('light', data_id, data_category, image_id)
            if isinstance(light, float):
                output['light'] = torch.Tensor([light])
            else:
                light = cv2.resize(
                    self.dataset.get('light', data_id, data_category, image_id)[..., ::-1],
                    tuple(self.light_size[::-1]), interpolation=cv2.INTER_AREA
                )
                output['light'] = torch.Tensor(light)
        return output


    def __iter__(self):
        if self.shuffle:
            while True:
                data_id = torch.randint(len(self.dataset), (1, )).item()
                output = {'data_name': self.dataset.data_names[data_id]}
                for prefix in ['source', 'target']:
                    if prefix == 'target' and torch.rand(1).item() < self.rotate_ratio:
                        postfix = '_rotation'
                    else:
                        postfix = ''
                        image_id = torch.randint(
                            self.dataset.get('image_num', data_id, f'{prefix}_image'),
                            (1, )
                        ).item()

                    output[f'{prefix}_image_id'] = image_id
                    image_data = self.get_data(data_id, f'{prefix}_image{postfix}', image_id)
                    for property in image_data:
                        output[f'{prefix}_{property}'] = image_data[property]

                celeba_id = torch.randint(self.celeba_num, (1, )).item()
                output['celeba_image'] = torch.Tensor(srgb2linear(
                    cv2.imread(f'{self.celeba_path}/{celeba_id}.jpg')[:, :, ::-1] / 255.0
                ))

                yield output

        else:
            for data_id in range(len(self.dataset)):
                output = {'data_name': self.dataset.data_names[data_id]}
                for image_id in range(self.dataset.get('image_num', data_id, 'source_image')):
                    for prefix in ['source', 'target']:
                        output[f'{prefix}_image_id'] = image_id
                        image_data = self.get_data(data_id, f'{prefix}_image', image_id)
                        for property in image_data:
                            output[f'{prefix}_{property}'] = image_data[property]
                    yield output
