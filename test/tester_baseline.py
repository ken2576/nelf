import torch, cv2, os
import numpy as np
from utils.color import srgb2linear, linear2srgb

from .new_camera import new_camera
from .visualization import visualize_source_images, visualize_target_image

light_size = (8, 16)

# Precomputation on light
thetav, phiv = torch.meshgrid(
    torch.arange(light_size[0]).cuda(),
    torch.arange(light_size[1]).cuda()
)
thetav = (thetav.float() + 0.5) / light_size[0] * np.pi
phiv = (phiv.float() + 0.5) / light_size[1] * 2 * np.pi
light_coord = torch.stack([
    -torch.sin(thetav) * torch.sin(phiv),
    -torch.cos(thetav),
    torch.sin(thetav) * torch.cos(phiv)
], axis=-1) # Align with camera coordinate

def rot_light(light, source_extrinsic, target_extrinsic):
    light_coord_rotate = torch.matmul(
        torch.matmul(source_extrinsic[:, :3], torch.t(target_extrinsic[:, :3])).unsqueeze(0).unsqueeze(0),
        light_coord.unsqueeze(-1)
    ).squeeze()
    light_coord_sample = torch.stack([
        torch.atan2(light_coord_rotate[..., 0], -light_coord_rotate[..., 2]) / np.pi,
        2 * torch.acos(-light_coord_rotate[..., 1]) / np.pi - 1
    ], axis=-1) # Range: [-1, 1]
    # print(light_coord_sample)

    light_rot = torch.nn.functional.grid_sample(
        light.permute(2, 0, 1).unsqueeze(0),
        light_coord_sample.unsqueeze(0),
        align_corners=False
    )
    return light_rot[0].permute(1, 2, 0)



def tester_baseline(
    data_name, image_ids,
    dataset_color, extra_color_scale,
    cam_poses, lights, filenames,
    sipr, ibr, relight_first, test_path, folder='masked'
):
    # Prepare data
    os.makedirs(f'{test_path}/{data_name}/results', exist_ok=True)
    data = {}

    masks = np.stack([
        cv2.imread(f'{test_path}/{data_name}/source_image_{folder}/{image_id:03d}_mask.png', 0) / 255.0
        for image_id in image_ids
    ], axis=0)
    ims = np.stack([
        srgb2linear(cv2.imread(
            f'{test_path}/{data_name}/source_image_{folder}/{image_id:03d}_image.png'
        )[:, :, ::-1] / 255.0)
        for i, image_id in enumerate(image_ids)
    ], axis=0) * masks[:, :, :, np.newaxis]

    if dataset_color is not None:
        color_scale = dataset_color / np.mean(ims[0][masks[0] > 0], axis=0) * extra_color_scale
    else:
        color_scale = np.ones((3,))

    data['source_shape'] = torch.IntTensor(np.stack([masks[i].shape for i in range(len(image_ids))], axis=0))
    data['source_image'] = torch.Tensor(ims * color_scale)
    data['source_mask'] = torch.Tensor(masks)

    cameras = np.load(f'{test_path}/{data_name}/source_image_{folder}/cameras.npz')
    intrinsics = cameras['intrinsics'][image_ids]
    extrinsics = cameras['extrinsics'][image_ids, :3, :4]

    data['source_intrinsic'] = torch.Tensor(intrinsics)
    data['source_extrinsic'] = torch.Tensor(extrinsics)

    data['target_shape'] = torch.IntTensor(masks[0].shape)
    data['target_intrinsic'] = data['source_intrinsic'][0]

    depth = np.sqrt(np.sum(extrinsics[0, :, :3].T.dot(extrinsics[0, :, 3]) ** 2))
    for cam_pos, light, filename in zip(cam_poses, lights, filenames):
        data['target_extrinsic'] = torch.Tensor(new_camera(
            depth, cam_pos[0] / 180 * np.pi, cam_pos[1] / 180 * np.pi
        ))
        if isinstance(light, int):
            data['target_light'] = torch.Tensor([light])
        else:
            data['target_light'] = torch.Tensor(light * color_scale)

        # Run Model
        data_cuda = {k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}
        if relight_first:
            for i in range(data_cuda['source_image'].shape[0]):
                if data_cuda['target_light'].ndim < 2:
                    relight_cuda = {
                        'source_image': data_cuda['source_image'][i],
                        'source_mask': data_cuda['source_mask'][i],
                        'target_light': data_cuda['target_light']
                    }
                else:
                    relight_cuda = {
                        'source_image': data_cuda['source_image'][i],
                        'source_mask': data_cuda['source_mask'][i],
                        'target_light': rot_light(data_cuda['target_light'], data_cuda['source_extrinsic'][0], data_cuda['source_extrinsic'][i])
                    }
                relight = sipr.module.render(False, sipr, **relight_cuda)
                if i == 0:
                    pred_light = relight['light']
                data_cuda['source_image'][i] = relight['rgb']
            output = ibr.module.render(False, ibr, **data_cuda)
            output['light'] = pred_light
        else:
            view_syn = ibr.module.render(False, ibr, **data_cuda)
            if data_cuda['target_light'].ndim < 2:
                relight_cuda = {
                    'source_image': view_syn['rgb'],
                    'source_mask': view_syn['alpha'] > 0,
                    'target_light': data_cuda['target_light']
                }
            else:
                relight_cuda = {
                    'source_image': view_syn['rgb'],
                    'source_mask': view_syn['alpha'] > 0,
                    'target_light': rot_light(data_cuda['target_light'], data_cuda['source_extrinsic'][0], data_cuda['target_extrinsic'])
                }
            output = sipr.module.render(False, sipr, **relight_cuda)
            output['light'] = rot_light(output['light'], data_cuda['target_extrinsic'], data_cuda['source_extrinsic'][0])

        if isinstance(light, int):
            target_light = np.roll(output['light'].cpu().numpy() / color_scale, -light, axis=1)
        else:
            target_light = data['target_light'].cpu().numpy() / color_scale
        vis = (visualize_target_image(
            output['rgb'].cpu().numpy() / color_scale, target_light
        ))

        cv2.imwrite(
            f'{test_path}/{data_name}/results/{filename}.png',
            linear2srgb(vis[:, :, ::-1]) * 255
        )

    if os.path.exists(f'{test_path}/{data_name}/source_image_{folder}/source_light.hdr'):
        source_light = cv2.imread(f'{test_path}/{data_name}/source_image_{folder}/source_light.hdr', cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    else:
        source_light = None
    vis_source_image = visualize_source_images(
        data['source_image'].cpu().numpy() / color_scale,
        source_light, 
        output['light'].cpu().numpy() / color_scale
    )
    cv2.imwrite(f'{test_path}/{data_name}/results/_input.png', linear2srgb(vis_source_image[:, :, ::-1]) * 255)

