import torch, cv2, os
import numpy as np
from utils.color import srgb2linear, linear2srgb

from .new_camera import new_camera
from .visualization import visualize_source_images, visualize_target_image


def tester(
    data_name, image_ids,
    dataset_color, extra_color_scale,
    cam_poses, lights, filenames,
    model, test_path, folder='masked'
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
        color_scale = np.ones((3,)) * extra_color_scale

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
        print(filename)
        data['target_extrinsic'] = torch.Tensor(new_camera(
            depth, cam_pos[0] / 180 * np.pi, cam_pos[1] / 180 * np.pi
        ))
        if isinstance(light, int):
            data['target_light'] = torch.Tensor([light])
        else:
            data['target_light'] = torch.Tensor(light * color_scale)

        # Run Model
        output = model.module.render(
            False, model, 
            **{k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}
        )

        if isinstance(light, int):
            target_light = np.roll(output['light'].cpu().numpy() / color_scale, -light, axis=1)
        else:
            target_light = data['target_light'].cpu().numpy() / color_scale
        vis = visualize_target_image(
            output['rgb'].cpu().numpy() / color_scale, target_light
        )

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


