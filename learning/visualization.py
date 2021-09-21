import torch, cv2
import numpy as np
from utils.color import linear2srgb
import matplotlib.pyplot as plt

###################################
### Images <===> Visualizations ###
###################################
def visualize_image(output, **data):
    source_image = data['source_image'].cpu().numpy()
    if source_image.ndim == 3:
        source_image = source_image[np.newaxis]

    shape = source_image.shape
    source_image_vis = np.zeros((shape[1], 2*shape[2], 3))
    source_image_vis[:,shape[2]//2:3*shape[2]//2] = source_image[0]
    for i in range(shape[0]-1):
        source_image_vis[
            (i//2)*shape[1]//2:(i//2+1)*shape[1]//2,
            (i%2)*3*shape[2]//2:(i%2)*3*shape[2]//2+shape[2]//2,
        ] = cv2.resize(source_image[i+1], None, fx=0.5, fy=0.5)


    target_image = data['target_image'].cpu().numpy()
    target_image_pred = output['rgb'].cpu().numpy()

    vis_image = np.concatenate([
        linear2srgb(source_image_vis),
        linear2srgb(target_image),
        linear2srgb(target_image_pred),
    ], axis=1)

    if 'alpha' in output:
        alpha = output['alpha'].cpu().numpy()
        alpha_gt = data['target_mask'].cpu().numpy()
        alpha_vis = np.stack([alpha, alpha*alpha_gt, alpha_gt], axis=-1)
        vis_image = np.concatenate([vis_image, alpha_vis], axis=1)

    if 'depth' in output:
        depth = output['depth'].cpu().numpy()
        depth_gt = data['target_depth'].cpu().numpy()
        depth_diff = ((alpha * alpha_gt) > 0) * (depth - depth_gt) / 200
        depth_vis = np.stack([depth_diff.clip(0), np.zeros_like(depth_diff), (-depth_diff).clip(0)], axis=-1)
        vis_image = np.concatenate([vis_image, depth_vis], axis=1)

    if 'light' in output:
        source_light_pred = output['light'].cpu().numpy()
        target_light = data['target_light'].cpu().numpy()
        if target_light.ndim <= 2:
            target_light = np.roll(source_light_pred, -int(target_light[0]), axis=1)
        vis_light_size = (target_image.shape[1] // 8, target_image.shape[1] // 4)
        vis_light = linear2srgb(np.concatenate([
            cv2.resize(data['source_light'].cpu().numpy(), vis_light_size[::-1]),
            np.zeros((vis_light_size[0], vis_light_size[1] * 6, 3)),
            cv2.resize(source_light_pred, vis_light_size[::-1]),
            cv2.resize(target_light, vis_light_size[::-1]),
            np.zeros((vis_light_size[0], vis_light_size[1] * 3, 3)),
            cv2.resize(target_light, vis_light_size[::-1]),
            np.zeros((vis_light_size[0], vis_light_size[1] * 3, 3)),
            np.zeros((vis_light_size[0], vis_image.shape[1] - 4 * target_image.shape[1], 3))
        ], axis=1))
        vis_image = np.concatenate([vis_image, vis_light], axis=0)
    
    return cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR) * 255


def extract_image(image):
    image_size = 512
    target_image = image[:image_size, 2*image_size:3*image_size]
    target_image_pred = image[:image_size, 3*image_size:4*image_size]

    return target_image_pred, target_image


def visualize_sample_images(
    model, dataset, vis_path, image_name,
    data_id, source_camera_ids, target_camera_ids
):
    results = []
    for source_camera_id, target_camera_id in zip(source_camera_ids, target_camera_ids):
        source_data = dataset.get_data(data_id, 'source_image', source_camera_id)
        source_data = {f'source_{k}': v.cuda() for k, v in source_data.items() if isinstance(v, torch.Tensor)}

        target_data = dataset.get_data(data_id, 'target_image', target_camera_id)
        target_data = {f'target_{k}': v.cuda() for k, v in target_data.items() if isinstance(v, torch.Tensor)}
        
        output = model.module.render(False, model, **source_data, **target_data)
        vis_image = visualize_image(output, **source_data, **target_data)
        results.append(vis_image)

    cv2.imwrite(f'{vis_path}/{image_name}.jpg', np.concatenate(results, axis=0))


#######################
### Loss <===> Plot ###
#######################
def smoothing(loss, weight=0.6):
    s = np.array(loss)
    for i in range(1, s.shape[0]):
        s[i] = s[i-1] * weight + (1 - weight) * s[i]
    return s


def add_subplot(fig, subplot_id, name, x, y, smooth=False):
    ax = fig.add_subplot(*subplot_id)
    if smooth:
        loss = smoothing(y)
        ax.plot(x, loss)
        ax.plot(x, y, alpha = 0.5)
    else:
        loss = np.array(y)
        ax.plot(x, loss)

    if loss.shape[0] > 0:
        y_max = loss[-(1 + loss.shape[0] // 2):].max() * 1.5
        ax.set_ylim([0, y_max])
    ax.title.set_text(name)


def visualize_loss(
    vis_path, step, 
    train_loss, prompt_interval, val_loss, val_interval
):
    train_x = range(prompt_interval, step+1, prompt_interval)
    val_x = range(val_interval, step+1, val_interval)
    fig = plt.figure(figsize=(16, 8))
    add_subplot(fig, (2, 4, 1), 'train_loss', train_x, train_loss['total'], True)
    add_subplot(fig, (2, 4, 2), 'val_loss', val_x, val_loss, False)
    add_subplot(fig, (2, 4, 3), 'train_rgb_loss', train_x, train_loss['rgb_loss'], True)
    if 'depth_loss' in train_loss:
        add_subplot(fig, (2, 4, 5), 'train_depth_loss', train_x, train_loss['depth_loss'], True)
    if 'alpha_loss' in train_loss:
        add_subplot(fig, (2, 4, 6), 'train_alpha_loss', train_x, train_loss['alpha_loss'], True)
    if 'light_loss' in train_loss:
        add_subplot(fig, (2, 4, 7), 'train_light_loss', train_x, train_loss['light_loss'], True)
    if 'celeba_loss' in train_loss:
        add_subplot(fig, (2, 4, 4), 'train_celeba_loss', train_x, train_loss['celeba_loss'], True)

    plt.savefig(f'{vis_path}/_loss.png')
    plt.close(fig)