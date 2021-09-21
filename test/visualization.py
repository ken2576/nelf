import torch, cv2
import numpy as np
from utils.color import linear2srgb

def visualize_source_images(
    source_image, source_light, source_light_pred
):
    shape = source_image.shape
    source_image_vis = np.zeros((shape[1]+shape[1]//8, 2*shape[2], 3))
    source_image_vis[:shape[1],shape[2]//2:3*shape[2]//2] = source_image[0]
    for i in range(shape[0]-1):
        source_image_vis[
            (i//2)*shape[1]//2:(i//2+1)*shape[1]//2,
            (i%2)*3*shape[2]//2:(i%2)*3*shape[2]//2+shape[2]//2,
        ] = cv2.resize(source_image[i+1], None, fx=0.5, fy=0.5)

    light_size = (shape[1] // 4, shape[1] // 8)
    if source_light is not None:
        source_image_vis[-shape[1]//8:, :shape[1]//4] = cv2.resize(source_light, light_size)
    if source_light_pred is not None:
        source_image_vis[-shape[1]//8:, shape[2]*2-shape[1]//4:shape[2]*2] = cv2.resize(source_light_pred, light_size)
    
    return source_image_vis


def visualize_target_image(target_image, target_light):
    shape = target_image.shape
    target_image_vis = np.zeros((shape[0]+shape[0]//8, shape[1], 3))
    target_image_vis[:shape[0], :] = target_image

    light_size = (shape[0] // 4, shape[0] // 8)
    if target_light is not None:
        target_image_vis[-shape[0]//8:, shape[1]//2-shape[0]//8:shape[1]//2+shape[0]//8] = cv2.resize(target_light, light_size)
    
    return target_image_vis