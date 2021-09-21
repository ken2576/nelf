import numpy as np
import os, cv2

from utils.color import srgb2linear


def compute_cdf(pixels, bins=1024):
    cdfs, inv_cdfs = [], []
    for i in range(3):
        hist, bin_pos = np.histogram(pixels, bins, [0, 1])
        cdf = np.concatenate([[0], hist]).cumsum()
        cdf = cdf / float(cdf[-1])
        inv_cdf = np.interp(bin_pos, cdf, bin_pos)
        cdfs.append(cdf)
        inv_cdfs.append(inv_cdf)

    return bin_pos, np.stack(cdfs, axis=0), np.stack(inv_cdfs, axis=0)


def hist_match(im, bin_pos, source_cdf, target_inv_cdf):
    return np.stack([
        np.interp(np.interp(im[:, :, i], bin_pos, source_cdf[i]), bin_pos, target_inv_cdf[i])
        for i in range(3)
    ], axis=-1)
        


def compute_dataset_cdf(dataset_path, image_num=30, shape=(256, 256)):
    pixels = []
    data_names = os.listdir(dataset_path)
    for data_name in data_names:
        for i in range(image_num):
            im = srgb2linear(cv2.resize(
                cv2.cvtColor(cv2.imread(f'{dataset_path}/{data_name}/source_image/{i:03d}_image.jpg'), cv2.COLOR_BGR2RGB) / 255.0,
                shape, interpolation=cv2.INTER_AREA
            ))
            hat_mask = (im[:, :, 0] > 0.4) * (im[:, :, 1] < 0.2) * (im[:, :, 2] < 0.2)
            mask = (1 - hat_mask) * cv2.resize(
                cv2.imread(f'{dataset_path}/{data_name}/source_image/{i:03d}_mask.jpg', 0) / 255.0,
                shape, interpolation=cv2.INTER_AREA
            ) > 0
            pixels.append(im[mask])
    pixels = np.concatenate(pixels, axis=0)
    return compute_cdf(pixels)

