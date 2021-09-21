import numpy as np


def srgb2linear(srgb):
    linear = np.float32(srgb)
    return np.where(
        linear <= 0.04045,
        linear / 12.92,
        np.power((linear + 0.055) / 1.055, 2.4)
    )


def linear2srgb(linear):
    srgb = np.float32(linear)
    return np.where(
        srgb <= 0.0031308,
        srgb * 12.92,
        1.055 * np.power(srgb, 1.0 / 2.4) - 0.055
    )