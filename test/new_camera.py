import numpy as np

def normalize(vec):
    return vec / np.sqrt(np.sum(vec**2))

def new_camera(depth, cone_theta, phi, up=np.array([0, 0, 1])):
    camera_pos = np.array([
        depth * np.sin(cone_theta) * np.cos(phi),
        -depth * np.cos(cone_theta),
        depth * np.sin(cone_theta) * np.sin(phi)
    ])
    front = normalize(-camera_pos)
    right = normalize(np.cross(front, up))
    up = np.cross(right, front)
    R = np.stack([right, -up, front], axis=0)
    t = -R.dot(camera_pos)
    return np.concatenate([R, t[:, np.newaxis]], axis=1)