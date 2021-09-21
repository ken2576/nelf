import torch


def get_frontback_pix(mask):
    '''
    Get all the pixel coordinates of front and back given the mask.
    The two outputs both have the shape (N, 2), and is ordered as (x, y)
    '''
    return (
        torch.nonzero(mask).float()[:, [1, 0]],
        torch.nonzero(1 - mask).float()[:, [1, 0]]
    )


def get_image_pix(shape):
    '''
    Get all the pixel coordinates given the shape of the image.
    The output has the shape (N, 2), and is ordered as (x, y)
    '''
    return torch.stack(torch.meshgrid(
        torch.arange(shape[0], device=shape.device),
        torch.arange(shape[1], device=shape.device)
    )[::-1], axis=-1).view((-1, 2)).float()



def project_to_cam(intrinsic, extrinsic, point):
    pix = torch.matmul(
        torch.cat([point, torch.ones_like(point[..., :1])], -1),
        torch.t(torch.mm(intrinsic, extrinsic))
    )
    pix = pix[..., :2] / pix[..., 2:] 
    return pix


def pix_to_ray(intrinsic, extrinsic, pix, near, far):
    pix_num = pix.shape[0]
    pos = -torch.mv(torch.t(extrinsic[:, :3]), extrinsic[:, 3])
    pix2dirc = torch.mm(torch.t(torch.inverse(intrinsic)), extrinsic[:, :3])
    dirc = torch.matmul(torch.cat([pix, torch.ones_like(pix[:, :1])], -1), pix2dirc)
    dirc = dirc / torch.norm(dirc, p=2, dim=-1, keepdim=True)
    return torch.cat([
        pos.repeat(pix_num, 1),
        dirc,
        pix.new_full((pix_num, 1), near), 
        pix.new_full((pix_num, 1), far)], dim=-1)

