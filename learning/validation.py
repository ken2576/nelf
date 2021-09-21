import os, cv2, torch
from .visualization import visualize_image, extract_image
from .image_metric import PSNR, SSIM


def validate(model, data_loader, val_path, write_image=False):
    val_loss_sum = 0.0
    for val_num, data in enumerate(data_loader):
        data_cuda = {k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}
        output = model.module.render(False, model, **data_cuda)
        val_loss = model.module.val_loss(output, **data_cuda)
        val_loss_sum += sum([v for _, v in val_loss.items()])

        if write_image:
            image_name = f'{data["data_name"]}_{data["source_image_id"]}_{data["target_image_id"]}'
            cv2.imwrite(
                f'{val_path}/{image_name}.jpg',
                visualize_image(output, **data_cuda)
            )

    val_loss_sum /= (val_num + 1)

    return val_loss_sum


def compute_metric(val_path):
    image_list = [f for f in os.listdir(val_path) if f.endswith('.jpg')]

    psnrs, ssims = 0.0, 0.0
    psnr_calc, ssim_calc = PSNR(), SSIM()
    for image_name in image_list:
        image = cv2.imread(f'{val_path}/{image_name}')
        image_pred, image_gt = extract_image(image)
        psnrs += psnr_calc.compute(image_pred, image_gt)
        ssims += ssim_calc.compute(image_pred, image_gt)
    return psnrs / len(image_list), ssims / len(image_list)