import numpy as np
import torch
import os.path as osp
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from monai.metrics import DiceMetric
from dipy.io.image import load_nifti, save_nifti

def calculating_val_metrics_onesubject_gfa(images, target, mask_path, save_image=False, save_path=''):
    mask, affine = load_nifti(mask_path)
    metrics_gt = target[mask>0, :]
    metrics_prediction = images[mask>0, :]

    nrmse_value = nrmse(metrics_gt, metrics_prediction)
    psnr_value = psnr(metrics_gt, metrics_prediction, data_range=1)
    ssim_value = ssim(metrics_gt, metrics_prediction, data_range=1, channel_axis=1)

    if save_image:
        save_nifti(osp.join(save_path, "pre_" + osp.basename(mask_path)), images * mask[..., np.newaxis].astype(np.float32), affine)
        print("success save image")
        save_nifti(osp.join(save_path, "gt_" + osp.basename(mask_path)), target * mask[..., np.newaxis].astype(np.float32), affine)
        print("success save target")
    return images, target, nrmse_value, psnr_value, ssim_value

def calculating_val_metrics_onesubject_fodf_optimized(images, target, mask_path, sh_order=8, save_image=False, save_path=''):
    mask, affine = load_nifti(mask_path)
    mask = (mask > 0).astype(np.uint8)

    totalNumber = np.sum(mask)

    if save_image:
        save_nifti(osp.join(save_path, "pre_" + osp.basename(mask_path)), images * mask[..., np.newaxis].astype(np.float32), affine)
        print("success save image")
        save_nifti(osp.join(save_path, "gt_" + osp.basename(mask_path)), target * mask[..., np.newaxis].astype(np.float32), affine)
        print("success save target")

    assert images.shape == target.shape, "Input and target shapes do not match."
    assert sh_order % 2 == 0 and sh_order > 0, "sh_order must be a positive even number."
    c_number = ((sh_order + 1) * (sh_order + 2)) // 2
    assert images.shape[-1] == c_number, f"Expected {c_number} coefficients, got {images.shape[-1]}."

    coeffs1 = images[..., 1:]
    coeffs2 = target[..., 1:]

    # 向量化计算分子和分母
    deno1 = np.linalg.norm(coeffs1, axis=-1)
    deno2 = np.linalg.norm(coeffs2, axis=-1)
    deno = deno1 * deno2

    numerator = np.sum(coeffs1 * coeffs2, axis=-1)

    valid_mask = (mask > 0) & (deno != 0)
    badPoint = np.sum((mask > 0) & (deno == 0))

    acc = np.zeros_like(mask, dtype=np.float64)
    acc[valid_mask] = numerator[valid_mask] / deno[valid_mask]

    total_valid = totalNumber - badPoint
    avg_acc_xq = np.sum(acc) / total_valid if total_valid > 0 else 0.0

    return avg_acc_xq

def calculating_val_tract_metrics_onesubject(images, target, mask_path, save_image=False, save_path=''):
    mask, affine = load_nifti(mask_path)
    mask = mask.astype(np.bool_)

    prediction_binary = (images > 0.5).astype(np.int_)

    mask = mask[..., np.newaxis]
    prediction_binary = prediction_binary * mask
    target = target * mask

    if save_image:
        save_nifti(osp.join(save_path, "pre_" + osp.basename(mask_path)), prediction_binary.astype(np.float64), affine)
        print("success save image")
        save_nifti(osp.join(save_path, "gt_" + osp.basename(mask_path)), target.astype(np.float64), affine)
        print("success save target")

    prediction_binary = prediction_binary.astype(np.float32)
    target = target.astype(np.float32)
    pred_tensor = torch.from_numpy(prediction_binary).permute(3, 0, 1, 2).unsqueeze(0)
    target_tensor = torch.from_numpy(target).permute(3, 0, 1, 2).unsqueeze(0)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric(pred_tensor, target_tensor)
    dice_score = dice_metric.aggregate().item()
    return dice_score