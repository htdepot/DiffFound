import os
import warnings
import datetime
import json
import time
import random
import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from utils import misc
from utils.metrics import calculating_val_metrics_onesubject_gfa
from model.difffound_finetune_microstructrue_estimation import build_difffound
from dataset.finetune_dataset import Dataset_Dmri

def main(args):
    if args.env.seed is not None:
        seed = args.env.seed + misc.get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    args.env.distributed = args.env.world_size > 1 or (args.env.distributed and ngpus_per_node > 1)
    if args.env.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)


def main_worker(local_rank, args):
    misc.init_distributed_mode(local_rank, args)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    job_dir = f"{args.output_dir}/{args.job_name}"
    print(f'job dir: {job_dir}')
    print("{}".format(args).replace(', ', ',\n'))

    num_tasks = misc.get_world_size()
    num_tasks_per_node = max(1, torch.cuda.device_count())
    global_rank = misc.get_rank()
    args.env.workers = args.env.workers // num_tasks_per_node


    cudnn.benchmark = True

    db_eval = Dataset_Dmri(args.test_data_path, args.test_data_gt_path, mask_path=args.mask_data_path)

    if args.env.distributed:
        sampler_eval = torch.utils.data.DistributedSampler(
            db_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_eval = torch.utils.data.SequentialSampler(db_eval)


    data_loader_eval = torch.utils.data.DataLoader(
        db_eval,
        sampler=sampler_eval,
        batch_size=1,
        num_workers=args.env.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_difffound(
        args.encoder,
        loss=args.loss,
        grid_size=args.grid_size,
        tau=args.tau,
        num_vis=args.num_vis,
        avg_sim_coeff=args.avg_sim_coeff,
        drop=args.drop,
        attn_drop=args.attn_drop,
        drop_path=args.drop_path,
        freeze_pe=args.freeze_pe,
        proj_cfg=args.proj,
        mask_target=args.mask_target
    )

    if args.test:
        checkpoint = torch.load(args.test, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.test)
        checkpoint_model = checkpoint['state_dict']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    start_time = time.time()


    test_stats = evaluate(data_loader_eval, model, device, args.save_images, args.output_dir)

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'n_parameters': n_parameters}
    if misc.is_main_process():
        with open(os.path.join(job_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


@torch.no_grad()
def evaluate(data_loader, model, device, save_images, output_dir):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    psnr_sall = []
    ssim_sall = []
    mse_sall = []

    child_psnr_sall = []
    child_ssim_sall = []
    child_mse_sall = []

    hcp_psnr_sall = []
    hcp_ssim_sall = []
    hcp_mse_sall = []

    bids_psnr_sall = []
    bids_ssim_sall = []
    bids_mse_sall = []

    max_psnr_sall = []
    max_ssim_sall = []
    max_mse_sall = []

    ppmi_psnr_sall = []
    ppmi_ssim_sall = []
    ppmi_mse_sall = []

    adni_psnr_sall = []
    adni_ssim_sall = []
    adni_mse_sall = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1][0].unsqueeze(3)
        mask_path = batch[2][0]
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(images)
        out = out[0].cpu().detach().numpy()

        vis_output, vis_target, nrmse_all, psnr_all, ssim_all = (
            calculating_val_metrics_onesubject_gfa(out, target.numpy(), mask_path, save_images, output_dir))

        psnr_sall.append(psnr_all)
        ssim_sall.append(ssim_all)
        mse_sall.append(nrmse_all)

        if os.path.basename(mask_path).split("_")[0] == "child":
            child_psnr_sall.append(psnr_all)
            child_ssim_sall.append(ssim_all)
            child_mse_sall.append(nrmse_all)

        if os.path.basename(mask_path).split("_")[0] == "HCP":
            hcp_psnr_sall.append(psnr_all)
            hcp_ssim_sall.append(ssim_all)
            hcp_mse_sall.append(nrmse_all)

        if os.path.basename(mask_path).split("_")[0] == "bids":
            bids_psnr_sall.append(psnr_all)
            bids_ssim_sall.append(ssim_all)
            bids_mse_sall.append(nrmse_all)

        if os.path.basename(mask_path).split("_")[0] == "max":
            max_psnr_sall.append(psnr_all)
            max_ssim_sall.append(ssim_all)
            max_mse_sall.append(nrmse_all)

        if os.path.basename(mask_path).split("_")[0] == "ppmi":
            ppmi_psnr_sall.append(psnr_all)
            ppmi_ssim_sall.append(ssim_all)
            ppmi_mse_sall.append(nrmse_all)

        if os.path.basename(mask_path).split("_")[0] == "adni":
            adni_psnr_sall.append(psnr_all)
            adni_ssim_sall.append(ssim_all)
            adni_mse_sall.append(nrmse_all)

        metric_logger.meters['psnr'].update(psnr_all.item())
        metric_logger.meters['ssim'].update(ssim_all.item())
        metric_logger.meters['nrmse'].update(nrmse_all.item())

        print('PNSR {psnr:.2f} SSIM {ssim:.3f} NRMSE {nrmse:.3f} '
              .format(psnr=psnr_all, ssim=ssim_all, nrmse=nrmse_all))

    metric_logger.synchronize_between_processes()

    print("**************************************************************************")
    print("Test data")
    print(psnr_sall)
    print(ssim_sall)
    print(mse_sall)
    print(
        'PNSR avg {psnr:.2f} std {psnr_std:.2f} SSIM avg {ssim:.3f} std {ssim_std:.3f} avg NRMSE {nrmse:.3f} std {nrmse_std:.3f}'
        .format(psnr=np.array(psnr_sall).mean(), psnr_std=np.array(psnr_sall).std(), ssim=np.array(ssim_sall).mean(),
                ssim_std=np.array(ssim_sall).std(), nrmse=np.array(mse_sall).mean(),
                nrmse_std=np.array(mse_sall).std()))

    print("**************************************************************************")
    print("CMIHBN")
    print(child_psnr_sall)
    print(child_ssim_sall)
    print(child_mse_sall)
    print(
        'PNSR avg {psnr:.2f} std {psnr_std:.2f} SSIM avg {ssim:.3f} std {ssim_std:.3f} avg NRMSE {nrmse:.3f} std {nrmse_std:.3f}'
        .format(psnr=np.array(child_psnr_sall).mean(), psnr_std=np.array(child_psnr_sall).std(),
                ssim=np.array(child_ssim_sall).mean(),
                ssim_std=np.array(child_ssim_sall).std(), nrmse=np.array(child_mse_sall).mean(),
                nrmse_std=np.array(child_mse_sall).std()))

    print("**************************************************************************")
    print("HCP")
    print(hcp_psnr_sall)
    print(hcp_ssim_sall)
    print(hcp_mse_sall)
    print(
        'PNSR avg {psnr:.2f} std {psnr_std:.2f} SSIM avg {ssim:.3f} std {ssim_std:.3f} avg NRMSE {nrmse:.3f} std {nrmse_std:.3f}'
        .format(psnr=np.array(hcp_psnr_sall).mean(), psnr_std=np.array(hcp_psnr_sall).std(),
                ssim=np.array(hcp_ssim_sall).mean(),
                ssim_std=np.array(hcp_ssim_sall).std(), nrmse=np.array(hcp_mse_sall).mean(),
                nrmse_std=np.array(hcp_mse_sall).std()))

    print("**************************************************************************")
    print("BIDS")
    print(bids_psnr_sall)
    print(bids_ssim_sall)
    print(bids_mse_sall)
    print(
        'PNSR avg {psnr:.2f} std {psnr_std:.2f} SSIM avg {ssim:.3f} std {ssim_std:.3f} avg NRMSE {nrmse:.3f} std {nrmse_std:.3f}'
        .format(psnr=np.array(bids_psnr_sall).mean(), psnr_std=np.array(bids_psnr_sall).std(),
                ssim=np.array(bids_ssim_sall).mean(),
                ssim_std=np.array(bids_ssim_sall).std(), nrmse=np.array(bids_mse_sall).mean(),
                nrmse_std=np.array(bids_mse_sall).std()))

    print("**************************************************************************")
    print("MPILMBD")
    print(max_psnr_sall)
    print(max_ssim_sall)
    print(max_mse_sall)
    print(
        'PNSR avg {psnr:.2f} std {psnr_std:.2f} SSIM avg {ssim:.3f} std {ssim_std:.3f} avg NRMSE {nrmse:.3f} std {nrmse_std:.3f}'
        .format(psnr=np.array(max_psnr_sall).mean(), psnr_std=np.array(max_psnr_sall).std(),
                ssim=np.array(max_ssim_sall).mean(),
                ssim_std=np.array(max_ssim_sall).std(), nrmse=np.array(max_mse_sall).mean(),
                nrmse_std=np.array(max_mse_sall).std()))

    print("**************************************************************************")
    print("ADNI")
    print(adni_psnr_sall)
    print(adni_ssim_sall)
    print(adni_mse_sall)
    print(
        'PNSR avg {psnr:.2f} std {psnr_std:.2f} SSIM avg {ssim:.3f} std {ssim_std:.3f} avg NRMSE {nrmse:.3f} std {nrmse_std:.3f}'
        .format(psnr=np.array(adni_psnr_sall).mean(), psnr_std=np.array(adni_psnr_sall).std(),
                ssim=np.array(adni_ssim_sall).mean(),
                ssim_std=np.array(adni_ssim_sall).std(), nrmse=np.array(adni_mse_sall).mean(),
                nrmse_std=np.array(adni_mse_sall).std()))

    print("**************************************************************************")
    print("PPMI")
    print(ppmi_psnr_sall)
    print(ppmi_ssim_sall)
    print(ppmi_mse_sall)
    print(
        'PNSR avg {psnr:.2f} std {psnr_std:.2f} SSIM avg {ssim:.3f} std {ssim_std:.3f} avg NRMSE {nrmse:.3f} std {nrmse_std:.3f}'
        .format(psnr=np.array(ppmi_psnr_sall).mean(), psnr_std=np.array(ppmi_psnr_sall).std(),
                ssim=np.array(ppmi_ssim_sall).mean(),
                ssim_std=np.array(ppmi_ssim_sall).std(), nrmse=np.array(ppmi_mse_sall).mean(),
                nrmse_std=np.array(ppmi_mse_sall).std()))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
