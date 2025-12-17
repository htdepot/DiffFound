import os
import sys
import warnings
import datetime
import json
import time
import random
import math
import numpy as np
from typing import Iterable
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import wandb
import torch.nn as nn
from utils import misc, lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.metrics import calculating_val_metrics_onesubject_gfa
from model.difffound_finetune_super_resolution import build_difffound
from dataset.finetune_dataset_super_resolution import Dataset_Dmri


def interpolate_pos_embed(checkpoint_model):
    if 'encoder.pos_embed.pos_embeds' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['encoder.pos_embed.pos_embeds']
        embedding_size = pos_embed_checkpoint.shape[-1]
        pos_embed_checkpoint = pos_embed_checkpoint.view(1, 30, -1, embedding_size)
        new_pos_embed = torch.nn.functional.interpolate(pos_embed_checkpoint[0].permute(0, 2, 1), size=27, mode='linear', align_corners=False)
        new_pos_embed = new_pos_embed.permute(0, 2, 1).contiguous().view(-1, embedding_size).unsqueeze(0)
        checkpoint_model['encoder.pos_embed.pos_embeds'] = new_pos_embed


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
    eff_batch_size = args.batch_size * args.accum_iter * num_tasks
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    cudnn.benchmark = True

    db_train = Dataset_Dmri(args.data_path, args.data_gt_path, is_train=True)

    db_eval = Dataset_Dmri(args.test_data_path, args.test_data_gt_path, mask_path=args.mask_data_path)


    if args.env.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            db_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_eval = torch.utils.data.DistributedSampler(
            db_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(db_train)
        sampler_eval = torch.utils.data.SequentialSampler(db_eval)

    data_loader_train = torch.utils.data.DataLoader(
        db_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=False,
        drop_last=True
    )

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

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['state_dict']
        checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint_model.items()}

        interpolate_pos_embed(checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # freezing encoder
        for group_name, p in model.named_parameters():
            if group_name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False


    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.env.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.env.gpu],
                                                          find_unused_parameters=False)




    param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    modules = {
        'state_dict': model,
        'optimizer': optimizer,
        'loss_scaler': loss_scaler,
    }

    ckpt_manager = misc.CheckpointManager(
        modules=modules,
        ckpt_dir=f"{job_dir}/checkpoints",
        epochs=args.epochs,
        save_freq=args.log.save_freq)

    if args.resume:
        args.start_epoch = ckpt_manager.resume()

    if args.log.use_wandb and args.log.rank == 0:

        if args.log.wandb_id is None:
            args.log.wandb_id = wandb.util.generate_id()
        run = wandb.init(project=f"{args.log.proj_name}_{args.log.dataset}",
                         name=args.log.run_name,
                         id=args.log.wandb_id,
                         resume='allow',
                         dir=args.log.output_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    criterion = nn.MSELoss()
    for epoch in range(args.start_epoch, args.epochs):
        if args.env.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, criterion,
            args=args
        )

        if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
            test_stats = evaluate(data_loader_eval, model, device)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch,
                         'n_parameters': n_parameters}
            if misc.is_main_process():
                with open(os.path.join(job_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                         'n_parameters': n_parameters}
            if misc.is_main_process():
                with open(os.path.join(job_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        # save checkpoint
        ckpt_manager.checkpoint(epoch + 1, {'epoch': epoch + 1})



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, criterion, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.log.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for it, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # use acuumulated gradients
        if it % accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(optimizer, it / len(data_loader) + epoch, args)
            metric_logger.update(lr=lr)

        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():

            outputs = model(images)
            loss = criterion(outputs, targets.permute(0, 2, 3, 4, 1).contiguous())
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(it + 1) % accum_iter == 0)

        if (it + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

    if args.log.use_wandb and args.log.rank == 0:
        wandb.log(
            {
                "loss": metric_logger.meters['loss'].avg
            },
            step=epoch,
        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):

    metric_logger = misc.MetricLogger(delimiter=" ")
    header = 'Test:'

    # compute output
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
        target = batch[1][0].permute([1, 2, 3, 0])
        mask_path = batch[2][0]
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(images)
        out = out[0].cpu().detach().numpy()

        vis_output, vis_target, nrmse_all, psnr_all, ssim_all  = (
            calculating_val_metrics_onesubject_gfa(out, target.numpy(), mask_path, save_image=False))

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
