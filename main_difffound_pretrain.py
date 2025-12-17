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

import timm.optim.optim_factory as optim_factory

from utils import misc, lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from model.difffound_pretrain import build_lmim
from dataset.datasets_pretrain import CachedMMapDataset

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
    # ngpus_per_node = 2
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
    print(eff_batch_size)
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    cudnn.benchmark = True

    db_train = CachedMMapDataset(args.data_path, warmup_cache=20, max_cache_size=50, max_open_files=100)

    if args.env.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            db_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    else:
        sampler_train = torch.utils.data.RandomSampler(db_train)


    data_loader_train = torch.utils.data.DataLoader(
        db_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.env.workers > 0 else False,
        prefetch_factor=2
    )

    # define the model
    model = build_lmim(
         args.encoder,
        loss=args.loss,
        grid_size=args.grid_size,
        tau=args.tau,
        target_depth=args.target_depth,
        decoder_depth=args.decoder_depth,
        patch_gap=args.patch_gap,
        num_vis=args.num_vis,
        avg_sim_coeff=args.avg_sim_coeff,
        avg_vis_mask_token=args.avg_vis_mask_token,
        drop=args.drop,
        attn_drop=args.attn_drop,
        drop_path=args.drop_path,
        freeze_pe=args.freeze_pe,
        proj_cfg=args.proj,
        mask_target=args.mask_target,
        mask_ratio = args.mask_ratio
    )
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.env.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.env.gpu],find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay,
                                                           skip_list=[n for n, p in
                                                                                 model_without_ddp.named_parameters() if
                                                                                 'bias' in n or 'norm' in n or 'embed' in n])

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Checkpointing
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
        # misc.init_wandb(args, job_dir, entity=args.log.wandb_entity, project=args.log.wandb_project, job_name=args.job_name)
        if args.log.wandb_id is None:
            args.log.wandb_id = wandb.util.generate_id()
        run = wandb.init(project=f"{args.log.proj_name}_{args.log.dataset}",
                        name=args.log.run_name,
                        # config=vars(args),
                        id=args.log.wandb_id,
                        resume='allow',
                        dir=args.log.output_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.env.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )

        # save checkpoint
        ckpt_manager.checkpoint(epoch+1, {'epoch': epoch+1})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        if misc.is_main_process():
            with open(os.path.join(job_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # if args.env.rank == 0 and not args.log.use_wandb:
    #     run.finish()

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.log.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for it, images in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # use acuumulated gradients
        if it % accum_iter == 0:
            lr = lr_sched.adjust_learning_rate(optimizer, it / len(data_loader) + epoch, args)
            mom = adjust_target_momentum(it / len(data_loader) + epoch, args)
            sim_trg = cosine_decay_schedule(it / len(data_loader) + epoch, args.epochs, init_val=args.sim_init, end_val=args.sim_end)
            
            metric_logger.update(mom=mom)
            metric_logger.update(lr=lr)
            metric_logger.update(sim_trg=sim_trg)

        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, metrics = model(images, mom=mom, sim_trg=sim_trg, update_ema=it % accum_iter == 0)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value, **metrics)

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
        # global_step = epoch * len(data_loader) + it
        log_dict = {k: meter.avg for k, meter in metric_logger.meters.items()}
        wandb.log(log_dict, step=epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def adjust_target_momentum(epoch, args):
    """Adjust target momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.target_mom)
    return m

def cosine_decay_schedule(ep, epochs, init_val, end_val):
    return 0.5 * (1+math.cos(math.pi * ep / epochs)) * (init_val - end_val) + end_val

