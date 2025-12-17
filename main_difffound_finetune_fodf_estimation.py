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
from utils.metrics import calculating_val_metrics_onesubject_fodf_optimized
from model.difffound_finetune_fodf_estimation import build_difffound
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
            loss = criterion(outputs, targets)
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

    acc_sall = []

    c_acc_sall = []

    h_acc_sall = []

    b_acc_sall = []

    m_acc_sall = []

    a_acc_sall = []

    p_acc_sall = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1][0]
        mask_path = batch[2][0]
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(images.permute([0, 4, 1, 2, 3]).contiguous())
        out = out[0].cpu().detach().numpy()

        acc_all = calculating_val_metrics_onesubject_fodf_optimized(out, target.numpy(), mask_path, save_image=False)

        acc_sall.append(acc_all)

        if os.path.basename(mask_path).split("_")[0] == "child":
            c_acc_sall.append(acc_all)

        if os.path.basename(mask_path).split("_")[0] == "HCP":
            h_acc_sall.append(acc_all)

        if os.path.basename(mask_path).split("_")[0] == "bids":
            b_acc_sall.append(acc_all)

        if os.path.basename(mask_path).split("_")[0] == "max":
            m_acc_sall.append(acc_all)

        if os.path.basename(mask_path).split("_")[0] == "adni":
            a_acc_sall.append(acc_all)

        if os.path.basename(mask_path).split("_")[0] == "ppmi":
            p_acc_sall.append(acc_all)

        metric_logger.meters['acc'].update(acc_all.item())

        print('ACC {acc:.3f}'
              .format(acc=acc_all))
    metric_logger.synchronize_between_processes()

    print("*************************************************************")
    print("All Data")
    print(acc_sall)
    print('ACC avg {acc:.3f} {acc_std:.3f}'
          .format(acc=np.array(acc_sall).mean(), acc_std=np.array(acc_sall).std()))

    print("*************************************************************")
    print("Child Data")
    print(c_acc_sall)
    print('ACC avg {acc:.3f} {acc_std:.3f}'
          .format(acc=np.array(c_acc_sall).mean(), acc_std=np.array(c_acc_sall).std()))

    print("*************************************************************")
    print("HCP Data")
    print(h_acc_sall)
    print('ACC avg {acc:.3f} {acc_std:.3f}'
          .format(acc=np.array(h_acc_sall).mean(), acc_std=np.array(h_acc_sall).std()))

    print("*************************************************************")
    print("BIDS Data")
    print(b_acc_sall)
    print('ACC avg {acc:.3f} {acc_std:.3f}'
          .format(acc=np.array(b_acc_sall).mean(), acc_std=np.array(b_acc_sall).std()))

    print("*************************************************************")
    print("MAX Data")
    print(m_acc_sall)
    print('ACC avg {acc:.3f} {acc_std:.3f}'
          .format(acc=np.array(m_acc_sall).mean(), acc_std=np.array(m_acc_sall).std()))

    print("*************************************************************")
    print("ADNI Data")
    print(a_acc_sall)
    print('ACC avg {acc:.3f} {acc_std:.3f}'
          .format(acc=np.array(a_acc_sall).mean(), acc_std=np.array(a_acc_sall).std()))

    print("*************************************************************")
    print("PPMI Data")
    print(p_acc_sall)
    print('ACC avg {acc:.3f} {acc_std:.3f}'
          .format(acc=np.array(p_acc_sall).mean(), acc_std=np.array(p_acc_sall).std()))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
