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

        acc_all = calculating_val_metrics_onesubject_fodf_optimized(out, target.numpy(), mask_path, save_images, output_dir)

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
