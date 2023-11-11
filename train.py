import argparse
import os
import time

import numpy as np
from glob import glob

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchprofile import profile_macs

from utils import (
        create_logger, save_ckpt, 
        update_ema, requires_grad,  
        sample_image, sample_fid, compute_fid_is
        )
from models import model_dict
from losses import loss_dict
from datasets import PairedDataset, PairedCondDataset

# For future ImageNet training & sampling
# from diffusers.models import AutoencoderKL

from torchdeq import add_deq_args
from torchdeq.loss import fp_correction


def main(args):
    '''
    Model training.
    '''
    # Setup DDP
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert args.global_batch_size % world_size == 0, f'Batch size must be divisible by world size.'
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f'Starting rank={rank}, seed={seed}, world_size={world_size}.')

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f'{args.results_dir}/*'))
        model_string_name = args.model.replace('/', '-')
        experiment_dir = f'{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.name}'

        checkpoint_dir = f'{experiment_dir}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        sample_dir = f'{experiment_dir}/samples'
        os.makedirs(sample_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f'Experiment directory created at {experiment_dir}')
    else:
        logger = create_logger()

    # Create model
    model = model_dict[args.model](
            args=args,
            input_size=args.input_size,
            num_classes=args.num_classes,
            cond=args.cond
            )
    ema = model_dict[args.model](
            args=args,
            input_size=args.input_size,
            num_classes=args.num_classes,
            cond=args.cond
            ).to(device)
    requires_grad(ema, False)
    
    # Setup DDP
    model = DDP(model.to(device), device_ids=[rank])
    logger.info(f'Model Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test FLOPs
    if rank == 0:
        test_case = torch.randn(1, 3, args.input_size, args.input_size).to(device)
        if args.cond:
            test_c = torch.randint(0, 10, (1,1)).to(device)
            macs = profile_macs(model, (test_case, test_c))
            del test_case, test_c
        else:
            macs = profile_macs(model, test_case)
            del test_case
        logger.info(f'Model MACs: {macs:,}')
    dist.barrier()

    # For future ImageNet training
    # vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{args.vae}').to(device)

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data
    if args.cond:
        dataset = PairedCondDataset(args.data_path, world_size=world_size, rank=rank)
    else:
        dataset = PairedDataset(args.data_path, world_size=world_size, rank=rank)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f'Dataset contains {len(dataset):,} images ({dataset.data_dir})')

    # Prepare models for training
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()
    ema.eval()                              # EMA model should always be in eval mode
    
    # Loss fn
    loss_fn = loss_dict[args.loss]().to(device)

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0
    total_steps = args.epochs * (len(dataset) / args.global_batch_size)

    # Resume from the prev checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=torch.device('cpu'))
        model.module.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        train_steps = max(args.resume_iter, 0)
        
        logger.info(f'Resume from {args.resume}..')

    start_time = time.time()
    logger.info(f'Training for {args.epochs} epochs...')
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f'Beginning epoch {epoch}...')

        for data in loader:
            # Unpack data
            if args.cond:
                z, x, c = data
                z, x, c = z.to(device), x.to(device), c.to(device).max(dim=1)[1]
            else:
                z, x = data
                z, x, c = z.to(device), x.to(device), None
            
            # Loss & Grad
            x_pred = model(z, c)
            loss, loss_list = fp_correction(loss_fn, (x_pred, x), return_loss_values=True)
            opt.zero_grad()
            loss.backward()
            
            # LR Warmup
            if train_steps < args.warmup_iter:
                curr_lr = args.lr * (train_steps+1) / args.warmup_iter
                opt.param_groups[0]['lr'] = curr_lr

            opt.step()
            update_ema(ema, model.module, decay=args.ema_decay)

            running_loss += loss_list[-1]
            log_steps += 1
            train_steps += 1

            # Log training progress
            if train_steps % args.log_every == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f'(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}')

                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f'{checkpoint_dir}/{train_steps:07d}.pth'
                    save_ckpt(args, model, ema, opt, checkpoint_path)
                    logger.info(f'Saved checkpoint to {checkpoint_path}')
                dist.barrier()

            # Save the latest checkpoint
            if train_steps % args.save_latest_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f'{checkpoint_dir}/latest.pth'
                    save_ckpt(args, model, ema, opt, checkpoint_path)
                    logger.info(f'Saved latest checkpoint to {checkpoint_path}')
                dist.barrier()

            # Sample images
            if train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    image_path = f'{sample_dir}/{train_steps}.png'
                    sample_image(args, ema, device, image_path, cond=args.cond)
                    logger.info(f'Saved samples to {image_path}')
                dist.barrier()
            
            # Compute FID and IS
            if train_steps % args.eval_every == 0 and train_steps > 0:
                images = sample_fid(args, ema, device, rank, cond=args.cond)

                # In case you want to sample from the online model
                # images = sample_fid(args, model.module, device, rank, cond=args.cond, set_grad=True)
                
                # DDP sync
                all_images = [torch.zeros_like(images) for _ in range(world_size)]
                dist.gather(images, all_images if rank == 0 else None, dst=0)
                if rank == 0:
                    FID, IS = compute_fid_is(args, all_images, rank)
                    logger.info(f'FID {FID:0.2f}, IS {IS:0.2f} at iters {train_steps}.')
                
                del images, all_images
                dist.barrier()

            # Check training schedule
            if train_steps > total_steps:
                break

    if rank == 0:
        checkpoint_path = f'{checkpoint_dir}/final.pth'
        save_ckpt(args, model, ema, opt, checkpoint_path)
        logger.info(f'Saved final checkpoint to {checkpoint_path}')
    dist.barrier()
    
    # Finish training
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--results_dir', type=str, default='results')

    parser.add_argument('--model', type=str, choices=list(model_dict.keys()), default='GET-S/2')
    parser.add_argument('--input_size', type=int, default=32)

    parser.add_argument('--cond', action='store_true', help='Run conditional model.')
    parser.add_argument('--num_classes', type=int, default=10)
    
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'lpips', 'dists'], default='l1')
    parser.add_argument('--vae', type=str, choices=['ema', 'mse'], default='ema')
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_iter', type=int, default=0, help="warmup for the given iterations")
    parser.add_argument('--ema_decay', type=float, default=0.9999)
 
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--global_batch_size', type=int, default=256)
    parser.add_argument('--global_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--mem', action='store_true', help='Enable O(1) memory.')

    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--ckpt_every', type=int, default=50000)
    parser.add_argument('--save_latest_every', type=int, default=10000)
    parser.add_argument('--sample_every', type=int, default=10000)

    parser.add_argument('--eval_every', type=int, default=50000)
    parser.add_argument('--eval_samples', type=int, default=50000)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--stat_path', type=str, default='YOUR_STAT_PATH/cifar10.test.npz')

    parser.add_argument('--resume', help="restore checkpoint for training")
    parser.add_argument('--resume_iter', type=int, default=-1, help="resume from the given iterations")

    # Add for DEQs
    add_deq_args(parser)

    args = parser.parse_args()
    main(args)
