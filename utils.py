from collections import OrderedDict

import logging
import torch

import torch.distributed as dist

from PIL import Image
from pytorch_gan_metrics import get_inception_score_and_fid


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    '''
    Step the EMA model towards the current model.
    '''
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    '''
    Set requires_grad flag for all parameters in a model.
    '''
    for p in model.parameters():
        p.requires_grad = flag


def save_ckpt(args, model, ema, opt, checkpoint_path):
    '''
    Save a checkpoint containing the online model, EMA, and optimizer states.
    '''
    checkpoint = {
            'args': args,
            'model': model.module.state_dict(),
            'ema': ema.state_dict(),
            'opt': opt.state_dict(),
            }
    torch.save(checkpoint, checkpoint_path)
 

def sample_image(args, model, device, image_path, set_train=False, cond=False):
    '''
    sample a batch of images for visualization.
    set set_train to true if you are using the online model for sampling.
    '''
    model.eval()
    
    n_row = 16
    size = args.input_size

    z = torch.randn(n_row*n_row, 3, size, size).to(device)
    c = torch.randint(0, args.num_classes, (n_row*n_row,)).to(device) if cond else None
    with torch.no_grad():
        x = model(z, c)
    
    x = x.view(n_row, n_row, 3, size, size)
    x = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
    images = x.permute(0, 3, 1, 4, 2).reshape(n_row*size, n_row*size, 3).cpu().numpy()
    
    Image.fromarray(images, 'RGB').save(image_path)
    del images, x, z, c
    torch.cuda.empty_cache()

    if set_train:
        model.train()


def num_to_groups(num, divisor):
    '''
    Compute number of samples in each batch to evenly divide the total eval samples.
    '''
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def sample_fid(args, model, device, rank, set_train=False, cond=False):
    '''
    Sample args.eval_samples images in parallel for FID and IS calculation. Default 50k images.
    Set set_train to True if you are using the online model for sampling.
    '''
    # Setup batches for each node
    assert args.eval_samples % dist.get_world_size() == 0
    samples_per_node = args.eval_samples // dist.get_world_size()
    batches = num_to_groups(samples_per_node, args.eval_batch_size)
    
    # Dist EMA/online evaluation
    # No need to use the DDP wrapper here
    # As we do not need grad sycn (by DDP)
    model.eval()
    model = model.to(device)
    
    n_cls = args.num_classes
    size = args.input_size

    images = []
    with torch.no_grad():
        for n in batches:
            z = torch.randn(n, 3, size, size).to(device)
            c = torch.randint(0, n_cls, (n,)).to(device) if cond else None
            x = model(z, c)
            images.append(x)
    images = torch.cat(images, dim=0)
    
    torch.cuda.empty_cache()
    if set_train:
        model.train()

    return images


def compute_fid_is(args, all_images, rank):
    '''
    Compute FID and IS using provided images.
    '''
    # Post-process to images.
    all_images = torch.cat(all_images, dim=0)
    all_images = (all_images * 127.5 + 128).clip(0, 255).to(torch.uint8).float().div(255).cpu()
    
    # Compute FID & IS
    (IS, IS_std), FID = get_inception_score_and_fid(all_images, args.stat_path)
    torch.cuda.empty_cache()

    return FID, IS
 
