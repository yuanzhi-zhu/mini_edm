import argparse
import os
join = os.path.join
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import random
import logging
from datetime import datetime
from diffusers import UNet2DModel

extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2). (independent design)
## https://github.com/NVlabs/edm/blob/main/generate.py#L25

@torch.no_grad()
def edm_sampler(
    edm, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([edm.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_hat = x_next
        t_hat = t_cur
        
        # Euler step.
        denoised = edm(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = edm(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


#----------------------------------------------------------------------------
# EDM model

class EDM():
    def __init__(self, model=None, cfg=None):
        self.cfg = cfg
        self.device = self.cfg.device
        self.model = model.to(self.device)
        ## parameters
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.rho = cfg.rho
        self.sigma_data = cfg.sigma_data
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5

    def model_forward_wrapper(self, x, sigma, **kwargs):
        """Wrapper for the model call"""
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        label = kwargs['labels'] if 'labels' in kwargs else None
        model_output = self.model(torch.einsum('b,bijk->bijk', c_in, x), c_noise, class_labels=label)
        try:
            model_output = model_output.sample
        except:
            pass
        return torch.einsum('b,bijk->bijk', c_skip, x) + torch.einsum('b,bijk->bijk', c_out, model_output)
        
    def train_step(self, images, labels=None, augment_pipe=None, **kwargs):
        ## https://github.com/NVlabs/edm/blob/main/training/loss.py#L66
        rnd_normal = torch.randn([images.shape[0]], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noise = torch.randn_like(y)
        n = torch.einsum('b,bijk->bijk', sigma, noise)
        D_yn = self.model_forward_wrapper(y + n, sigma, labels=labels, augment_labels=augment_labels)
        if self.cfg.gt_guide_type == 'l2':
            loss = torch.einsum('b,bijk->bijk', weight, ((D_yn - y) ** 2))
        elif self.cfg.gt_guide_type == 'l1':
            loss = torch.einsum('b,bijk->bijk', weight, (torch.abs(D_yn - y)))
        else:
            raise NotImplementedError(f'gt_guide_type {self.cfg.gt_guide_type} not implemented')
        return loss.sum()
    
    def __call__(self, x, sigma, labels=None, augment_labels=None):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), labels=labels, augment_labels=augment_labels)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

## model creater
def create_model(config):
    ## get block_out_channels using model_channels and channel_mult
    block_out_channels = []
    for i in range(len(config.channel_mult)):
        block_out_channels.append(config.model_channels*config.channel_mult[i])
    block_out_channels = tuple(block_out_channels)
    ## get down_block_types and up_block_types using config.img_size, config.attn_resolutions and config.channel_mult
    down_block_types = []
    up_block_types = []
    for i in range(len(config.channel_mult)):
        res = config.img_size >> i
        if res in config.attn_resolutions:
            down_block_types.append("AttnDownBlock2D")
        else:
            down_block_types.append("DownBlock2D")
        if config.img_size // res in config.attn_resolutions:
            up_block_types.append("AttnUpBlock2D")
        else:
            up_block_types.append("UpBlock2D")
    down_block_types = tuple(down_block_types)
    up_block_types = tuple(up_block_types)
    ## create model
    unet = UNet2DModel(
                sample_size=config.img_size,
                in_channels=config.channels,
                out_channels=config.channels,
                layers_per_block=config.layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=down_block_types,
                up_block_types=up_block_types,
                norm_num_groups=min(32, config.model_channels),
                )
    return unet
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument('--seed', default=42, type=int, help='global seed')
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--save_model_iters", type=int, default=5000)
    # EDM models parameters
    parser.add_argument('--gt_guide_type', default='l2', type=str, help='gt_guide_type loss type')
    parser.add_argument('--sigma_min', default=0.002, type=float, help='sigma_min')
    parser.add_argument('--sigma_max', default=80.0, type=float, help='sigma_max')
    parser.add_argument('--rho', default=7., type=float, help='Schedule hyper-parameter')
    parser.add_argument('--sigma_data', default=0.5, type=float, help='sigma_data used in EDM for c_skip and c_out')
    # Sampling parameters
    parser.add_argument('--total_steps', default=20, type=int, help='total_steps')
    parser.add_argument("--save_images_step", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    # Model architecture
    parser.add_argument('--model_channels', default=64, type=int, help='model_channels')
    parser.add_argument('--channel_mult', default=[1,2,2,2], type=int, nargs='+', help='channel_mult')
    parser.add_argument('--attn_resolutions', default=[], type=int, nargs='+', help='attn_resolutions')
    parser.add_argument('--layers_per_block', default=4, type=int, help='num_blocks')
    
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    channels = {'mnist': 1, 'cifar10': 3}
    config.channels = channels[config.dataset]

    # workdir setup
    config.expr = f"{config.expr}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    outdir = f"exps/{config.expr}_{run_id}"
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(filename=f'{outdir}/std.log', filemode='w', 
                        format='%(asctime)s %(levelname)s --> %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.info("#################### Arguments: ####################")
    for arg in vars(config):
        logger.info(f"\t{arg}: {getattr(config, arg)}")

    ## set random seed everywhere
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)  # for multi-GPU.
    random.seed(config.seed)  # Python random module.
    torch.manual_seed(config.seed)

    ## load dataset
    ### create dataloader
    if config.dataset == 'mnist':
        img_dataset = torchvision.datasets.MNIST(root='datasets/mnist', download=True,
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.Resize(config.img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5,), (0.5,))]
                                            ),)
    elif config.dataset == 'cifar10':
        img_dataset = torchvision.datasets.CIFAR10(root='datasets/cifar', download=True,
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.Resize(config.img_size),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                            ),)
    else:
        raise NotImplementedError(f'dataset {config.dataset} not implemented')
    dataloader = torch.utils.data.DataLoader(img_dataset,
                                                batch_size=config.train_batch_size,
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=True)
    logger.info(f'length of dataloader: {len(dataloader)}')

    ## init model
    unet = create_model(config)
    edm = EDM(model=unet, cfg=config)
    edm.model.train()
    logger.info("#################### Model: ####################")
    logger.info(f'{unet}')
    logger.info(f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}')

    ## setup optimizer
    optimizer = torch.optim.AdamW(edm.model.parameters(),lr=config.learning_rate)
    # optimizer = torch.optim.Adam(edm.model.parameters(),lr=config.learning_rate)

    logger.info("#################### Training ####################")
    train_loss_values = []
    progress_bar = tqdm(total=config.num_steps)
    for step in range(config.num_steps):
        batch_loss = torch.tensor(0.0, device=device)
        # accumulation steps
        for _ in range(config.accumulation_steps):
            batch, label_dic = next(iter(dataloader))
            batch = batch.to(device)
            loss = edm.train_step(batch)
            loss /= (config.accumulation_steps * config.train_batch_size)
            loss.backward()
            batch_loss += loss
        nn.utils.clip_grad_norm_(edm.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        train_loss_values.append(batch_loss.detach().item())
        logs = {"loss": loss.detach().item()}
        progress_bar.set_postfix(**logs)
        ## save images
        if config.save_images_step and (step % config.save_images_step == 0 or step == config.num_steps - 1):
            # generate data with the model to later visualize the learning process
            edm.model.eval()
            x_T = torch.randn([config.eval_batch_size, config.channels, config.img_size, config.img_size]).to(device).float()
            sample = edm_sampler(edm, x_T, num_steps=config.total_steps).detach().cpu()
            save_image((sample/2+0.5).clamp(0, 1), f'{outdir}/image_{step}.png')
            logger.info(f'step: {step:08d}, average loss: {np.average(train_loss_values):0.10f}; batch loss: {batch_loss.detach().item():0.10f}')
            edm.model.train()
        ## save model
        if config.save_model_iters and (step % config.save_model_iters == 0 or step == config.num_steps - 1) and step > 0:
            torch.save(edm.model.state_dict(), f"{outdir}/model_{step}.pth")
