import argparse
import os
join = os.path.join
import torch
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import copy
import random
import logging
from datetime import datetime

extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2). (independent design)
## https://github.com/NVlabs/edm/blob/main/generate.py#L25
## deterministic case
@torch.no_grad()
def edm_sampler(
    edm, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    use_ema=True,
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
        denoised = edm(x_hat, t_hat, class_labels, use_ema=use_ema).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = edm(x_next, t_next, class_labels, use_ema=use_ema).to(torch.float64)
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
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        ## parameters
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.rho = cfg.rho
        self.sigma_data = cfg.sigma_data
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.ema_rampup_ratio = 0.05
        self.ema_halflife_kimg = 500

    def model_forward_wrapper(self, x, sigma, use_ema=False, **kwargs):
        """Wrapper for the model call"""
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        label = kwargs['labels'] if 'labels' in kwargs else None
        if use_ema:
            model_output = self.ema(torch.einsum('b,bijk->bijk', c_in, x), c_noise, class_labels=label)
        else:
            model_output = self.model(torch.einsum('b,bijk->bijk', c_in, x), c_noise, class_labels=label)
        try:
            model_output = model_output.sample
        except:
            pass
        return torch.einsum('b,bijk->bijk', c_skip, x) + torch.einsum('b,bijk->bijk', c_out, model_output)
        
    def train_step(self, images, labels=None, augment_pipe=None, **kwargs):
        ### sigma sampling --> continuous & weighted sigma
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
        return loss.mean()
    
    def update_ema(self):
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, step * config.train_batch_size * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (config.train_batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
    
    # used for sampling, set use_ema=True
    def __call__(self, x, sigma, labels=None, augment_labels=None, use_ema=True):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), use_ema=use_ema, labels=labels, augment_labels=augment_labels)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

## UNet model creater
def create_model(config):
    from networks_edm import SongUNet
    unet = SongUNet(in_channels=config.channels, 
                    out_channels=config.channels, 
                    num_blocks=config.layers_per_block, 
                    attn_resolutions=config.attn_resolutions, 
                    model_channels=config.model_channels, 
                    channel_mult=config.channel_mult, 
                    dropout=0.13, 
                    img_resolution=config.img_size, 
                    label_dim=0,
                    embedding_type='positional', 
                    encoder_type='standard', 
                    decoder_type='standard', 
                    augment_dim=9, 
                    channel_mult_noise=1, 
                    resample_filter=[1,1], 
                    )
    pytorch_total_grad_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logging.info(f'total number of trainable parameters in the Score Model: {pytorch_total_grad_params}')
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    logging.info(f'total number of parameters in the Score Model: {pytorch_total_params}')
    return unet
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument('--seed', default=42, type=int, help='global seed')
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--save_model_iters", type=int, default=5000)
    parser.add_argument("--log_step", type=int, default=500)
    parser.add_argument("--train_dataset", action='store_true', default=True)
    parser.add_argument("--desired_class", type=str, default='all')
    parser.add_argument("--train_progress_bar", action='store_true', default=False)
    parser.add_argument("--warmup", type=int, default=5000)
    # EDM models parameters
    parser.add_argument('--gt_guide_type', default='l2', type=str, help='gt_guide_type loss type')
    parser.add_argument('--sigma_min', default=0.002, type=float, help='sigma_min')
    parser.add_argument('--sigma_max', default=80.0, type=float, help='sigma_max')
    parser.add_argument('--rho', default=7., type=float, help='Schedule hyper-parameter')
    parser.add_argument('--sigma_data', default=0.5, type=float, help='sigma_data used in EDM for c_skip and c_out')
    # Sampling parameters
    parser.add_argument('--total_steps', default=18, type=int, help='total_steps')
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
    config.expr = f"{config.expr}_{config.dataset}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    outdir = f"exps/{config.expr}_{run_id}"
    os.makedirs(outdir, exist_ok=True)
    sample_dir = f"{outdir}/samples"
    os.makedirs(sample_dir, exist_ok=True)
    ckpt_dir = f"{outdir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
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
        img_dataset = torchvision.datasets.MNIST(root='datasets/mnist', download=True, train=config.train_dataset,
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.Resize(config.img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5,), (0.5,))]
                                            ),)
        # mnist class labels
        classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
                    '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    elif config.dataset == 'cifar10':
        img_dataset = torchvision.datasets.CIFAR10(root='datasets/cifar', download=True, train=config.train_dataset,
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.Resize(config.img_size),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                            ),)
        # CIFAR10 class labels
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        raise NotImplementedError(f'dataset {config.dataset} not implemented')
    # Filter the dataset to only keep desired_class images
    if config.desired_class != 'all':
        class_idx = classes.index(config.desired_class)
        img_dataset = [(img, label) for img, label in img_dataset if label == class_idx]
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
    # logger.info(f'{unet}')
    logger.info(f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}')

    ## setup optimizer
    # optimizer = torch.optim.AdamW(edm.model.parameters(),lr=config.learning_rate)
    optimizer = torch.optim.Adam(edm.model.parameters(),lr=config.learning_rate)

    logger.info("#################### Training ####################")
    train_loss_values = 0
    if config.train_progress_bar:
        progress_bar = tqdm(total=config.num_steps)
    for step in range(config.num_steps):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        # accumulation steps
        for _ in range(config.accumulation_steps):
            try:
                batch, label_dic = next(data_iterator)
            except:
                data_iterator = iter(dataloader)
                batch, label_dic = next(data_iterator)
            batch = batch.to(device)
            loss = edm.train_step(batch)
            loss /= (config.accumulation_steps)
            loss.backward()
            batch_loss += loss
        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = config.learning_rate * min(step / config.warmup, 1)
        for param in unet.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()
        train_loss_values += (batch_loss.detach().item())
        ## Update EMA.
        edm.update_ema()
        ## Update state
        if config.train_progress_bar:
            logs = {"loss": loss.detach().item()}
            progress_bar.update(1) 
            progress_bar.set_postfix(**logs)
        ## log
        if step % config.log_step == 0 or step == config.num_steps - 1:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'step: {step:08d}, current lr: {current_lr:0.6f} average loss: {train_loss_values/(step+1):0.10f}; batch loss: {batch_loss.detach().item():0.10f}')
        ## save images
        if config.save_images_step and (step % config.save_images_step == 0 or step == config.num_steps - 1):
            # generate data with the model to later visualize the learning process
            edm.model.eval()
            x_T = torch.randn([config.eval_batch_size, config.channels, config.img_size, config.img_size]).to(device).float()
            sample = edm_sampler(edm, x_T, num_steps=config.total_steps).detach().cpu()
            save_image((sample/2+0.5).clamp(0, 1), f'{sample_dir}/image_{step}.png')
            edm.model.train()
        ## save model
        if config.save_model_iters and (step % config.save_model_iters == 0 or step == config.num_steps - 1) and step > 0:
            # torch.save(edm.model.state_dict(), f"{ckpt_dir}/model_{step}.pth")
            torch.save(edm.ema.state_dict(), f"{ckpt_dir}/ema_{step}.pth")
