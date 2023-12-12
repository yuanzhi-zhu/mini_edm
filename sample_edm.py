import argparse
import os
join = os.path.join
import glob
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import random
from datetime import datetime

extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

#----------------------------------------------------------------------------
# EDM sampler & EDM model

from train_edm import edm_sampler
from train_edm import EDM
from train_edm import create_model

class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = fid.build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_np = self.files[i]
        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized)) * 255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


# https://github.com/openai/consistency_models_cifar10/blob/main/jcm/metrics.py#L117
def compute_fid(
    samples,
    feat_model,
    dataset_name="cifar10",
    dataset_res=32,
    dataset_split="train",
    batch_size=128,
    num_workers=12,
    mode="legacy_tensorflow",
    device=torch.device("cuda:0"),
    seed=0,
):
    dataset = ResizeDataset(samples, mode=mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    l_feats = []
    for batch in tqdm(dataloader):
        l_feats.append(fid.get_batch_features(batch, feat_model, device))
    np_feats = np.concatenate(l_feats)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    ref_mu, ref_sigma = fid.get_reference_statistics(
        dataset_name, dataset_res, mode=mode, seed=seed, split=dataset_split
    )

    score = fid.frechet_distance(mu, sigma, ref_mu, ref_sigma)

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="sampling", help="experiment name")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument('--seed', default=42, type=int, help='global seed')
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--model_paths", default='', type=str, help='model paths')
    # EDM models parameters
    parser.add_argument('--sigma_min', default=0.002, type=float, help='sigma_min')
    parser.add_argument('--sigma_max', default=80.0, type=float, help='sigma_max')
    parser.add_argument('--rho', default=7., type=float, help='Schedule hyper-parameter')
    parser.add_argument('--sigma_data', default=0.5, type=float, help='sigma_data used in EDM for c_skip and c_out')
    # Sampling parameters
    parser.add_argument('--total_steps', default=18, type=int, help='total_steps')
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--sample_mode", type=str, default='fid', help='sample mode')
    parser.add_argument('--num_fid_sample', default=10000, type=int, help='num_fid_sample')
    parser.add_argument('--t_path', default='./CIFAR-10-images/train', type=str, help='source clean image path')
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

    print("#################### Arguments: ####################")
    for arg in vars(config):
        print(f"\t{arg}: {getattr(config, arg)}")
    ## set random seed everywhere
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)  # for multi-GPU.
    random.seed(config.seed)  # Python random module.
    torch.manual_seed(config.seed)
    ## init model
    unet = create_model(config)
    edm = EDM(model=unet, cfg=config)
    ## set up fid recorder
    if config.sample_mode == 'fid':
        from cleanfid import fid
        ### build feature extractor
        mode = "legacy_tensorflow"
        feat_model = fid.build_feature_extractor(mode, device)
    elif config.sample_mode == 'save':
        # workdir setup
        config.expr = f"{config.expr}_{config.dataset}_{config.sample_mode}"
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        outdir = f"exps/{config.expr}_{run_id}"
        os.makedirs(outdir, exist_ok=True)
    else:
        raise NotImplementedError(f"sample_mode: {config.sample_mode} not implemented!")

    # Free any unused GPU memory
    torch.cuda.empty_cache()

    ## get all model paths in the folder config.model_paths
    model_extensions = ['*.pt', '*.pth']
    model_paths = []
    for extension in model_extensions:
        search_path = os.path.join(config.model_paths, '**', extension)
        model_paths.extend(glob.glob(search_path, recursive=True))
        model_paths = sorted(model_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # model_paths = model_paths[10:]
    for model_path in model_paths:
        ## set random seed everywhere
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)  # for multi-GPU.
        random.seed(config.seed)  # Python random module.
        torch.manual_seed(config.seed)
        print(f"#################### Model path: {model_path} ####################")
        ## get model name
        model_name = model_path.split('/')[-1].split('.')[0]
        ## load model
        checkpoint = torch.load(model_path, map_location=device)
        edm.model.load_state_dict(checkpoint)
        for param in edm.model.parameters():
            param.requires_grad = False
        edm.model.eval()
        if config.sample_mode == 'fid':
            # save samples for fid calculation
            fid_batch_size = config.fid_batch_size
            fid_iters = config.num_fid_sample // fid_batch_size + 1
            # sampling
            all_samples = []
            for r in range(fid_iters):
                with torch.no_grad():
                    noise = torch.randn([fid_batch_size, config.channels, config.img_size, config.img_size]).to(device)
                    samples = edm_sampler(edm, noise, num_steps=config.total_steps, use_ema=False).detach().cpu()
                    samples.mul_(0.5).add_(0.5)
                print(f"fid sampling -- model_name: {model_name}, round: {r}, steps: {config.total_steps*2-1}")
                samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                samples = samples.reshape((-1, config.img_size, config.img_size, config.channels))
                all_samples.append(samples)

            # compute FID
            all_samples = np.concatenate(all_samples, axis=0)
            print(f'all_samples shape: {all_samples.shape}')
            print(f'{all_samples.mean()}, {all_samples.std()}')
            fid_score = compute_fid(
                        all_samples[: config.num_fid_sample],
                        mode=mode,
                        device=device,
                        feat_model=feat_model,
                        seed=config.seed,
                        num_workers=0,
                    )
            print(f'model: {model_name}; fid_score: {fid_score:0.6f}')

        elif config.sample_mode == 'save':
            x_T = torch.randn([config.eval_batch_size, config.channels, config.img_size, config.img_size]).to(device).float()
            sample = edm_sampler(edm, x_T, num_steps=config.total_steps).detach().cpu()
            save_image((sample/2+0.5).clamp(0, 1), f'{outdir}/image_{model_name}.png')
            print(f"save sample with shape {sample.shape} to {outdir}/image_{model_name}.png")
