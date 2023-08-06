import argparse
import os
join = os.path.join
import glob
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import random
from datetime import datetime

extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

#----------------------------------------------------------------------------
# EDM sampler & EDM model

from edm import edm_sampler
from edm import EDM
from edm import create_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="sampling", help="experiment name")
    parser.add_argument('--seed', default=42, type=int, help='global seed')
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--model_paths", default='', type=str, help='model paths')
    # EDM models parameters
    parser.add_argument('--sigma_min', default=0.002, type=float, help='sigma_min')
    parser.add_argument('--sigma_max', default=80.0, type=float, help='sigma_max')
    parser.add_argument('--rho', default=7., type=float, help='Schedule hyper-parameter')
    parser.add_argument('--sigma_data', default=0.5, type=float, help='sigma_data used in EDM for c_skip and c_out')
    # Sampling parameters
    parser.add_argument('--total_steps', default=20, type=int, help='total_steps')
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--sample_mode", type=str, default='fid', help='sample mode')
    parser.add_argument('--num_fid_sample', default=10000, type=int, help='num_fid_sample')
    parser.add_argument('--t_path', default='./CIFAR-10-images/train', type=str, help='source clean image path')
    # Model architecture
    parser.add_argument('--architecture', default='SongUnet', type=str, help='unet architecture')
    parser.add_argument("--channels", type=int, default=3)
    
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    # workdir setup
    config.expr = f"{config.expr}_{config.architecture}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    outdir = f"exps/{config.expr}_{run_id}"
    os.makedirs(outdir, exist_ok=True)

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
        fid_tmp_dir = outdir
        os.makedirs(fid_tmp_dir, exist_ok=True)
        # Create a list of all image file paths in the folder and its subfolders
        src_image_paths = []
        for extension in extensions:
            search_path = os.path.join(config.t_path, '**', extension)
            src_image_paths.extend(glob.glob(search_path, recursive=True))
        src_image_paths_fid = random.sample(src_image_paths, config.num_fid_sample)
        from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
        from pytorch_fid.inception import InceptionV3
        # Load a pre-trained Inception-v3 model and move it to the appropriate device
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        fid_device = device # torch.device('cpu')
        InceptionV3_model = InceptionV3([block_idx]).to(fid_device)
        InceptionV3_model.eval()
        real_features = calculate_activation_statistics(src_image_paths_fid, InceptionV3_model, batch_size=min(128, len(src_image_paths_fid)), device=fid_device)
    # Free any unused GPU memory
    torch.cuda.empty_cache()

    ## get all model paths in the folder config.model_paths
    model_extensions = ['*.pt', '*.pth']
    model_paths = []
    for extension in model_extensions:
        search_path = os.path.join(config.model_paths, '**', extension)
        model_paths.extend(glob.glob(search_path, recursive=True))
    for model_path in model_paths:
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
            fid_iters = config.num_fid_sample // fid_batch_size
            with torch.no_grad():
                for i in tqdm(range(fid_iters)):
                    noise = torch.randn([fid_batch_size, config.channels, config.img_size, config.img_size]).to(device)
                    sample = edm_sampler(edm, noise, num_steps=config.total_steps).detach().cpu()
                    sample.mul_(0.5).add_(0.5)
                    for j in range(fid_batch_size):
                        save_image(sample[j], f"{fid_tmp_dir}/{fid_batch_size*i+j}.jpg")
                gen_image_paths = []
                for extension in extensions:
                    search_path = os.path.join(fid_tmp_dir, '**', extension)
                    gen_image_paths.extend(glob.glob(search_path, recursive=True))
                # Calculate the FID score
                gen_features = calculate_activation_statistics(gen_image_paths, InceptionV3_model, device=fid_device)
                fid_score = calculate_frechet_distance(real_features[0], real_features[1], \
                                                        gen_features[0], gen_features[1])
            print(f'model: {model_name}; fid_score: {fid_score}')
        elif config.sample_mode == 'save':
            x_T = torch.randn([config.eval_batch_size, config.channels, config.img_size, config.img_size]).to(device).float()
            sample = edm_sampler(edm, x_T, num_steps=config.total_steps).detach().cpu()
            save_image((sample/2+0.5).clamp(0, 1), f'{outdir}/image_{model_name}.png')
