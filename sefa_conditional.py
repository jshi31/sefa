"""SeFa."""

import os
import pdb
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import json

import torch

import sys
sys.path.append('CoModStyleTrans')

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from utils import HtmlPageVisualizer
from CoModStyleTrans.projector import load_image


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('--source_dir', type=str, help='image directory')
    parser.add_argument('--w_dir', type=str, help='w0 directory')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-K', '--num_semantics', type=int, default=5,
                        help='Number of semantic boundaries corresponding to '
                             'the top-k eigen values. (default: %(default)s)')
    parser.add_argument('--start_distance', type=float, default=-3.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=3.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=11,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--viz_size', type=int, default=256,
                        help='Size of images to visualize on the HTML page. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU(s) to use. (default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.save_dir, exist_ok=True)

    # Factorize weights.
    generator = load_generator(args.model_name)
    gan_type = parse_gan_type(generator) 
    layers, boundaries, values = factorize_weight(generator, args.layer_idx)

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare image with overlap of w path
    # start the glob to grab all the code list
    z_names = [v.split('/')[-1].split('.')[0] for v in sorted(glob(os.path.join(args.w_dir, '*')))]
    z_names = np.random.choice(z_names, args.num_samples, replace=False)
    input_dir = '/home/jshi31/dataset/discover60k/before'
    ws = []
    for name in z_names:
        w_path = os.path.join(args.w_dir, name + '.npz')
        w = np.load(w_path)['w'][:, 0]  # (1, 512)
        ws.append(w)
    ws = np.concatenate(ws, axis=0)  # (N, 512)
    print('totally {} w'.format(len(ws)))

    # Load image
    device = torch.device('cuda')
    sources = [os.path.join(input_dir, name + '.jpg') for name in z_names]
    print('source images', sources)
    sources_uint8 = [load_image(source, generator.img_resolution)[1] for source in sources]
    source_images = [torch.tensor(source_uint8.transpose([2, 0, 1]), device=device).unsqueeze(0).to(torch.float32)/127.5 - 1 for source_uint8 in sources_uint8] # value range (-1, 1)
    source_images = torch.cat(source_images, dim=0)

    # Prepare codes.
    z_space_dim = generator.z_dim if gan_type == 'comodgan' else generator.z_space_dim
    codes = torch.randn(args.num_samples, z_space_dim).cuda()
    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type in ['stylegan', 'stylegan2']:
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes,
                                     trunc_psi=args.trunc_psi,
                                     trunc_layers=args.trunc_layers)
    elif gan_type == 'comodgan':
        codes = torch.from_numpy(ws).float().to(device)
        # codes = generator.mapping(codes, None)
    codes = codes.detach().cpu().numpy()

    # Generate visualization pages.
    distances = np.linspace(args.start_distance,args.end_distance, args.step)
    num_sam = args.num_samples
    num_sem = args.num_semantics
    vizer_1 = HtmlPageVisualizer(num_rows=num_sem * (num_sam + 1),
                                 num_cols=args.step + 1,
                                 viz_size=args.viz_size)
    vizer_2 = HtmlPageVisualizer(num_rows=num_sam * (num_sem + 1),
                                 num_cols=args.step + 1,
                                 viz_size=args.viz_size)

    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer_1.set_headers(headers)
    vizer_2.set_headers(headers)
    for sem_id in range(num_sem):
        value = values[sem_id]
        vizer_1.set_cell(sem_id * (num_sam + 1), 0,
                         text=f'Semantic {sem_id:03d}<br>({value:.3f})',
                         highlight=True)
        for sam_id in range(num_sam):
            vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, 0,
                             text=f'Sample {sam_id:03d}')
    for sam_id in range(num_sam):
        vizer_2.set_cell(sam_id * (num_sem + 1), 0,
                         text=f'Sample {sam_id:03d}',
                         highlight=True)
        for sem_id in range(num_sem):
            value = values[sem_id]
            vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, 0,
                             text=f'Semantic {sem_id:03d}<br>({value:.3f})')

    for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
        code = codes[sam_id:sam_id + 1]
        for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
            boundary = boundaries[sem_id:sem_id + 1]
            for col_id, d in enumerate(distances, start=1):
                temp_code = code.copy()
                if gan_type == 'pggan':
                    temp_code += boundary * d
                    image = generator(to_tensor(temp_code))['image']
                elif gan_type in ['stylegan', 'stylegan2']:
                    temp_code[:, layers, :] += boundary * d
                    image = generator.synthesis(to_tensor(temp_code))['image']
                elif gan_type == 'comodgan':
                    temp_code[:, layers, :] += boundary * d
                    image = generator.synthesis(source_images[sam_id:sam_id+1], to_tensor(temp_code))
                image = postprocess(image)[0]
                vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, col_id,
                                 image=image)
                vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, col_id,
                                 image=image)

    prefix = (f'{args.model_name}_'
              f'N{num_sam}_K{num_sem}_L{args.layer_idx}_seed{args.seed}')
    vizer_1.save(os.path.join(args.save_dir, f'{prefix}_sample_first.html'))
    vizer_2.save(os.path.join(args.save_dir, f'{prefix}_semantic_first.html'))


if __name__ == '__main__':
    main()