import os
import torch

from simulator.model import Model


def print_channel(args):
    """Print channel parameters (h and n) from the Rayleigh+AWGN model.
    Loads from checkpoint if present, otherwise uses default inits.
    """
    model_dir = os.path.join(args.model_dir, 'channel_io_rayleigh_awgn')
    path = os.path.join(model_dir, 'model_best.pth.tar')

    model = Model(args).to(args.device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print(f'[+] Loaded channel from {path}')
    else:
        print('[+] No checkpoint found; using default parameters.')

    with torch.no_grad():
        sigma = model.sigma.item()
        h_std_real = model.h_std_real.item()
        h_std_imag = model.h_std_imag.item()
        h_mean_real = model.h_mean_real.item()
        h_mean_imag = model.h_mean_imag.item()

    print('--- Channel parameters ---')
    print('  h (Rayleigh fading):')
    print(f'    h_mean_real = {h_mean_real:+.6f}')
    print(f'    h_mean_imag = {h_mean_imag:+.6f}')
    print(f'    h_std_real  = {h_std_real:.6f}')
    print(f'    h_std_imag  = {h_std_imag:.6f}')
    print('  n (AWGN):')
    print(f'    sigma       = {sigma:.6f}  (noise std; variance σ² = {sigma**2:.6f})')
