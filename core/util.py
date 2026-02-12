import numpy as np
import argparse
import random
import torch
import os

def create_folders(args):
    ls = [args.csv_dir, args.figure_dir, args.dataset_dir, args.model_dir]
    for folder in ls:
        if not os.path.exists(folder):
            os.makedirs(folder)

def set_random_seed(args):
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='ideal')
    # sdr tx/rx
    parser.add_argument('--sample_rate', type=int, default=int(2e6)) # Hz
    parser.add_argument('--carrier_freq', type=int, default=int(915e6)) # Hz
    parser.add_argument('--tx_power', type=float, default=-30) # dB: -90 to 0
    parser.add_argument('--n_clear_buffer', type=int, default=20) #
    parser.add_argument('--gain_control_mode', type=str, default='manual',
                        choices=['fast_attack', 'slow_attack', 'manual'],
                        help='Rx AGC: fast_attack=fast fading, slow_attack/manual=slow fading')
    # sdr attacker
    parser.add_argument('--transceiver', type=str, default='basic',
                        choices=['basic'])
    # frame structure
    parser.add_argument('--n_preamble', type=int, default=11) # baker's preamble sequence
    parser.add_argument('--n_symbol', type=int, default=256)
    parser.add_argument('--n_sps', type=int, default=8)
    # channel dataset
    parser.add_argument('--n_prep', type=int, default=2000)
    # trainer (z -> z_hat)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=100)
    # AR channel model (h = z_hat/z)
    parser.add_argument('--ma_window', type=int, default=8, help='Moving average window L for past h')
    parser.add_argument('--ar_hidden', type=int, default=128, help='Hidden size for AR channel model')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # data directory
    parser.add_argument('--dataset_dir', type=str, default=f'../data/dataset')
    parser.add_argument('--figure_dir', type=str, default=f'../data/figure')
    parser.add_argument('--model_dir', type=str, default=f'../data/model')
    parser.add_argument('--csv_dir', type=str, default=f'../data/csv')
    # others
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_pbar', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug_idx', type=int, default=0, help='Example index for debug scenario')
    # parse args
    args = parser.parse_args()
    # other defaults
    create_folders(args)
    # reproducible
    set_random_seed(args)
    return args
