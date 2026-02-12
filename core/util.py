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
    parser.add_argument('--sample_rate', type=int, default=int(1e6)) # Hz
    parser.add_argument('--carrier_freq', type=int, default=int(915e6)) # Hz
    parser.add_argument('--tx_power', type=float, default=-30) # dB: -90 to 0
    parser.add_argument('--n_clear_buffer', type=int, default=20) #
    # sdr attacker
    parser.add_argument('--transceiver', type=str, default='basic',
                        choices=['basic'])
    # frame structure
    parser.add_argument('--n_preamble', type=int, default=11) # baker's preamble sequence
    parser.add_argument('--n_symbol', type=int, default=64)
    parser.add_argument('--n_sps', type=int, default=16)
    # channel dataset
    parser.add_argument('--n_prep', type=int, default=1000)
    # trainer
    # parser.add_argument('--trainer', type=str, default='classifier')
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--max_epoch', type=int, default=100)
    # parser.add_argument('--lr', type=float, default=3e-4)
    # tester
    # parser.add_argument('--n_test', type=int, default=30)
    # data directory
    parser.add_argument('--dataset_dir', type=str, default=f'../data/dataset')
    parser.add_argument('--figure_dir', type=str, default=f'../data/figure')
    parser.add_argument('--model_dir', type=str, default=f'../data/model')
    parser.add_argument('--csv_dir', type=str, default=f'../data/csv')
    # others
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no_pbar', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # parse args
    args = parser.parse_args()
    # other defaults
    create_folders(args)
    # reproducible
    set_random_seed(args)
    return args
