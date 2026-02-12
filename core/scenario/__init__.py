from .prep_data import prep_data
from .print_channel import print_channel
from .visualize_channel import visualize_channel
from .debug_channel import debug


def train(args):
    """Train z -> z_hat model on channel_io.npz (80/20 split, KL/MSE loss)."""
    from simulator import Trainer
    trainer = Trainer(args)
    trainer.train()
