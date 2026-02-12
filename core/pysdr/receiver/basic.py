import numpy as np
try:
    import adi
except ImportError:
    # import warnings
    # warnings.warn('no plutosdr device found', UserWarning)
    pass

from pysdr import util

class Receiver:

    def __init__(self, name, args):
        # save args
        self.args = args
        # initialize sdr
        sdr = adi.Pluto(name)
        # configure
        sdr.rx_lo                   = args.carrier_freq
        sdr.sample_rate             = args.sample_rate
        sdr.rx_buffer_size          = (args.n_symbol + args.n_preamble) * args.n_sps * 3 # x2 to fit the whole frame within buffer
        sdr.rx_rf_bandwidth         = args.sample_rate
        sdr.gain_control_mode_chan0 = 'fast_attack'
        # save sdr
        self.sdr = sdr

    def receive(self):
        # receive and decode x_hat: np.array float32
        # extract args
        args = self.args
        sdr = self.sdr
        # extract preamble
        preamble = util.get_preamble()
        # clear buffer
        for _ in range(args.n_clear_buffer):
            sdr.rx()
        # receive samples
        samples = sdr.rx()
        # normalize
        samples = samples / np.max(np.abs(samples))
        # util.plot_constellation(samples, f'rx_0_normalization', args)
        # time synchronization
        samples = util.mueller_clock_recovery(samples, args)
        # util.plot_constellation(samples, f'rx_1_mueller_clock_recovery', args)
        # print(f'mueller_clock_recovery: {samples.shape=}')
        # frame synchronization
        symbols = util.frame_sync(samples, preamble, args)
        # util.plot_constellation(symbols, f'rx_2_frame_sync', args)
        # print(f'frame_sync: {symbols.shape=}')
        return symbols

    def decode(self, z_hat):
        y = z_hat
        return y

    def evaluate(self, y, y_hat):
        info = {
            'mse': np.sqrt(((y - y_hat) ** 2).sum())
        }
        return info
