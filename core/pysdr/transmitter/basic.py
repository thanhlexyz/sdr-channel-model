import numpy as np
import time
try:
    import adi
except ImportError:
    # import warnings
    # warnings.warn('no plutosdr device found', UserWarning)
    pass


from pysdr import util

class Transmitter:

    def __init__(self, name, args):
        # save args
        self.args = args
        # initialize sdr
        sdr = adi.Pluto(name)
        # configure
        sdr.sample_rate           = args.sample_rate
        sdr.tx_rf_bandwidth       = args.sample_rate
        sdr.tx_lo                 = args.carrier_freq
        sdr.tx_hardwaregain_chan0 = args.tx_power
        sdr.tx_cyclic_buffer = True
        # save sdr
        self.sdr = sdr

    def sample(self):
        # extract args
        args = self.args
        # sample random input
        real = np.ones(args.n_symbol)
        imag = np.zeros(args.n_symbol)
        # real = np.random.rand(args.n_symbol) * 2 - 1
        # imag = np.random.rand(args.n_symbol) * 2 - 1
        x = real + 1j * imag
        x /= np.linalg.norm(x)
        return x, x

    def encode(self, x):
        z = x
        return z

    def prepare(self, z):
        # z: np.array, complex64
        # extract args
        args = self.args
        sdr = self.sdr
        # extract preamble
        preamble = util.get_preamble()
        # create frame
        symbols = np.concatenate([preamble, z])
        # util.plot_constellation(symbols, f'tx_0_symbols', args)
        # print(f'transmit: {symbols.shape=}')
        # apply pulse shaping
        samples = util.apply_pulse_shaping(symbols, n_sps=args.n_sps)
        # print(f'{preamble.shape=} {symbols.shape=} {samples.shape=}')
        # scale for PlutoSDR (16-bit DAC)
        samples *= 2**14
        # save for transmit
        self.samples = samples.astype(np.complex64)

    def transmit(self):
        # print(f'[transmiter] transmitting {x=}')
        # print(f'    - using  {len(samples)=}')
        # print(f'[transmitter] transmit at {time.time():0.1f}')
        self.sdr.tx(self.samples)

    def stop(self):
        # stop transmission and cleanup
        self.sdr.tx_destroy_buffer()
        # print("[transmitter] stopped transmission")
