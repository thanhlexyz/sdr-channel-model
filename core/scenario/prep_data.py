import numpy as np
import torch
import tqdm

import pysdr

def prep_data(args):
    # create transceiver
    transmitter = pysdr.transmitter.create('ip:192.168.3.1', args)
    receiver = pysdr.receiver.create('ip:192.168.2.1', args)
    Z, Z_hat = [], []
    for _ in tqdm.tqdm(range(args.n_test)):
        # stop anything transmitting before
        transmitter.stop()
        # transmitter logic
        x, y = transmitter.sample()
        z = transmitter.encode(x)
        # transmit
        transmitter.prepare(z)
        transmitter.transmit()
        z_hat = receiver.receive()
        # receiver logic
        # y_hat = receiver.decode(z_hat)
        Z.append(z)
        Z_hat.append(z_hat)
    # save data
    Z=np.array(Z)
    Z_hat=np.array(Z_hat)
    path = os.path.join(args.dataset_dir, 'channel_io.npz')
    np.savez_compressed(path, Z=Z, Z_hat=Z_hat)
    # stop transmit
    transmitter.stop()
