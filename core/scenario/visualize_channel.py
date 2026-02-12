import matplotlib.pyplot as plt
import simulator

def visualize_channel(args):
    # hardcode args
    args.batch_size = 1
    # load data
    train_loader, test_loader = simulator.dataloader.create(args)
    # get one example
    generator = iter(test_loader)
    for i in range(100):
        z_batch, z_hat_batch = next(generator)
        z = z_batch.squeeze()
        z_hat = z_hat_batch.squeeze()
        # find h
        h = z_hat / z
        # plot h
        plt.plot(h.real, label='h.real')
        plt.plot(h.imag, label='h.imag')
        plt.tight_layout()
        plt.legend()
        plt.ylim((-3, 3))
        plt.show()
        plt.cla()
        plt.clf()
