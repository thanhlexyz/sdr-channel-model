import os
import numpy as np
import matplotlib.pyplot as plt


def debug(args):
    """Debug scenario: one (z, z_hat) from the dataset; compute and print ground-truth h and n;
    visualize z, z_hat, and n for that example only.
    Assumes z_hat = h*z + n with one scalar h and vector n (slow fading).
    """
    path_data = os.path.join(args.dataset_dir, 'channel_io.npz')
    if not os.path.exists(path_data):
        raise FileNotFoundError(f'Dataset not found: {path_data}')

    data = np.load(path_data)
    Z = np.asarray(data['Z'], dtype=np.complex128)
    Z_hat = np.asarray(data['Z_hat'], dtype=np.complex128)

    # One example: first sample
    idx = getattr(args, 'debug_idx', 0)
    z = Z[idx]       # (64,)
    z_hat = Z_hat[idx]

    # Ground truth: z_hat = h*z + n. Solve for h (one scalar) s.t. n = z_hat - h*z has minimal power.
    # Least squares: h = (z.conj() @ z_hat) / (z.conj() @ z)
    print(z_hat.shape, z.shape)
    for i in range(len(z)):
        h_i = z_hat[i] / z[i]
        print(f'{i=} {h_i=}')

    z_dot = np.vdot(z, z)
    if np.abs(z_dot) < 1e-12:
        h_gt = 0.0 + 0.0j
    else:
        h_gt = np.vdot(z, z_hat) / z_dot
    n_gt = z_hat - h_gt * z
    print(h_gt.shape)
    exit()

    # Print
    print('--- Debug: one example (ground truth only) ---')
    print(f'  Example index: {idx}')
    print(f'  h (scalar, slow fading): {h_gt.real:.6f} + j*{h_gt.imag:.6f}')
    print(f'  |h| = {np.abs(h_gt):.6f}, angle = {np.angle(h_gt):.6f} rad')
    print(f'  n (length-64): mean = {np.mean(n_gt):.6f}, std = {np.std(n_gt):.6f}, |n|² = {np.sum(np.abs(n_gt)**2):.6f}')
    print(f'  z:  mean = {np.mean(z):.6f}, std = {np.std(z):.6f}')
    print(f'  z_hat: mean = {np.mean(z_hat):.6f}, std = {np.std(z_hat):.6f}')
    print(f'  Check z_hat - (h*z + n) = {np.max(np.abs(z_hat - (h_gt * z + n_gt))):.2e} (should be ~0)')

    # Visualize ground truth only: one figure with z, z_hat, n (constellation)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def scatter_one(ax, c, title, color='C0'):
        ax.scatter(c.real, c.imag, s=20, alpha=0.8, c=color, edgecolors='k', linewidths=0.3)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imag')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

    scatter_one(axes[0], z, 'z (input, 64 symbols)')
    scatter_one(axes[1], z_hat, 'z_hat (output, 64 symbols)')
    scatter_one(axes[2], n_gt, 'n = z_hat - h*z (noise, 64 symbols)')

    plt.tight_layout()
    out_path = os.path.join(args.figure_dir, 'debug_channel_ground_truth.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved {out_path}')
