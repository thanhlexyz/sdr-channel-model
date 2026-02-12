"""
Analyze channel_io.npz (z, z_hat) to classify slow vs fast fading.
Uses temporal correlation of h = z_hat/z along the symbol axis (64): high correlation -> slow fading.
"""
import numpy as np
import os


def _autocorr_lag1(x):
    """Sample correlation between x[:, :-1] and x[:, 1:] along axis 1, averaged over axis 0."""
    a, b = x[:, :-1], x[:, 1:]
    a = a - np.mean(a, axis=1, keepdims=True)
    b = b - np.mean(b, axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a**2, axis=1)) * np.sqrt(np.sum(b**2, axis=1))
    den = np.where(den <= 0, np.nan, den)
    r = num / den
    return np.nanmean(r)


def _coherence_length(x, threshold=0.5):
    """Mean number of steps (along axis 1) until correlation with first sample drops below threshold."""
    n_frames, n_symbol = x.shape
    if n_symbol < 2:
        return float("inf")
    first = x[:, 0:1]  # (N, 1)
    first = first - np.mean(first)
    norms_first = np.sqrt(np.sum(first**2, axis=1, keepdims=True))
    lengths = []
    for lag in range(1, n_symbol):
        y = x[:, lag : lag + 1]
        y = y - np.mean(y)
        num = np.sum(first * y, axis=1)
        den = norms_first.squeeze() * np.sqrt(np.sum(y**2, axis=1))
        den = np.where(den <= 0, np.nan, den)
        r = num / den
        still_corr = np.nanmean(r) >= threshold
        if not still_corr:
            lengths.append(lag)
            break
    return lengths[0] if lengths else n_symbol


def check_fading(args):
    path = os.path.join(args.dataset_dir, "channel_io.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. Generate it with --scenario prep_data.")

    data = np.load(path)
    Z = data["Z"]       # (N, 64) complex
    Z_hat = data["Z_hat"]  # (N, 64) complex

    eps = 1e-12
    H = Z_hat / (Z + eps * (np.abs(Z) < 1e-10))

    h_real = np.real(H)
    h_imag = np.imag(H)

    r_real = _autocorr_lag1(h_real)
    r_imag = _autocorr_lag1(h_imag)
    r_mean = 0.5 * (r_real + r_imag)

    # Normalized step-wise change: E[|h_{t+1} - h_t|^2] / E[|h_t|^2]
    dh = np.diff(H, axis=1)
    step_var = np.mean(np.abs(dh) ** 2)
    h_var = np.mean(np.abs(H) ** 2)
    normalized_change = step_var / (h_var + 1e-20)

    # Coherence length in symbols (how many symbols until corr drops below 0.5)
    coh_real = _coherence_length(h_real)
    coh_imag = _coherence_length(h_imag)
    coh_symbols = min(coh_real, coh_imag)

    # Classify: slow fading = high lag-1 correlation (e.g. > 0.6) and low normalized change
    slow_fading = r_mean > 0.6 and normalized_change < 0.5

    print("Channel fading analysis (from channel_io.npz, h = z_hat/z)")
    print(f"  Lag-1 correlation (real): {r_real:.4f}")
    print(f"  Lag-1 correlation (imag): {r_imag:.4f}")
    print(f"  Lag-1 correlation (mean): {r_mean:.4f}")
    print(f"  Normalized step change E[|Δh|^2]/E[|h|^2]: {normalized_change:.4f}")
    print(f"  Coherence length (symbols, corr≥0.5): ~{coh_symbols}")
    print()
    if slow_fading:
        print("  -> SLOW FADING (channel highly correlated across symbols)")
    else:
        print("  -> FAST FADING (channel changes appreciably symbol-to-symbol)")
