import matplotlib.pyplot as plt
import numpy as np
import random
import struct
import os

def apply_pulse_shaping(symbols, n_sps=16):
    # Apply raised cosine pulse shaping
    # create raised cosine filter
    num_taps = 101
    beta = 0.35  # Roll-off factor
    Ts = n_sps
    t = np.arange(-50, 51)
    h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)
    h[np.isnan(h)] = np.sinc(0) * np.cos(0) / (1 - 0)  # Fix NaN at t=±Ts/(2β)
    # upsample symbols
    upsampled = np.zeros(len(symbols) * n_sps, dtype=np.complex64)
    upsampled[::n_sps] = symbols
    # apply filter
    shaped = np.convolve(upsampled, h, mode='same')
    return shaped

def frame_sync(samples, preamble, args):
    # print(f'[frame_sync] input samples {len(samples)=}')
    # Frame synchronization using preamble detection with phase correction
    # cross-correlate with known preamble
    corr = np.abs(np.correlate(samples, preamble, mode='full'))
    corr = corr[:-(args.n_symbol + args.n_preamble)] # avoid decoding incomplete frame at the end
    peak_idx = np.argmax(corr)
    # find frame start
    frame_start = peak_idx - args.n_preamble + 1
    if frame_start < 0:
        frame_start = 0
    # extract detected preamble for verification
    detected_preamble_samples = samples[frame_start:frame_start + args.n_preamble]
    # convert to bits for comparison (demodulate detected preamble)
    expected_preamble_bits = ['0' if s >= 0 else '1' for s in preamble]
    detected_preamble_bits = ['0' if np.real(s) >= 0 else '1' for s in detected_preamble_samples]
    # print(f'[frame_sync] expected_preamble_bits: {"".join(expected_preamble_bits)}')
    # print(f'[frame_sync] detected_preamble_bits:  {"".join(detected_preamble_bits)}')
    # calculate bit error rate for preamble
    matches = sum(1 for e, d in zip(expected_preamble_bits, detected_preamble_bits) if e == d)
    ber = 1 - (matches / len(expected_preamble_bits))
    # print(f"[frame_sync] preamble BER: {ber:.3f} ({matches}/{len(expected_preamble_bits)} bits correct)")
    # check for phase inversion (180° phase ambiguity in BPSK)
    phase_inverted = False
    if ber > 0.5:  # If more than half the bits are wrong, likely phase inversion
        # check if detected bits are the inverse of expected bits
        inverted_matches = sum(1 for e, d in zip(expected_preamble_bits, detected_preamble_bits) if e != d)
        if inverted_matches == len(expected_preamble_bits):
            phase_inverted = True
            # print(f"[frame_sync] Phase inversion detected!")
    # extract data portion (skip preamble)
    data_start = frame_start + args.n_preamble
    data_end   = data_start + args.n_symbol
    if data_end <= len(samples):
        data_samples = samples[data_start:data_end]
    else:
        data_samples = samples[data_start:]
    # apply phase correction if needed
    if phase_inverted:
        # invert all symbols to correct phase
        data_samples = -data_samples
        # print(f"[frame_sync] Correcting all symbols as phase inversion detected...")
    return data_samples

def mueller_clock_recovery(samples, args):
    # extract args
    n_sps = args.n_sps
    # Mueller and Muller clock recovery
    mu = 0
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)
    i_in = 0
    i_out = 2
    # more conservative loop condition to preserve more symbols
    while i_out < len(out) - 1 and i_in < len(samples) - n_sps:
        out[i_out] = samples[i_in]
        out_rail[i_out] = (np.real(out[i_out]) > 0) + 1j*(np.imag(out[i_out]) > 0)
        if i_out >= 2:
            x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
            y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
            mm_val = np.real(y - x)
            mu += n_sps + 0.3 * mm_val
        else:
            mu += n_sps  # Initial advance without correction
        i_in += int(np.floor(mu))
        mu = mu - np.floor(mu)
        i_out += 1
    result = out[2:i_out]
    # print(f'[mueller_clock_recovery] {len(samples)=} {len(result)=} (reduction factor={len(samples)/len(result):.2f})')
    return result

def get_preamble():
    return np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])  # 11-bit Barker code

def plot_constellation(symbols, name, args):
    plt.scatter(np.real(symbols), np.imag(symbols), color='blue')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    path = os.path.join(args.figure_dir, f'{name}_constellation.pdf')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.tight_layout()
    plt.savefig(path)
    plt.cla()
    plt.clf()
