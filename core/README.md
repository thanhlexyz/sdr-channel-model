# SDR Channel in Pytorch

This project sample channel input and output from a pair of PlutoSDR devices.
Then, it fit a deep learning layer to simulate the real channel.

# Installation

```bash
pip3 install -r requirements.txt
```
# USAGE

### Step 1: Generate the channel dataset

Generate `--n_prep=2000` random complex symbols.
Send through PySDR wired/wireless connection.
Save the channel input and output in `data/dataset/channel_io.npz`

```bash
python3 main.py --scenario=prep_data \
                --n_prep=2000 \
                --n_symbol=256
```

### Step 2: Visualize to understand the channel

Model the channel by `y = h * x + n`,
where `x \in \mathbb{C}` is input symbol,
`h \sim \mathcal{C}\mathcal{N}(0, \sigma^2_h)` is i.i.d Rayleigh fading coefficient,
`n \sim \mathcal{C}\mathcal{N}(0, \sigma^2_n)` is AWGN noise, and
`y \in \mathbb{C}` is output symbol.

But after visualization it shouldn't be i.i.d.
Should fit the kernel to generate moving average.
Then fit the variance.

### Reconfiguring for slow fading

If `check_fading` reports **fast fading**, you can push the channel toward **slow fading** by:

1. **Receiver AGC**
   Use slower or fixed gain so the channel doesn’t change symbol-to-symbol:
   ```bash
   --gain_control_mode=slow_attack   # or manual
   ```
   Default is `fast_attack`, which can make the effective channel look fast-fading.

2. **Higher symbol rate (shorter symbol time)**
   For a given coherence time, more symbols per second ⇒ channel looks “slower” per symbol:
   - Increase sample rate, e.g. `--sample_rate=2000000` (2 MHz), or
   - Use fewer samples per symbol, e.g. `--n_sps=8` (must still work with your sync).

3. **Physical setup**
   Keep the link **static** (no movement), **short range**, and **line-of-sight** to increase coherence time.

Then regenerate the dataset and re-run the fading check:
```bash
python main.py --scenario prep_data --n_prep=2000 --n_symbol=64 \
  --gain_control_mode=slow_attack --sample_rate=2000000
python main.py --scenario check_fading
```

### TODO
- [x] check the implementation closely
- [ ] clean the code, write better readme for publication of this repo
