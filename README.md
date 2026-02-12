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
