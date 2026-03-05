# SDR Channel in Pytorch

This project sample channel input and output from a pair of PlutoSDR devices.
Then, it fit a deep learning layer to simulate that channel.

# Installation

```bash
pip3 install -r requirements.txt
```
# USAGE

### Step 1: Generate the channel dataset

Generate `--n_prep=10` random complex symbols.
Send through PySDR wired/wireless connection.
Save the channel input and output in `data/dataset/channel_io.npz`

```bash
python3 main.py --scenario=prep_data \
                --n_prep=10 \
                --n_symbol=2048
```

### Step 2: Fit the channel

Model the channel by `y(t) = h(t) * x(t) + n(t)`,
where `x \in \mathbb{C}` is input symbol,
`h(t) = a \exp(j*(2*\pi*f_D*t + \phi))` is channel gain,
where `f_D` is Doppler frequency.
`n \sim \mathcal{C}\mathcal{N}(0, \sigma^2_n)` is AWGN noise, and
`y \in \mathbb{C}` is output symbol.

Then regenerate the dataset and re-run the fading check:
```bash
python main.py --scenario=fit
```

### Result

```
../data
├── csv
├── dataset
│   └── channel_io.npz
├── figure
└── model
    └── doppler.pt # this is the saved layers that simulate the channel
```

Visualize the generated channel gain and noise vs channel gain and noise in dataset
```bash
python main.py --scenario=sample
```
