import os
import pandas as pd
import tqdm


def get_label(args):
    return 'channel_io_z2zhat'


class Monitor:
    def __init__(self, args):
        self.args = args
        if not getattr(args, 'no_pbar', False):
            self.bar = tqdm.tqdm(range(getattr(args, 'max_epoch', 100)))
        else:
            self.bar = None
        self.csv_data = {}
        self.global_step = 0

    def _update_time(self):
        if self.bar is not None:
            self.bar.update(1)

    def _update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            if key in ('test_loss', 'train_loss', 'best_loss', 'lr'):
                _kwargs[key] = f'{kwargs[key]:.4f}'
            elif key == 'epoch':
                _kwargs[key] = f'{kwargs[key]:d}'
            elif key == 'best_epoch':
                _kwargs[key] = f'{kwargs[key]:d}'
        if self.bar is not None:
            self.bar.set_postfix(**_kwargs)

    def _display(self):
        if self.bar is not None:
            self.bar.display()
            print()

    def step(self, info):
        self._update_time()
        self._update_description(**info)
        self._display()
        self._update_csv(info)
        self.global_step += 1

    @property
    def label(self):
        return get_label(self.args)

    def _update_csv(self, info):
        for key in info:
            try:
                val = float(info[key])
            except (TypeError, ValueError):
                continue
            if key not in self.csv_data:
                self.csv_data[key] = [val]
            else:
                self.csv_data[key].append(val)

    def export_csv(self):
        path = os.path.join(self.args.csv_dir, f'{self.label}.csv')
        if not self.csv_data:
            return
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=False)
