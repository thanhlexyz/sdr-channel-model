import os
import torch
import torch.nn as nn

from . import dataloader
from .model import Model
from .monitor import Monitor


class Trainer:
    """Train the channel Model (Rayleigh fading + AWGN) so that the distribution of
    generated z_hat = Model(z) matches labeled z_hat, by minimizing NLL (equiv. KL)."""

    def __init__(self, args):
        self.args = args
        self.train_loader, self.test_loader = dataloader.create(args)
        self.model = Model(args).to(args.device)
        self.monitor = Monitor(args)
        self._create_other()

    def _create_other(self):
        args = self.args
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(args, 'lr', 3e-4),
            weight_decay=getattr(args, 'weight_decay', 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=getattr(args, 'max_epoch', 100),
            eta_min=1e-5,
        )
        model_dir = os.path.join(args.model_dir, 'channel_io_rayleigh_awgn')
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.best_loss = float('inf')
        self.best_epoch = -1

    def train_epoch(self, epoch):
        args = self.args
        self.model.train()
        running_nll = 0.0
        n_samples = 0
        for batch_id, (z, z_hat) in enumerate(self.train_loader):
            z = z.to(args.device)
            z_hat = z_hat.to(args.device)
            log_p = self.model.log_prob_z_hat_given_z(z, z_hat)
            nll = -log_p.mean()
            self.optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_nll += nll.item() * z.size(0)
            n_samples += z.size(0)
        info = {'train_loss': running_nll / n_samples}
        return info

    @torch.no_grad()
    def test_epoch(self, epoch):
        args = self.args
        self.model.eval()
        running_nll = 0.0
        n_samples = 0
        for batch_id, (z, z_hat) in enumerate(self.test_loader):
            z = z.to(args.device)
            z_hat = z_hat.to(args.device)
            log_p = self.model.log_prob_z_hat_given_z(z, z_hat)
            nll = -log_p.mean()
            running_nll += nll.item() * z.size(0)
            n_samples += z.size(0)
        test_loss = running_nll / n_samples
        info = {'test_loss': test_loss}
        return info

    def get_info(self, epoch):
        lrl = [pg['lr'] for pg in self.optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        return {'epoch': epoch, 'lr': lr}

    def save(self, epoch, info):
        test_loss = info.get('test_loss', float('inf'))
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_epoch = epoch
            path = os.path.join(self.model_dir, 'model_best.pth.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
            }, path)
        return {'best_loss': self.best_loss, 'best_epoch': self.best_epoch}

    def train(self):
        args = self.args
        max_epoch = getattr(args, 'max_epoch', 100)
        for epoch in range(max_epoch):
            info = self.get_info(epoch)
            info.update(self.train_epoch(epoch))
            info.update(self.test_epoch(epoch))
            info.update(self.save(epoch, info))
            self.monitor.step(info)
            self.monitor.export_csv()
            self.scheduler.step()

    def load(self):
        args = self.args
        path = os.path.join(self.model_dir, 'model_best.pth.tar')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=args.device, weights_only=False)
            if 'state_dict' in ckpt:
                self.model.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt)
            self.model.to(args.device)
            self.model.eval()
            print(f'[+] loaded trainer from {path}')
