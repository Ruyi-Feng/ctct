import math
from net.model_atari import GPT
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from train.dataloader import STAR_Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time


class Train:
    def __init__(self, args):
        self.args = args
        self.best_score = None
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.model = self._build_model()

    def _build_model(self):
        model = GPT(self.args).float().to(self.device)
        return model

    def _get_data(self):
        data_set = STAR_Dataset(self.args.data_path, self.args.block_size, self.args.if_total_rtg)
        data_loader = DataLoader(data_set, batch_size=self.args.batch_size, drop_last=self.args.drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        lamda1 = lambda step: 1 / np.sqrt(max(step , self.args.warmup))
        scheduler = optim.lr_scheduler.LambdaLR(model_optim, lr_lambda=lamda1, last_epoch=-1)
        return model_optim, scheduler

    def _save_model(self, latest_loss, path):
        if self.best_score is None:
            self.best_score = latest_loss
            torch.save(self.model.module.state_dict(), path)
        elif latest_loss < self.best_score:
            self.best_score = latest_loss
            torch.save(self.model.module.state_dict(), path)

    def train(self):
        self.tokens = 0 # counter used for learning rate decay
        _, train_loader = self._get_data()
        train_steps = len(train_loader)
        model_optim, scheduler = self._select_optimizer()
        path = self.args.save_path + 'checkpoint_'
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            iter_count = 0
            pbar = tqdm(enumerate(train_loader), total=train_steps)
            for i, (x, y, r, t) in pbar:
                iter_count += 1
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                logits, loss = self.model(x, y, y, r, t)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                train_loss.append(loss.item())

                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)
                model_optim.step()
                scheduler.step()

                if self.args.lr_decay:
                    self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < self.args.warmup:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, self.args.warmup))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - self.args.warmup) / float(max(1, self.args.final_tokens - self.args.warmup))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.args.learning_rate * lr_mult
                    for param_group in model_optim.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = -1

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            self._save_model(train_loss, path + 'best.pth')
        self._save_model(train_loss, path + 'latest.pth')
