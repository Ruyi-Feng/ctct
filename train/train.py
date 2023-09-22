import math
from net.model_atari import GPT
import numpy as np
import os
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from train.dataloader import STAR_Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class Exp:
    def __init__(self, args):
        self.args = args
        self.best_score = None
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.model = self._build_model()

    def _build_model(self):
        model = GPT(self.args).float().to(self.device)
        if os.path.exists(self.args.save_path + 'checkpoint_best.pth'):
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_best.pth', map_location=torch.device(self.device)))
        elif os.path.exists(self.args.save_path + 'checkpoint_last.pth'):
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_last.pth', map_location=torch.device(self.device)))
        return model

    def _get_data(self, batch=None):
        b_sz = batch if batch is not None else self.args.batch_size
        data_set = STAR_Dataset(self.args.data_path, self.args.block_size, self.args.if_total_rtg)
        data_loader = DataLoader(data_set, batch_size=b_sz, drop_last=self.args.drop_last, shuffle=True)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        lamda1 = lambda step: 1 / np.sqrt(max(step , self.args.warmup))
        scheduler = optim.lr_scheduler.LambdaLR(model_optim, lr_lambda=lamda1, last_epoch=-1)
        return model_optim, scheduler

    def _save_model(self, latest_loss, path):
        if self.best_score is None:
            self.best_score = latest_loss
            torch.save(self.model.state_dict(), path)
        elif latest_loss < self.best_score:
            self.best_score = latest_loss
            torch.save(self.model.state_dict(), path)

    def train(self):
        self.tokens = 0 # counter used for learning rate decay
        _, train_loader = self._get_data()
        train_steps = len(train_loader)
        model_optim, scheduler = self._select_optimizer()
        path = self.args.save_path + 'checkpoint_'
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            pbar = tqdm(enumerate(train_loader), total=train_steps)
            for i, (x, y, r, t) in pbar:
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

                if (i + 1) % 50 == 1:
                    pbar.set_description("loss: {0:.7f}, lr: {1:.7f}".format(loss.item(), lr))

            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            self._save_model(train_loss, path + 'best.pth')
        self._save_model(train_loss, path + 'latest.pth')

    def _get_action(self, logits, top_k=None, sample=True):
        logits = logits[:, -1, :] / 1.0
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out

    def test(self):
        _, test_loader = self._get_data(batch=1)
        self.model.eval()
        test_loss = []
        true = 0
        total = 10000
        top_k = self.args.top_k
        for i, (x, y, r, t) in enumerate(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            r = r.to(self.device)
            t = t.to(self.device)
            logits, _ = self.model(x, y, None, r, t)
            tk = self.args.top_k if self.args.top_k != 0 else None
            action = self._get_action(logits, tk, sample=True)
            # print("------test:%d ------"%i)
            # print("pred:", action)
            # print("gdth:", y[:, -1, :])
            if action.item() == y[:, -1, :].item():
                true += 1
            if i > total:
                break
        print("acc: {0:.7f}".format(true/total))
