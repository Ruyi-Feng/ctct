
from net.model_atari import GPT
from train.dataloader import STAR_Dataset
from torch.utils.data.dataloader import DataLoader
from train.params import params


def test_net():
    args = params()
    model = GPT(args)
    model.train()
    data_set = STAR_Dataset(args.data_path, args.block_size)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, drop_last=args.drop_last)
    for i, (x, y, r, t) in enumerate(data_loader):
        logits, loss = model(x, y, y, r, t)
        print("loss: {}".format(loss.item()))
        model.zero_grad()
        loss.backward()
        if i > 10:
            break

test_net()
