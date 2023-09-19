from train.dataloader import STAR_Dataset
from torch.utils.data.dataloader import DataLoader
from train.params import params


def test_dataset(args):
    data_set = STAR_Dataset(args.data_path, args.block_size)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, drop_last=args.drop_last)
    for i, (x, y, r, t) in enumerate(data_loader):
        print(x.shape, y.shape, r.shape, t.shape)
        break

args = params()
test_dataset(args)
