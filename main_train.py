from train.params import params
from train.train import Train


if __name__ == '__main__':
    args = params()
    exp = Train(args)
    exp.train()
