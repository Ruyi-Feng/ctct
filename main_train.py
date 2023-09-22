from train.params import params
from train.train import Exp


if __name__ == '__main__':
    args = params()
    exp = Exp(args)
    exp.train()
    exp.test()
