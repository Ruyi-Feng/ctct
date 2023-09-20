from train.params import params
from train.train import Exp

def test_train():
    args = params()
    exp = Exp(args)
    exp.train()

test_train()
