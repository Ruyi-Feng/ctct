from train.params import params
from train.train import Train

def test_train():
    args = params()
    exp = Train(args)
    exp.train()

test_train()
