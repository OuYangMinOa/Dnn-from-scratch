import sys
sys.path.append('../ou_dl')
import matplotlib.pyplot as plt
import numpy as np
import os

from ou_dl.Layers import Dnn, Layers
from ou_dl import Module
from ou_dl.Func import ReLU,Sigmoid,Tanh,Linear, Mish
from ou_dl.Opt import Adam, Adagrad
from ou_dl.Loss import mse

class test(Module.Module):
    def __init__(self):
        super().__init__()
        self.dnn1 = Dnn(1,10,activate_function=ReLU())
        self.dnn2 = Dnn(10,1,activate_function=Linear())

    def forward(self, x):
        x = self.dnn1(x)
        x = self.dnn2(x)
        return x
tt = test()

