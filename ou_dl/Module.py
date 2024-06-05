
from typing import Any
from ou_dl.Loss import mse
from tqdm.auto import tqdm

import numpy as np


class seq:
    def __init__(self,*layers,error_func = mse()):
        self.layers = layers
        self.error_func = error_func
        self._set_layers_name()

    def _set_layers_name(self):
        for i,layer in enumerate(self.layers):
            layer.name = f"{i}_layers"

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            # print(i,layer.weight)
            x = layer.forward(x)

        return x

    def __call__(self, x) -> Any:
        return self.forward(x)

    def __repr__(self) -> str:
        output = "\n"
        for layer in self.layers:
            output += layer.__repr__() + "\n"
        return output

    def back(self,x,lr):
        for layer in reversed(self.layers):
            x = layer.back(x,lr)
        return x
    
    def train(self,x,y,optimizer,epochs=1,batch_size=32,shuffle=False):
        if shuffle:
            x, y = self.shuffle_data(x, y)

        if optimizer.iter==0:
            for i,layer in enumerate(self.layers):
                optimizer._init_momentum(layer.name, layer.weight.shape, layer.bias.shape)

        for epochs in range(1,epochs+1):
            for i in range(0,len(x)//batch_size):
                pred = self.forward(x[i*batch_size:(i+1)*batch_size])
                error = self.error_func(pred, y[i*batch_size:(i+1)*batch_size] )
                d_err  = self.error_func.back(pred, y[i*batch_size:(i+1)*batch_size].reshape(pred.shape))

                self.back(d_err,optimizer)

        return error
            
    def shuffle_data(self,x, y):
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        return x[idx], y[idx]