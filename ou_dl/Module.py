
from typing import Any
from ou_dl.Loss import mse
from tqdm.auto import tqdm
from ou_dl.Layers import Layers
import numpy as np
class Module:
    def __init__(self) -> None:
        self.layers = {}
        
    def __call__(self, x) -> Any:
        return self.forward(x)

    def forward(self, x) -> Any:
        raise NotImplementedError

    def _set_layers_name(self):
        self.layer_count = 0 
        for each in self.__dict__:
            this_obj = self.__dict__[each]
            print(isinstance(this_obj, Layers))
            if isinstance(this_obj, Layers):
                self.layer_count += 1
                this_obj.name = this_obj.__repr__() + "_" + str(self.layer_count)
                self.layers[this_obj.name] = this_obj

    def parameters(self):
        self._set_layers_name()
        return {name:layer.parameters() for name,layer in self.layers.items()} 

class Seq(Module, Layers):
    def __init__(self,*layers):
        self.layers = layers
        self._set_layers_name()

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            # print(i,layer.weight)
            x = layer.forward(x)
        return x

    def __repr__(self) -> str:
        output = "\n"
        for layer in self.layers:
            output += layer.__repr__() + "\n"
        return output

    ## In future, this will be move to tensor
    def back(self,err,optimizer):
        last_w = np.tile(np.identity(err.shape[1]),(err.shape[0],1,1) )
        for layer in reversed(self.layers):
            last_w = layer.back(err,last_w,optimizer)
        return err
    
    def train(self,x,y,optimizer,epochs=1,batch_size=32,shuffle=False,error_func=mse()):
        self.error_func = error_func
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
        return self.error_func(self.forward(x), y)
            
    def shuffle_data(self,x, y):
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        return x[idx], y[idx]