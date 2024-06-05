import numpy as np

from typing import Any
from ou_dl.Func import Linear

class Dnn:
    def __init__(self,input_shape=None, output_shape=None, activate_function=Linear(), bias = True) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight = np.random.randn( input_shape, output_shape)
        # self.weight = np.zeros( (input_shape, output_shape))
        self.is_bias = bias
        self.name = "Dnn"
        if (bias):
            self.bias = np.random.randn(output_shape)
        else:
            self.bias = np.zeros(output_shape)

        self.activate_function = activate_function
        self._last_input  = np.zeros(input_shape)
        self._last_output = np.zeros(output_shape)

    def set_weight_zero(self):
        self.weight = np.zeros(self.weight.shape)
        self.bias   = np.zeros(self.bias.shape)

    def forward(self, x):
        if (len(x.shape) ==1 ):
            x = x.reshape(-1,self.input_shape) 

        output = self.activate_function(x @ self.weight + np.tile(self.bias,(x.shape[0],1)))

        self._last_input  = np.array(x)
        self._last_output = np.array(output)
        return np.array(output)
    
    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self) -> str:
        return f"Dnn(input_shape={self.input_shape}, output_shape={self.output_shape}, activate_function={self.activate_function})"
    
    def back(self,d_err,optimizer): 
        """Implement back propagation algorithm
        """
        
        ## w_new = w_old - alpha * dj/dw
        total_par_w = np.zeros(self.weight.shape)
        total_par_b = np.zeros(self.bias.shape)
        batch_size = self._last_input.shape[0]
        batch_weight_total  = []
        for i in range(batch_size):
            this_act   = self.activate_function.back(self._last_output[i,:])
            # print("act",this_act.shape, d_err[i].shape,self.output_shape)

            par_b = np.array([ this_act[j] * np.sum(d_err[i][j] ) for j in range(self.output_shape) ])

            # print("parb",par_b.shape,self.output_shape,self.bias.shape)

            par_w = np.atleast_2d(self._last_input[i,:]).T @ np.atleast_2d(par_b)
            # print(par_w)

            alpha_b = optimizer.step_b(par_b,self.name )
            alpha_w = optimizer.step(  par_w,self.name )

            total_par_w += alpha_w
            total_par_b += alpha_b

            batch_weight_total.append(self.weight - total_par_w)

            # self.weight = self.weight - alpha_w 
            # # print(alpha_w)
            # if (self.is_bias):
            #     self.bias   = self.bias - alpha_b 
                
        # print(np.mean(total_par_w))。
        

        self.weight = self.weight - total_par_w / batch_size
        if (self.is_bias):
            self.bias   = self.bias   - total_par_b / batch_size
        
        return np.array(batch_weight_total).reshape(batch_size,*self.weight.shape)