# mean square error
import numpy as np


class mse:
    def __call__(self,x,y):
        return self.forward(x, y)
    def forward(self,x,y):
        return np.mean((x - y) ** 2)
    def back(self, x, y):
        return 2 * (x - y) / x.size
    

class sqaure_error:
    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        return np.sum((x - y) ** 2)

    def back(self, x, y):
        return 2 * (x - y) / x.size 