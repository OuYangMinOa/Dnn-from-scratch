from typing import Any
import numpy as np


class ReLU:
    def forward(self, x):
        return np.array(x * (x > 0))
    def back(self, x):
        return np.array(    (x > 0))
    def __call__(self, x):
        return self.forward(x)
    def __repr__(self) -> str:
        return "ReLU"

class Linear:
    def forward(self, x):
        return np.array(x)
    def back(self,x):
        return np.array([1,]*len(x))
    
    def __call__(self, x):
        return self.forward(x)
    def __repr__(self) -> str:
        return "Linear"
    

class Sigmoid:
    def forward(self, x):
        return np.array(1 / (1 + np.exp(-x)))
    def back(self, x):
        return np.array(self.forward(x) * (1 - self.forward(x)))
    def __call__(self, x):
        return self.forward(x)
    def __repr__(self) -> str:
        return "Sigmoid"
    

class Tanh:
    def forward(self, x):
        return np.tanh(x)
    def back(self, x):
        return np.array(1 - self.forward(x) ** 2 )
    def __call__(self, x):
        return self.forward(x)
    def __repr__(self) -> str:
        return "Tanh"