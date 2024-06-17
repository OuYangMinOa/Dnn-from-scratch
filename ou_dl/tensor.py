import numpy as np


class tensor:
    def __init__(self,data):
        self.data = np.array(data)
    def __repr__(self):
        return f"tensor({self.data})"
    def __add__(self, other):
        return tensor(self.data + other.data)
    def __sub__(self, other):
        return tensor(self.data - other.data)
    def __mul__(self, other):
        return tensor(self.data * other.data)
    def __getitem__(self, index):
        return tensor(self.data[index])
    def __setitem__(self, index, value):
        self.data[index] = value.data
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data)
    def __eq__(self, other):
        return np.array_equal(self.data, other.data)
    def __ne__(self, other):
        return not np.array_equal(self.data, other.data)
    def __lt__(self, other):
        return np.less(self.data, other.data)
    def __le__(self, other):
        return np.less_equal(self.data, other.data)
    def __gt__(self, other):
        return np.greater(self.data, other.data)
    def __ge__(self, other):
        return np.greater_equal(self.data, other.data)        
    def __truediv__(self, other):
        return tensor(self.data / other.data)
    def __matmul__(self, other):
        return tensor(self.data @ other.data)
        
    def backward(self, grad, last_w):
        return tensor(grad @ last_w.T)
    def T(self):
        return tensor(self.data.T)
    def sum(self, axis=None):
        return tensor(np.sum(self.data, axis=axis))
    def mean(self, axis=None):
        return tensor(np.mean(self.data, axis=axis))
    def reshape(self, shape):
        return tensor(np.reshape(self.data, shape))
    def flatten(self):
        return tensor(np.flatten(self.data))
    def argmax(self, axis=None):
        return tensor(np.argmax(self.data, axis=axis))
    def max(self, axis=None):
        return tensor(np.max(self.data, axis=axis))