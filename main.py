from ou_dl import Module
from ou_dl.Layers import Dnn
from ou_dl.Func import ReLU,Sigmoid,Tanh,Linear
from ou_dl.Opt import Adam, Adagrad

import numpy as np
import matplotlib.pyplot as plt
from ou_dl.Loss import mse

np.random.seed(8789)

model = Module.seq(
    Dnn(1, 5,activate_function=ReLU()),
    Dnn(5, 5,activate_function=ReLU()),
    Dnn(5, 5,activate_function=ReLU()),
    Dnn(5, 5,activate_function=ReLU()),
    Dnn(5, 1,activate_function=Linear()),
)   

n = 256
train_x = np.linspace(0,10,n)
train_y = 0.5*np.sin(train_x) + 0.5*np.random.rand(n) * 0.25 + 0.1*train_x


# plt.figure()
# plt.plot(train_x, train_y, "b.")
# plt.show()

model(train_x)
# model.train(train_x, train_y, epochs=1, lr=0.001,batch_size=20)
optimizer = Adam(lr=0.001)
# optimizer = Adagrad(lr=.01)

fig,ax  = plt.subplots(1)
for i in range(100000):
    error = model.train(train_x, train_y, epochs=20,optimizer=optimizer,batch_size=32)
    print(f"Epochs {i}, error {error}")
    fig.clear()
    plt.plot(train_x, train_y, "b.")
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())
    plt.plot(train_x, model(train_x), "r-")
    plt.grid()
    x_w,y_w = max(train_x) - min(train_x), max(train_y) - min(train_y)
    plt.xlim([min(train_x) - x_w * 0.05, max(train_x) + x_w * 0.05])
    plt.ylim([min(train_y) - y_w * 0.05, max(train_y) + y_w * 0.05])

    plt.pause(0.002)

plt.show()


