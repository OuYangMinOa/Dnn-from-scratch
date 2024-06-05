from ou_dl import Module
from ou_dl.Layers import Dnn
from ou_dl.Func import ReLU,Sigmoid,Tanh,Linear
from ou_dl.Opt import Adam, Adagrad

import numpy as np
import matplotlib.pyplot as plt
from ou_dl.Loss import mse


model = Module.seq(
    Dnn(1, 200,activate_function=Sigmoid()),
    Dnn(200, 1,activate_function=Linear()),
)   

n = 256
train_x = np.linspace(0-5,5,n)
train_y = 0.08*np.sin(train_x) + 0.5*np.random.rand(n) * 0.25 + 0.1*train_x


# plt.figure()
# plt.plot(train_x, train_y, "b.")
# plt.show()

model(train_x)
# model.train(train_x, train_y, epochs=1, lr=0.001,batch_size=20)
optimizer = Adam(lr=0.01)
# optimizer = Adagrad(lr=.01)

fig = plt.figure()
for i in range(1000):
    error = model.train(train_x, train_y, epochs=20,optimizer=optimizer,batch_size=64)
    print(f"Epochs {i}, error {error}")
    fig.clear()
    plt.plot(train_x, train_y, "b.")
    plt.plot(train_x, model(train_x), "r-")
    plt.grid()
    plt.xlim([-5.5,5.5])
    plt.ylim([-.6, .6])
    plt.pause(0.002)

plt.show()


