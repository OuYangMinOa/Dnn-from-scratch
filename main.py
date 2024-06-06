from ou_dl import Module
from ou_dl.Layers import Dnn
from ou_dl.Func import ReLU,Sigmoid,Tanh,Linear
from ou_dl.Opt import Adam, Adagrad

import numpy as np
import matplotlib.pyplot as plt

from ou_dl.Loss import mse

import torch

np.random.seed(8789)

torch.set_default_dtype(torch.float64)

torch_model = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
torch_model.zero_grad() 

model = Module.seq(
    Dnn(1, 32,activate_function=Sigmoid()),
    Dnn(32, 32,activate_function=Sigmoid()),
    Dnn(32, 1,activate_function=Linear()),
)   

n = 256
train_x = np.linspace(0,5,n)
train_y = np.sin(train_x) + 0.5*np.random.rand(n)

# plt.figure()
# plt.plot(train_x, train_y, "b.")
# plt.show()

# model.train(train_x, train_y, epochs=1, lr=0.001,batch_size=20)
# optimizer = Adam(lr=0.0005)

torch_optimizer = torch.optim.Adagrad(torch_model.parameters(), lr=0.005)

optimizer = Adagrad(lr=.005)
fig,ax  = plt.subplots(1)
plt.ion()
for i in range(100000):
    error = model.train(train_x, train_y, epochs=1,optimizer=optimizer,batch_size=64,shuffle=True)


    # train torch model
    criterion = torch.nn.MSELoss()
    output = torch_model(torch.from_numpy(train_x.reshape(-1,1)) )
    loss = criterion(output, torch.from_numpy(train_y.reshape(-1,1)) )
    loss.backward()
    torch_optimizer.step()

    print(f"Epochs {i}, error {error}")
    fig.clear()
    plt.title(f"Epochs {i}")
    plt.plot(train_x, train_y, "b.",label="Original data")
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())
    plt.plot(train_x, model(train_x), "r-",label="My Dnn")
    output = torch_model(torch.from_numpy(train_x.reshape(-1,1)))
    plt.plot(train_x, output.detach().numpy(), "g-",label="Torch Dnn")
    plt.legend()
    plt.grid()
    x_w,y_w = max(train_x) - min(train_x), max(train_y) - min(train_y)
    plt.xlim([min(train_x) - x_w * 0.05, max(train_x) + x_w * 0.05])
    plt.ylim([min(train_y) - y_w * 0.05, max(train_y) + y_w * 0.05])
    plt.savefig(f"fig/Epochs_{i}.png")
    plt.pause(0.002)
plt.ioff()
plt.show()


