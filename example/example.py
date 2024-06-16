import sys
sys.path.append('../ou_dl')
from ou_dl import Module
from ou_dl.Layers import Dnn
from ou_dl.Func import ReLU,Sigmoid,Tanh,Linear, Mish
from ou_dl.Opt import Adam, Adagrad
from ou_dl.Loss import mse

import numpy as np
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, TensorDataset

import torch
import os

SEED = 8789

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("fig",exist_ok=True
            )
torch.set_default_dtype(torch.float64)

torch_model = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
torch_model.zero_grad() 

model = Module.Seq(
    Dnn(1, 16,activate_function =Mish()),
    Dnn(16, 32,activate_function=Mish()),
    Dnn(32, 16,activate_function=Mish()),
    Dnn(16, 1,activate_function=Linear()),
)   
print(model.parameters())
###### dataset
n = 1024
train_x = np.linspace(0.1,6,n)
train_y = np.sin(train_x*3)/train_x * np.exp(train_x*0.25)+ 0.6*np.random.rand(n)

###### torch dataset
train_x_tensor = torch.tensor(train_x.reshape(-1,1), dtype=torch.float64)
train_y_tensor = torch.tensor(train_y.reshape(-1,1), dtype=torch.float64)
tensor_dataset = TensorDataset(train_x_tensor, train_y_tensor)
train_dataloader = DataLoader(tensor_dataset, batch_size=64, shuffle=True)

##### Optimizers
torch_optimizer = torch.optim.Adagrad(torch_model.parameters(), lr=0.008)
optimizer = Adagrad(lr=.008)


fig,ax  = plt.subplots(1)
plt.ion()
for i in range(100000):
    error = model.train(train_x, train_y, epochs=1,optimizer=optimizer,batch_size=64,shuffle=False)

    # train torch model
    for each_batch in DataLoader(tensor_dataset, batch_size=64, shuffle=False):
        torch_x, torch_y = each_batch
        criterion = torch.nn.MSELoss()
        output = torch_model(torch_x )
        loss = criterion(output, torch_y )
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


