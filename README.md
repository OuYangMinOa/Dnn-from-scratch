# Build a DP module from scratch
  Because it uses pure python and is just a learning project, the speed is very slow and the effect is very poor.

![til](https://github.com/OuYangMinOa/Dnn-from-scratch/blob/main/example.gif)

# Example.py
  - Build a simple Dnn
    ```python
    from ou_dl import Module
    from ou_dl.Layers import Dnn
    from ou_dl.Func import ReLU,Sigmoid,Tanh,Linear

    model = Module.seq(
        Dnn(1, 32,activate_function=Tanh()),
        Dnn(32, 32,activate_function=Tanh()),
        Dnn(32, 32,activate_function=Tanh()),
        Dnn(32, 1,activate_function=Linear()),
    )   
    ```
  - Optimizers
    ```python
    from ou_dl.Opt import Adam, Adagrad

    optimizer = Adagrad(lr=.005)
    ```

  - Train the Model
    ```python
    from ou_dl.Loss import mse

    for i in range(100000):
      error = model.train(train_x, train_y, epochs=1,
      optimizer=optimizer,batch_size=64,shuffle=True,error_func=mse())
      print(f"Epochs {i}, error {error}")
    ```
