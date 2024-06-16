import numpy as np

class Adam:
    def __init__(self,lr = 0.0001,beta1 = 0.9, beta2=0.99, epsilon=1e-6) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = lr
        self.iter = 0
        self.mt_dict = {}
        self.vt_dict = {}

        self.mt_b_dict = {}
        self.vt_b_dict = {}

    def _init_momentum(self,layer_name,w_size, b_size):

        self.mt_dict[layer_name] = np.zeros(w_size)
        self.vt_dict[layer_name] = np.zeros(w_size)

        self.mt_b_dict[layer_name] = np.zeros(b_size)
        self.vt_b_dict[layer_name] = np.zeros(b_size)

    def step(self):
        self.iter += 1

    def step_w(self,gt,layer_name):
        self.mt_dict[layer_name] = self.beta1*self.mt_dict[layer_name] + (1-self.beta1)*gt
        self.vt_dict[layer_name] = self.beta2*self.vt_dict[layer_name] + (1-self.beta2)*gt**2

        mt_hat = self.mt_dict[layer_name]/(1-self.beta1**(self.iter+1))
        vt_hat = self.vt_dict[layer_name]/(1-self.beta2**(self.iter+1))

        # print(f"{self.iter=}, {self.mt=} {self.vt=} {mt_hat=} {vt_hat=}")
        return mt_hat*self.eta/(vt_hat**(1/2)+self.epsilon)
    


    def step_b(self,gt,layer_name):
        self.mt_b_dict[layer_name] = self.beta1*self.mt_b_dict[layer_name] + (1-self.beta1)*gt
        self.vt_b_dict[layer_name] = self.beta2*self.vt_b_dict[layer_name] + (1-self.beta2)*gt**2

        mt_hat = self.mt_b_dict[layer_name]/(1-self.beta1**(self.iter+1))
        vt_hat = self.vt_b_dict[layer_name]/(1-self.beta2**(self.iter+1))

        return mt_hat*self.eta/(vt_hat**(1/2)+self.epsilon)
    



class Adagrad:
    def __init__(self,lr = 0.01, epsilon=1e-8) -> None:
        self.epsilon = epsilon
        self.lr = lr
        self.iter = 0
        self.gt_dict = {}
        self.gt_b_dict = {}


    def _init_momentum(self,layer_name,w_size, b_size):
        self.gt_dict[layer_name] = np.zeros(w_size)
        self.gt_b_dict[layer_name] = np.zeros(b_size)

    def step(self):
        self.iter += 1

    def step_w(self, gt, layer_name):
        self.gt_dict[layer_name] += gt**2
        self.iter += 1
        return self.lr*gt/(np.sqrt(self.gt_dict[layer_name])+self.epsilon)
    
    def step_b(self, gt, layer_name):
        self.gt_b_dict[layer_name] += gt**2
        return self.lr*gt/(np.sqrt(self.gt_b_dict[layer_name])+self.epsilon)