import torch.optim


class OptimizerFactory:
    def __init__(self,  optimizer_name, net, lr):
        self.optimizer = None
        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
            # self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        elif optimizer_name == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)

    def get_learning_rate(self):
        # 获取优化器的当前学习率
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
