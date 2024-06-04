import torch.nn as nn


class ModelInitializer:
    def __init__(self, method='xavier', uniform=True):
        """
        初始化 ModelInitializer 类。
        :param method: 初始化方法，支持 'xavier', 'he' 等。
        :param uniform: 是否使用均匀分布来初始化参数。对于 'he' 和其他某些方法可能被忽略。
        """
        self.method = method
        self.uniform = uniform

    def initialize(self, model):
        """
        对给定的模型应用初始化。
        :param model: 要初始化的 PyTorch 模型。
        """
        if self.method == 'xavier':
            self._xavier_init(model)
        elif self.method == 'he':
            self._he_init(model)
        else:
            raise ValueError(f"Unsupported initialization method: {self.method}")

    def _xavier_init(self, model):
        """
        应用 Xavier 初始化。
        """
        for p in model.parameters():
            if p.dim() > 1:  # 避免初始化偏置参数
                if self.uniform:
                    print("Using uniform initialization for Xavier initialization.")
                    # nn.init.xavier_uniform_(p)
                else:
                    print("Using normal initialization for Xavier initialization.")
                    # nn.init.xavier_normal_(p)

    def _he_init(self, model):
        """
        应用 He 初始化。
        """
        for p in model.parameters():
            if p.dim() > 1:  # 避免初始化偏置参数
                nn.init.kaiming_normal_(p, nonlinearity='relu')
