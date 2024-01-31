import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, weight_decay=0.01):
        super(MLP, self).__init__()

        # 输入层
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_relu = nn.ReLU()

        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            #self.hidden_layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            self.hidden_layers.append(nn.ReLU())


        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.weight_decay = weight_decay

    def forward(self, x):
        # 输入层
        x = self.input_layer(x)
        x = self.input_relu(x)


        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        x_clamped = torch.clone(x)
        x_clamped[:,0] = torch.clamp(x[:,0], max=573)
        x_clamped[:,1] = torch.clamp(x[:,1], max=609)
        x_clamped[:,2] = torch.clamp(x[:,2], max=706)
        # x = torch.clamp(x, 0, 705)
        return x_clamped

    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).to(torch.device("cuda"))  # 初始化 L2 正则化项
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)  # 累加每个参数的 L2 范数
        return 0.1*l2_reg

# class MLP(torch.nn.Module):
#     # 定义神经网络
#     def __init__(self, n_feature, n_hidden, n_hidden1,n_hidden2, n_output):
#         # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
#         super(MLP, self).__init__()
#         # 此步骤是官方要求
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)
#         self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
#         # 设置输入层到隐藏层的函数
#         self.predict = torch.nn.Linear(n_hidden2, n_output)
#
#     # 设置隐藏层到输出层的函数
#
#     def forward(self, x):
#         # 定义向前传播函数
#         x = F.relu(self.hidden(x))
#         # 给x加权成为a，用激励函数将a变成特征b
#         x = F.relu((self.hidden1(x)))
#         x = F.relu(self.hidden2(x))
#         x = self.predict(x)
#         # 给b加权，预测最终结果
#         return x