from torch import nn
import torch

in_step = 16  # 输入步长
out_step = 8  # 输出步长

# in_step = 2  # 输入步长
# out_step = 1  # 输出步长

channels = 1  # 通道数量
#
# width = 85  # 图像宽
# height = 104  # 图像高
# width, height = 310, 381
width, height = 256, 256
# 训练参数
epochs = 20  # 训练轮数
batch_size = 1  # 批大小
loss_fn = nn.MSELoss()  # 用于优化器的损失函数
model_name = 'model.pt'  # 模型名称
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选取
lr = 1e-3  # 学习率

KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

f = False
use_cache_data = True
