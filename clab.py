# env.py

from torch import nn
import torch
from label_evaluate import evaluate, un_normalization, distance, toReal01

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

# earlystopping.py

import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path, full_model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path, full_model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path, full_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, save_path, full_model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(
            model, save_path + "/" +
                   "checkpoint_{}_{:.6f}.pth.tar".format(epoch, val_loss))
        torch.save(
            full_model, save_path + "/" +
                        "full_{}_{:.6f}.pth.tar".format(epoch, val_loss))
        self.val_loss_min = val_loss


# ConvRNN.py

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ConvRNN.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   convrnn cell
'''

import torch
import torch.nn as nn


class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()
        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=16):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)

# model.py

from torch import nn
import torch.nn.functional as F
import torch


class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class ED(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output

# net_params.py

from collections import OrderedDict


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [2, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(width,height), input_channels=16, filter_size=5, num_features=32),
        CLSTM_cell(shape=(width // 2,height // 2), input_channels=32, filter_size=5, num_features=96),
        CLSTM_cell(shape=(width // 4,height // 4), input_channels=96, filter_size=5, num_features=96)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [32, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(width // 4,height // 4), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(width // 2,height // 2), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(width,height), input_channels=96, filter_size=5, num_features=32),
    ]
]

convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(width,height), input_channels=16, filter_size=5, num_features=64),
        CGRU_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        CGRU_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        CGRU_cell(shape=(width,height), input_channels=96, filter_size=5, num_features=64),
    ]
]

# utils.py

from torch import nn
from collections import OrderedDict


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


# decoder.py

from torch import nn
import torch


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))
        # self.l = D2NL()
        # self.CNN =CNN()

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=out_step)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        # TODO 接全链接预测位置
        # inputs = self.l(inputs)
        # inputs = self.CNN(inputs)
        return inputs


# encoder.py

from torch import nn
import torch
import logging

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

# loader.py

from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np
import os
import pandas as pd
import torch
import pickle

from PIL import Image
import torchvision.transforms as transforms


def resize(data, size=64):
    mode = Image.fromarray(data, mode="F")
    transform1 = transforms.Compose([
        transforms.Scale(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor()
    ])
    mode = transform1(mode)
    return mode


class SupervisionDataSet(Dataset):

    def __init__(self, x):
        total, c, w, h = x.shape
        self.x = x
        # print(x.shape)
        self.y = x[:, 1, :, :].view(total, 1, w, h)
        self.__len = max(total - in_step - out_step, 0)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        x = self.x.clone()
        y = self.y.clone()
        y = y[index + in_step:index + in_step + out_step]
        x = x[index:index + in_step]

        # cnn
        # x = x[index]
        # y = y[index]
        return x, y


# def read_all(dir_path):
#     # mock
#     # all = []
#     # for i in range(100):
#     #   all.append(SupervisionDataSet(torch.randn(100,2,width,height)))
#     # return ConcatDataset(all)
#     # real
#     result = []
#     for dirname, _, filenames in os.walk(dir_path):
#         for filename in filenames:
#             if filename.endswith(".npy"):
#                 full_path = os.path.join(dirname, filename)
#                 g, i, lat, lon, *_ = filename.split("_")
#                 d = np.load(full_path)
#                 # d.resize((101, 101))
#                 d = d.reshape((1, d.shape[0], d.shape[1]))
#                 # d = resize(d, width)
#                 # d = d.view()
#                 result.append([int(g), int(i), float(lat), float(lon), d])
#     result = pd.DataFrame(result, columns=["g", "i", "lat", "lon", "p"])
#     all = []
#     for g in result.groupby("g").size().keys():
#         data = result[result['g'] == g].sort_values('i')
#
#         y = torch.tensor(data[['lat', 'lon']].to_numpy(), dtype=torch.float32)
#         x = torch.tensor(np.vstack(data['p'].values), dtype=torch.float32)
#         total, w, h = x.shape
#
#         ch0 = torch.cat([resize(d.numpy(), 256) for d in x])
#         ch1 = lon_lat_to_graph(y.clone(), 256, 256)
#         x = torch.cat([
#             ch0.view(total, 1, 256, 256),
#             ch1.view(total, 1, 256, 256),
#         ], dim=1)
#         x = move_window(x, 64, 64, 64, 64)
#         for d in x:
#             all.append(SupervisionDataSet(d))
#         # for d in x:
#         #     all.append(SupervisionDataSet(d))
#     all = ConcatDataset(all)
#     return all


def gen_grid_data_set(dir_path):
    all = read_data_from_cache()
    # if all is not None:
    #     all_length = len(all)
    #     train_size = int(all_length * 0.9)
    #     test_size = all_length - train_size
    #     train_data, test_data = torch.utils.data.random_split(all, [train_size, test_size])
    #     return train_data, test_data
    # all = read_all(dir_path)
    # save_data(all)
    all_length = len(all)
    train_size = int(all_length * 0.9)
    test_size = all_length - train_size
    train_data, test_data = torch.utils.data.random_split(all, [train_size, test_size])
    return train_data, test_data

data_path = "data.bin"

def save_data(data):
    with open(data_path, "wb") as data_file:
        pickle.dump(data, data_file)
    print("save data finish")


def read_data_from_cache():
    if not os.path.exists(data_path):
        return None
    with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)
    print("read data from cache")
    return data


def get_loader():
    train_data_set, test_data_set = gen_grid_data_set('/home/zmp/projects/data_acquisition/tasks/imgs')
    train_data_loader = DataLoader(
        dataset=train_data_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_data_loader = DataLoader(
        dataset=test_data_set,
        batch_size=batch_size,
        shuffle=False
    )
    return train_data_loader, test_data_loader


# main.py

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    default=True,
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    default=False,
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=1,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=16,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=8,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-task_name',
                    default="default",
                    type=str,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
# args = parser.parse_args()
args = {
    "task_name":"default",
    "convgru":False,
    "lr":1e-4,
    "epochs":10
}

TIMESTAMP = args['task_name']

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

# print(args)
#
# if args.convlstm:
#     encoder_params = convlstm_encoder_params
#     decoder_params = convlstm_decoder_params
if args['convgru']:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params

encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
trainLoader, validLoader = get_loader()


def train():
    '''
    main function to run the training
    '''
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ED(encoder, decoder)
    print(net)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args['epochs'] + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        # for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
        for i, (inputVar, targetVar) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item()
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            pred_imgs = []
            real_imgs = []
            # for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            for i, (inputVar, targetVar) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                pred_imgs.append(pred)
                real_imgs.append(label)
                loss = lossfunction(pred, label)
                loss_aver = loss.item()
                # record validation loss
                valid_losses.append(loss_aver)
                # print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        real = torch.cat(real_imgs).view(-1, 1, 64, 64)
        pred = torch.cat(pred_imgs).view(-1, 1, 64, 64)
        print(real.shape)
        print(pred.shape)
        tb.add_image("Real_imgs", torchvision.utils.make_grid(real, normalize=True, nrow=8, ), epoch)
        tb.add_image("Pred_imgs", torchvision.utils.make_grid(pred, normalize=True, nrow=8, ), epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args['epochs']))

        print_msg = (f'[{epoch:>{epoch_len}}/{args["epochs"]:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


def avg(arr):
    return torch.mean(torch.tensor(arr)).item()


lrg = {}


def record(k, v):
    if k not in lrg:
        lrg[k] = []
    lrg[k].append(v)


def avg_record():
    result = {}
    for k in lrg.keys():
        result[k] = avg(lrg[k])
    return result


def show(r):
    for k in r.keys():
        print(k, r[k])


def get_avg_pos(data):
    result = []
    for g in data:
        x, y = 0, 0
        l = 0
        for i in range(len(g)):
            r = g[i]
            for j in range(len(r)):
                if r[j] > 0:
                    l += 1
                    x = i
                    y = j
        result.append([x / l / 64, y / l / 64])
    return result


def get_max_pos(data):
    result = []
    for g in data:
        x, y = 0, 0
        l = 0
        mx = torch.max(g)
        for i in range(len(g)):
            r = g[i]
            for j in range(len(r)):
                if r[j] == mx:
                    l += 1
                    x += i
                    y += j
        result.append([x / l / 256, y / l / 256])
    return result



def eval():
    model_path = "/home/zmp/projects/undnow/save_model/colab/full_2_0.000000.pth.tar"
    net = torch.load(model_path)
    _, validLoader = get_loader()
    with torch.no_grad():
        net.eval()
        t = tqdm(validLoader, leave=False, total=len(validLoader))
        for i, (inputVar, targetVar) in enumerate(t):
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            pred = net(inputs)

            p = torch.tensor(get_max_pos(pred.clone().reshape(-1, 256, 256)), dtype=torch.float)
            l = torch.tensor(get_max_pos(label.clone().reshape(-1, 256, 256)), dtype=torch.float)

            print()
            print(l.tolist())
            print(p.tolist())

            # mse, rmse, mae, min_diff, max_diff = evaluate(pred.clone().reshape(-1, 64, 64),
            #                                               label.clone().reshape(-1, 64, 64))
            mse, rmse, mae, min_diff, max_diff = evaluate(torch.tensor(l, dtype=torch.float),
                                                          torch.tensor(p, dtype=torch.float))
            record("mse", mse)
            record("rmse", rmse)
            record("mae", mae)
            record("min_diff", min_diff)
            record("max_diff", max_diff)
            t0 = un_normalization(l)
            t1 = un_normalization(p)
            print()
            print(t0.tolist())
            print(t1.tolist())
            mse, rmse, mae, min_diff, max_diff = evaluate(t0, t1)
            record("u_mse", mse)
            record("u_rmse", rmse)
            record("u_mae", mae)
            record("u_min_diff", min_diff)
            record("u_max_diff", max_diff)
            e1, e2 = distance(toReal01(p), toReal01(l))
            record("distance", avg(e2))
        show(avg_record())



if __name__ == "__main__":
    eval()
