import numpy as np
import os
from sbln_learning.torch_v.dataset import DatasetDiscard
from torch.utils.data import DataLoader

from sbln_learning.torch_v.reward.GRU_model import DoubleGRUModel, DoubleSimpleGRUModel, CNNModel
from sbln_learning.torch_v.reward.dataset import DatasetTotal, DatasetGlobal
from torch.optim import AdamW, Adam
from torch import nn
import torch
import os

device = torch.device("cuda")
# 获取数据集
ds_train = DatasetGlobal("/home/tonnn/.nas/xy/output/globel", "train")
data_train = DataLoader(ds_train, batch_size=100)
ds_vel = DatasetGlobal("/home/tonnn/.nas/xy/output/globel", "test")
data_vel = DataLoader(ds_vel, batch_size=10)
# 构建网络
my_model = CNNModel(1330, 1, device)
my_model.load_state_dict(torch.load("/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/Globel"
                                    "/lr=0.0001_AdamW_ep199_test_loss_mean=431.3827.pth"))
my_model.to(device)

# 特征提取网络
# feature_extract_model = FeatureExtractModel(1330, 256)
# feature_extract_model.to(device)

# 学习率
lr = 1e-4

# 定义损失函数
loss_fn = nn.MSELoss()
loss_fn.to(device)

# 定义优化器
optimer = Adam(my_model.parameters(), lr=lr)

# epoch
start = 200
EPOCH = 500
acc_list = []
epo_list = []
# 开始训练
for i in range(start, EPOCH):

    print(f"---epoch-{i}-start---")
    loss_sum = 0
    my_model.train()
    for idx, (inputs, targets) in enumerate(data_train):
        # print(inputs.shape)
        # exit(1)
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs, size_lens, targets = inputs.to(device), size_lens.to(device), targets.to(device)
        # intputs = feature_extract_model(intputs)  # 特征提取

        outputs = my_model(inputs)
        # print(outputs)
        # print(targets)
        loss = loss_fn(outputs, targets)
        # print(loss)
        loss_sum += loss
        if idx % 5 == 1:
            print(f"epoch-{i}-train-{idx}-loss:{loss}")
        optimer.zero_grad()
        loss.backward()
        optimer.step()

    print(f"epoch-{i}-loss_sum:{loss_sum}")

    my_model.eval()
    print(f"--test-{i}-start--")
    with torch.no_grad():
        test_loss_sum = []
        for inputs,  targets in data_vel:
            # inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            # intputs = feature_extract_model(intputs)  # 特征提取
            outputs = my_model(inputs)
            # print(targets)
            # print(outputs)
            # exit()
            test_loss_sum.append(loss_fn(outputs, targets).item())
        print(f"--test--{i}--loss:{np.mean(test_loss_sum).item():0.4f}")
        if i % 10 == 0:
            torch.save(my_model.state_dict(),
                       os.path.join("/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/"
                                    "sbln_learning/torch_v/model_param/Globel",
                                    f"lr={lr}_AdamW_ep{i}_test_loss_mean={np.mean(test_loss_sum).item():0.4f}.pth"))

pass
