import os
from sbln_learning.torch_v.dataset import DatasetDiscard
from torch.utils.data import DataLoader

from sbln_learning.torch_v.policy_model.resnet_50 import ResNet50, ResNet50WithNB, ResNet50WithTanh, ResNet50WithLinear
from sbln_learning.torch_v.dataset import *
from torch.optim import AdamW
from torch import nn
import torch
import os

from sbln_learning.torch_v.policy_model.resnet_model import resnet50, resnet101

device = torch.device("cuda")
# 获取数据集
ds_train = DatasetDiscard("/home/tonnn/xiu/data/discard", "train", action_to_id)
data_train = DataLoader(ds_train, batch_size=1000)
ds_vel = DatasetDiscard("/home/tonnn/xiu/data/discard", "test", action_to_id)
data_vel = DataLoader(ds_vel, batch_size=1000)
# 构建网络
my_model = resnet101()
num_fc = my_model.fc.in_features
my_model.fc = nn.Linear(num_fc, 34)
# my_model.load_state_dict(torch.load("E:\\pythonCoding\\DL_AI-2\\model\\param\\ep29_acc=0.7933.pth"))
my_model.to(device)

# 特征提取网络
# feature_extract_model = FeatureExtractModel(1330, 256)
# feature_extract_model.to(device)

# 学习率
lr = 1e-4

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 定义优化器
optimer = AdamW(my_model.parameters(), lr=lr, weight_decay=0.01)

# epoch
EPOCH = 100
acc_list = []
epo_list = []
# 开始训练
for i in range(EPOCH):

    print(f"---epoch-{i}-start---")
    loss_sum = 0
    my_model.train()
    for idx, (inputs, targets) in enumerate(data_train):
        # print(inputs.shape)
        # exit(1)
        inputs, targets = inputs.to(device), targets.to(device)
        # intputs = feature_extract_model(intputs)  # 特征提取

        outputs = my_model(inputs)
        loss = loss_fn(outputs, targets)
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
        corret_sum = 0
        sample_sum = 0
        for inputs, targets in data_vel:
            inputs, targets = inputs.to(device), targets.to(device)
            # intputs = feature_extract_model(intputs)  # 特征提取

            # outputs = my_model.predict(inputs)
            outputs = my_model(inputs).argmax(dim=1)
            corret_sum += (outputs == targets).sum()
            sample_sum += targets.shape[0]
        acc = corret_sum / sample_sum
        acc_list.append(round(acc.item(), 4))
        epo_list.append(i)
        print(f"--test--{i}--acc:{acc:0.4f}")
        torch.save(my_model.state_dict(),
                   os.path.join("/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/"
                                "pretrain_wb",
                                f"net101_lr={lr}_AdamW_ep{i}_loss={loss_sum}_acc={acc:0.4f}.pth"))

pass
