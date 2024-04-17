# python3
# Create Dat3: 2022-12-27
# Func: PPO 输出action为连续变量
# =====================================================================================================

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
import numpy as np
import gym
import copy
import random
from collections import deque
from tqdm import tqdm
import typing as typ
from sbln_learning.torch_v.policy_model.resnet_50 import BaseLay, BaseLayWithTanh, BaseLayWithNB


class policyNet(nn.Module):
    """
    continuity action:
    normal distribution (mean, std)
    """

    def __init__(self, in_channels, out_channels, device=None):
        super(policyNet, self).__init__()
        self.device = device
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        base_lay = BaseLay(256)
        lays = []
        for i in range(50):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        # self.nb = nn.BatchNorm2d(1)
        # self.sf = nn.Softmax(dim=1)

        # self.nb = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU()
        # self.final_lay = nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1)

        self.final_lay = nn.Sequential(  # linear
            nn.Flatten(),
            nn.Linear(256 * 34 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
        )

    def forward(self, inputs, masks=None):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        # outputs = self.nb(outputs)
        # outputs = self.relu(outputs)
        outputs = self.final_lay(outputs)
        # outputs = self.final_lay(outputs)
        # outputs = outputs + 1e-10  # 防止有log零的出现
        # outputs = self.nb(outputs)
        outputs = outputs.view(outputs.shape[0], 34)
        # outputs = outputs - torch.max(outputs, dim=1).values.view(outputs.shape[0], 1)  # 防止softmax溢出问题
        if masks is not None:
            masks = torch.tensor(masks, dtype=torch.bool).to(self.device)
            outputs = torch.where(masks, outputs, torch.tensor(-1e6, dtype=torch.float).to(self.device))
        # print(outputs)
        # print(f"smb:{outputs.shape}")
        # outputs = F.softmax(outputs, dim=1)
        if torch.any(torch.isnan(outputs)):
            print(inputs.shape)
            print(inputs.sum())
            for parameters in self.parameters():  # 打印出参数矩阵及值
                print(parameters)
            torch.save(self.state_dict(), "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/po_1.pth")
            print(torch.any(torch.isnan(inputs)))
            print(outputs)
        assert not torch.any(torch.isnan(outputs))
        outputs = Categorical(logits=outputs).probs  # 也可以当softmax用
        # print(outputs.logits)
        # print(outputs.probs)
        # print(f"smf:{outputs.shape}")
        return outputs

    def predict(self, inputs, masks=None):
        inputs = torch.tensor(inputs, dtype=torch.float).view(1, 1330, 34, 1)
        inputs = inputs.to(self.device)
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        # outputs = self.nb(outputs)
        # outputs = self.relu(outputs)
        outputs = self.final_lay(outputs)
        # outputs = outputs + 1e-10  # 防止有log零的出现
        # outputs = self.nb(outputs)
        outputs = outputs.view(outputs.shape[0], 34)
        if masks is not None:
            masks = torch.tensor(masks, dtype=torch.bool).view(1, 34).to(self.device)
            outputs = torch.where(masks, outputs, torch.tensor(-1e4, dtype=torch.float).to(self.device))
        return outputs.argmax(dim=1)



class policyNetWithFC(nn.Module):
    """
    continuity action:
    normal distribution (mean, std)
    """

    def __init__(self, in_channels, out_channels):
        super(policyNetWithFC, self).__init__()
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        base_lay = BaseLay(256)
        lays = []
        for i in range(50):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        self.nb = nn.BatchNorm2d(out_channels)
        # self.sf = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256 * 34 * 1, 512)
        self.linear2 = nn.Linear(512, out_channels)

    def forward(self, inputs):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        outputs = self.flatten(outputs)
        outputs = self.linear1(outputs)
        outputs = self.linear2(outputs)
        # outputs = outputs + 1e-10  # 防止有log零的出现
        # outputs = self.nb(outputs)
        # outputs = outputs.view(outputs.shape[0], 34)
        # print(f"smb:{outputs.shape}")
        # outputs = F.softmax(outputs, dim=1)
        # print(f"smf:{outputs.shape}")
        return outputs

class policyNetWithTanh(nn.Module):
    """
    continuity action:
    normal distribution (mean, std)
    """

    def __init__(self, in_channels, out_channels):
        super(policyNetWithTanh, self).__init__()
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        base_lay = BaseLayWithTanh(256)
        lays = []
        for i in range(50):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        # self.nb = nn.BatchNorm2d(out_channels)
        # self.sf = nn.Softmax(dim=1)
        self.final_lay = nn.Conv2d(256, 1, kernel_size=1, padding=0, stride=1)

    def forward(self, inputs):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        outputs = self.final_lay(outputs)
        # outputs = outputs + 1e-10  # 防止有log零的出现
        # outputs = self.nb(outputs)
        # outputs = outputs.view(outputs.shape[0], 34)
        # print(f"smb:{outputs.shape}")
        # outputs = F.softmax(outputs, dim=1)
        # print(f"smf:{outputs.shape}")
        return outputs

class valueNet(nn.Module):
    def __init__(self, in_channels):
        super(valueNet, self).__init__()
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        base_lay = BaseLay(256)
        lays = []
        for i in range(50):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        # self.final_lay = nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(256 * 34 * 1, 1)

    def forward(self, inputs):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        outputs = self.flatten(outputs)
        return self.head(outputs)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    adv_list = []
    adv = 0
    for delta in td_delta[::-1]:
        adv = gamma * lmbda * adv + delta
        adv_list.append(adv)
    adv_list.reverse()
    return torch.FloatTensor(adv_list)


class PPO:
    """
    PPO算法, 采用截断方式
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 gamma: float,
                 PPO_kwargs: typ.Dict,
                 device: torch.device
                 ):
        self.actor = policyNet(state_dim, action_dim, device).to(device)
        self.critic = valueNet(state_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.last_critic_loss_sum = 0

        self.gamma = gamma
        self.lmbda = PPO_kwargs['lmbda']
        self.ppo_epochs = PPO_kwargs['ppo_epochs']  # 一条序列的数据用来训练的轮次
        self.eps = PPO_kwargs['eps']  # PPO中截断范围的参数
        self.count = 0
        self.device = device

    def take_action(self, state, mask=None):
        state = torch.tensor(state, dtype=torch.float).view(1, 1330, 34, 1)
        state = state.to(self.device)
        probs = self.actor(state, mask)
        # 创建以probs为标准的概率分布
        action_lists = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_lists.sample().item()
        # action_ = torch.tensor(action).view(1, 1).to(self.device)
        # old_log_probs = action_lists.log_prob(action_)
        # print(old_log_probs)
        # # print(action)
        # # print(action.shape)
        # exit(1)
        return action

    def predict(self, inputs, masks=None):
        return self.actor.predict(inputs, masks)

    def update(self, transition_dict):
        state = transition_dict.state
        action = np.expand_dims(transition_dict.action, axis=-1)  # 扩充维度
        reward = np.expand_dims(transition_dict.reward, axis=-1)  # 扩充维度
        next_state = transition_dict.next_state
        done = np.expand_dims(transition_dict.done, axis=-1)  # 扩充维度

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        # reward = (reward + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).to(self.device)

        # print(reward)
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_delta = td_target - self.critic(state)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        probs = self.actor(state)
        assert not torch.any(torch.isnan(probs))
        # 创建以probs为标准的概率分布
        action_lists = torch.distributions.Categorical(probs.detach())
        old_log_probs = action_lists.log_prob(action)
        critic_loss_sum = 0
        # print(state.shape)
        # print(action.shape)
        # print(reward.shape)
        # print(done.shape)
        # print(old_log_probs.shape)
        # exit(1)
        for _ in range(self.ppo_epochs):
            # print("--------------------")
            probs = self.actor(state)
            # 创建以probs为标准的概率分布
            action_lists = torch.distributions.Categorical(probs)
            log_prob = action_lists.log_prob(action)

            # e(log(a/b))
            ratio = torch.exp(log_prob - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = torch.mean(-torch.min(surr1, surr2)).float()
            critic_loss = torch.mean(F.mse_loss(self.critic(state).float(), td_target.detach().float())).float()
            critic_loss_sum += critic_loss
            # print(state)
            # print(critic_loss)
            # print(td_target.detach().float())
            # print(F.mse_loss(self.critic(state).float(), td_target.detach().float()))
            # print(actor_loss)
            # print(critic_loss)
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

        self.count += 1
        if self.count % 50 == 0:
            print(f"----Agent update {self.count}----")

        # print(critic_loss_sum.item())
        self.last_critic_loss_sum = critic_loss_sum

    # 训练
    def learn(self, transition_dict):
        # 提取数据集
        # states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        # rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1)  # 扩充维度
        rewards = np.expand_dims(transition_dict.reward, axis=-1)  # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1)  # 扩充维度

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        # rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # print(f"reward:{rewards}")
        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式 :计算优势函数估计，使用时间差分误差和上一个时间步的优势函数估计
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        critic_loss_sum = 0
        # 一组数据训练 epochs 轮
        for _ in range(self.ppo_epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            critic_loss_sum += critic_loss

            # print(f"actor_loss:{actor_loss}")
            # print(f"critic_loss:{critic_loss}")

            # 梯度清0
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_opt.step()
            self.critic_opt.step()
        self.count += 1
        if self.count % 100 == 0:
            print(f"----Agent update {self.count}----")

        # print(critic_loss_sum.item())
        self.last_critic_loss_sum = critic_loss_sum

if __name__ == '__main__':
    x = torch.randint(2, (2, 1330, 34, 1), dtype=torch.float)
    mas = torch.randint(2, (2, 34), dtype=torch.bool)
    mymodel = policyNet(1330, 1)
    print(mas)
    ou = mymodel(x, mas)
    print(ou)
    print(ou.sum())
