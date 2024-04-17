#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : pretrain_model_with_v6_ppo2_res101tf.py
# @Description:ResNet101网络的模型预训练，向知一学习
import sys

sys.path.append('../../../')
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

import mahEnv
from sbln_learning.custom_policy.tf.resnet34_policy_tf import Resnet34_policy
# from sbln_learning.resnet50_policy_new import Resnet50_policy_byZengw
from ppo2_copy import PPO2
from stable_baselines.gail import ExpertDataset
from sbln_learning.custom_policy.tf.resnet101_policy_tf import Resnet101_policy

'''
  train pretrain model from experience
'''

# from stable_baselines.common.callbacks import CheckpointCallback

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for uspiing the whole dataset
dataset = ExpertDataset(expert_path='../suphx_features_455_1pzhiyi_noGang_Mid_abort-1_nep5000_5.npz',
                        traj_limitation=-1, batch_size=256)

env = mahEnv.MahEnv()

model = PPO2(Resnet101_policy, env, verbose=1, learning_rate=2.5e-3, nminibatches=8, noptepochs=8, n_steps=1024)

model.pretrain(dataset, n_epochs=8, learning_rate=2.5e-4, val_interval=1)

model.save(
    "./save_premodel/suphx_features455_premodel_lr2.5e-4_ep100_v6_ppo2_lr2.5-4_nsteps1024_res101tf_noGang_abort-1_1pzhiyi",
    True)

# model.learn(total_timesteps=10000000)
# #
# # model.pretrain(dataset, n_epochs=500, learning_rate=2.5e-5, val_interval=1)
# model.save("./save_premodel/from_zhiyi_ep1000_after_ep10000000", True)  # 前面数字赢、输、中间的奖励，策略-状态大小-学习率-预训练方案-预训练的批次

# Pretrain the PPO1 model
