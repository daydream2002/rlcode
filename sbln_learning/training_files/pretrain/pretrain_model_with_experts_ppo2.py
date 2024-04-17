import sys
sys.path.append('../../../')
import os
import tensorflow as tf
GPU = "0"  # 使用哪块gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

import mahEnv
# from sbln_learning.custom_policy.tf.resnet101_policy_tf import Resnet101_policy
from sbln_learning.custom_policy.tf.resnet50_policy_tf import Resnet50_policy
# from sbln_learning.resnet50_policy_new import Resnet50_policy_byZengw
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.schedules import LinearSchedule
'''
  train pretrain model from experience
'''


# from stable_baselines.common.callbacks import CheckpointCallback

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for uspiing the whole dataset
LEARNING_RATE = 5e-4  # (float or callable) 学习率，可以是个函数
NMINIBATCHES = 8  # 训练时，把nstep划分成n个minibatchsize
NOPTEPOCHS = 8  # (int) 对每个minibatchsize训练的epochs
N_STEPS = 2048  # (int) 每次更新的步数
TOTAL_TIMESTEPS = 100000000  # 总共训练的步数
POLICY = "Resnet50_tf"  # 使用的策略


dataset = ExpertDataset(expert_path='../suphx_features_455_1pzhiyi_noGang_abort-1_nep2000.npz', traj_limitation=-1, batch_size=64)

env = mahEnv.MahEnv()

model = PPO2(Resnet50_policy, env, verbose=1, learning_rate=LinearSchedule(TOTAL_TIMESTEPS, initial_p=LEARNING_RATE,
            final_p=LEARNING_RATE/10).value, nminibatches=NMINIBATCHES, noptepochs=NOPTEPOCHS, n_steps=N_STEPS)

model.pretrain(dataset, n_epochs=100, learning_rate=2.5e-4, val_interval=1)
#
model.save("./save_premodel/suphx_features455_premodel_lr2.5e-4_ep200_ppo2_lr"+str(LEARNING_RATE)
           +"_nsteps"+str(N_STEPS)+"_"+POLICY+"noGang_abort-1_1pzhiyi_relu", True)


# model.learn(total_timesteps=10000000)
# #
# # model.pretrain(dataset, n_epochs=500, learning_rate=2.5e-5, val_interval=1)
# model.save("./save_premodel/from_zhiyi_ep1000_after_ep10000000", True)  # 前面数字赢、输、中间的奖励，策略-状态大小-学习率-预训练方案-预训练的批次

# Pretrain the PPO1 model
