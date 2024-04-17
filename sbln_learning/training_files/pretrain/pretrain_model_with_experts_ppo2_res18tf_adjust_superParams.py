import sys
sys.path.append('../../../')
import os
import tensorflow as tf
GPU = "1"  # 使用哪块gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

import mahEnv
# from sbln_learning.custom_policy.tf.resnet101_policy_tf import Resnet101_policy
from sbln_learning.custom_policy.tf.resnet18_policy_tf import Resnet18_policy
# from sbln_learning.resnet50_policy_new import Resnet50_policy_byZengw
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.schedules import LinearSchedule
'''
  train pretrain model from experience
'''


# from stable_baselines.common.callbacks import CheckpointCallback

# Using only one expert trajectory
VERBOSE = 1  # (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
LEARNING_RATE = 3e-4  # (float or callable) The learning rate, it can be a function

N_STEPS = 2048  # (int) The number of steps to run for each environment per update # 每个minibatches的步数
NMINIBATCHES = N_STEPS // 256  # (int) Number of training minibatches per update. For recurrent policies,
                # thee number of environments run in parallel should be a multiple of nminibatches.
NOPTEPOCHS = 8   # (int) Number of epoch when optimizing the surrogate
TOTAL_TIMESTEPS = 100000000  # 总共训练的步数
GAMMA = 0.925  # reward折扣因子  设计原则  （1 - 1/平均步长）
ENT_COEF = 0.01  # 策略熵的折扣系数，默认0.01
VF_COEF = 0.6  # loss的熵函数系数 默认0.5
TENSORBOARD_LOG = "./tensorboardLog"  # tensorboard的存放路径
FULL_LOG = False  # 是否记录所有数据记录

POLICY = "Resnet18_tf"  # 使用的策略

PRETRAIN_LR = 2e-4  # 预训练学习率
PRETRAIN_EP = 200  # 预训练ep

dataset = ExpertDataset(expert_path='../suphx_features_455_1pzhiyi_noGang_Mid_abort-1_nep5000.npz',
                        traj_limitation=-1, batch_size=256)

env = mahEnv.MahEnv()

model = PPO2(Resnet18_policy, env, gamma=GAMMA, ent_coef=ENT_COEF, verbose=VERBOSE, vf_coef=VF_COEF,
             learning_rate=LinearSchedule(TOTAL_TIMESTEPS, initial_p=LEARNING_RATE, final_p=LEARNING_RATE/10).value,
             nminibatches=NMINIBATCHES, noptepochs=NOPTEPOCHS, n_steps=N_STEPS, tensorboard_log=TENSORBOARD_LOG,
             full_tensorboard_log=FULL_LOG)

model.pretrain(dataset, n_epochs=PRETRAIN_EP, learning_rate=PRETRAIN_LR, val_interval=1)
#三
model.save("./save_premodel/suphx_features455_premodel_lr"+str(PRETRAIN_LR)+"_ep"+str(PRETRAIN_EP) +
           "_ppo2_lr"+str(LEARNING_RATE)+"_gamma"+str(GAMMA)+"_enc_coef"+str(ENT_COEF)+"_vfCoef"+str(VF_COEF) +
           "_nsteps"+str(N_STEPS)+"_"+POLICY+"noGang_Mid_abort-1_1pzhiyi", True)


# model.learn(total_timesteps=10000000)
# #
# # model.pretrain(dataset, n_epochs=500, learning_rate=2.5e-5, val_interval=1)
# model.save("./save_premodel/from_zhiyi_ep1000_after_ep10000000", True)  # 前面数字赢、输、中间的奖励，策略-状态大小-学习率-预训练方案-预训练的批次

# Pretrain the PPO1 model
