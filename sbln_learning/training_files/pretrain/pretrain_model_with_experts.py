import sys
sys.path.append('../../../')
import mahEnv
from sbln_learning.custom_policy.keras.resnet50_policy_new import Resnet50_policy_byZengw
from stable_baselines import PPO1
from stable_baselines.gail import ExpertDataset

'''
  train pretrain model from experience
'''


# from stable_baselines.common.callbacks import CheckpointCallback

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for uspiing the whole dataset
dataset = ExpertDataset(expert_path='./zhiyi_expert_data_with_suphx_features.npz', traj_limitation=-1, batch_size=128)

env = mahEnv.MahEnv()

model = PPO1(Resnet50_policy_byZengw, env, verbose=1, optim_epochs=4, optim_stepsize=2.5e-5, optim_batchsize=64, lam=0.95,
             adam_epsilon=1e-6) #n_cpu_tf_sess=8)  #  , n_cpu_tf_sess=1)

model.pretrain(dataset, n_epochs=12, learning_rate=2.5e-5, val_interval=1)
#
model.save("./save_premodel/zhiyi_expert_data_with_suphx_features_premodel_ep_12_lr2.5-5", True)


# model.learn(total_timesteps=10000000)
# #
# # model.pretrain(dataset, n_epochs=500, learning_rate=2.5e-5, val_interval=1)
# model.save("./save_premodel/from_zhiyi_ep1000_after_ep10000000", True)  # 前面数字赢、输、中间的奖励，策略-状态大小-学习率-预训练方案-预训练的批次

# Pretrain the PPO1 model
