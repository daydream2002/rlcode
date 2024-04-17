#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : imitation_learning_ppo.py
# @Description: 模仿学习，知一预训练
import sys

sys.path.append('../../')
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

from stable_baselines.gail import generate_expert_traj

from mahEnv import MahEnv

env = MahEnv()


# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller
def dummy_expert(_obs):
    """
    用知一生成预训练的数据集
    @param _obs: 状态
    @return: 推荐打出的牌
    """
    out_card = env.get_url_recommand()
    return out_card


generate_expert_traj(dummy_expert, 'suphx_features_455_1pzhiyi_noGang_Mid_abort-1_nep5000', env, n_episodes=5000)
