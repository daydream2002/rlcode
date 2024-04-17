import sys
sys.path.append('../../../')
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)
import mahEnv
from sbln_learning.custom_policy.keras.resnet50_policy_new import Resnet50_policy_byZengw
from stable_baselines import PPO2


env = mahEnv.MahEnv()

model = PPO2(Resnet50_policy_byZengw, env, verbose=1,n_steps=8) #, n_cpu_tf_sess=1)  #  , n_cpu_tf_sess=1)

model.learn(total_timesteps=10000)
model.save("./save_premodel/suphx_features_100000000ep_lr255", True)
