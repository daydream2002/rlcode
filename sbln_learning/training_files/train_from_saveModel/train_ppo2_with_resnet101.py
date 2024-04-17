import sys
sys.path.append('../../../')
sys.path.append("../")
import os
import tensorflow as tf
GPU = "1"  # 使用哪块gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.24  # 设置显存占用率大小
tf.compat.v1.Session(config=config)

# from sbln_learning.resnet50_policy_new import Resnet50_policy_byZengw
import mahEnv
from sbln_learning.custom_policy.tf.resnet101_policy_tf import Resnet101_policy
from stable_baselines import PPO2
from stable_baselines.common.schedules import LinearSchedule
'''
  train model from the pretrainModel
'''
# from stable_baselines.common.callbacks import CheckpointCallback
# random_abort-3_from_best_reward=2.74

VERBOSE = 1  # (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
# LEARNING_RATE = 3e-4  # (float or callable) The learning rate, it can be a function

N_STEPS = 2048  # (int) The number of steps to run for each environment per update # 每个minibatches的步数
NMINIBATCHES = N_STEPS // 256  # (int) Number of training minibatches per update. For recurrent policies,
                # thINIT_CLIPRANGE/2ee number of environments run in parallel should be a multiple of nminibatches.
NOPTEPOCHS = 8   # (int) Number of epoch when optimizing the surrogate
TOTAL_TIMESTEPS = 100000000  # 总共训练的步数
GAMMA = 0.925  # # reward折扣因子  设计原则  （1 - 1/平均步长） 微调
ENT_COEF = 0.004  # 策略熵的折扣系数，默认0.01  微调
VF_COEF = 0.5  # loss的熵函数系数 默认0.5  不变
TENSORBOARD_LOG = "./tensorboardLog"  # tensorboard的存放路径
FULL_LOG = False  # 是否记录所有数据记录

DynEnt = "_dynEnt_0.15"  # "_dynEnt_0.5"  # 是否动态调节策略熵  数字必须放到后面，并且用_隔开
MidReward = False  # 是否有中间步奖励
IllegaA = -3   # 非法动作的奖励
POLICY = "Resnet101_tf"  # 使用的策略

INIT_CLIPRANGE = 0.2 # clip阈值 不变
CLIPRANGE = INIT_CLIPRANGE # LinearSchedule(TOTAL_TIMESTEPS, initial_p=INIT_CLIPRANGE, final_p=INIT_CLIPRANGE/2).value

INIT_LR = 3e-4
END_LR = 3e-5  # LR不变
LEARNING_RATE = LinearSchedule(TOTAL_TIMESTEPS, initial_p=INIT_LR, final_p=END_LR).value  # (float or callable) The learning rate, it can be a function

TENSORBOARD_LOG_NAME = "PPO2_fea1274_ND_"+"lr_start"+str(INIT_LR)+"end"+str(END_LR)+"_"+POLICY+"_gm"+str(GAMMA)+"_clip"+str(INIT_CLIPRANGE)+"_entCoef"+\
                       str(ENT_COEF)+"_sqrtR_1p_mid"+str(MidReward) + DynEnt
# MODEL_PATH = "../pretrain/save_premodel/suphx_features455_premodel_lr0.0002_ep100_ppo2_lr0.0003_" \
#              "gamma0.95_enc_coef0.01_vfCoef0.6_nsteps2048_Resnet18_tfnoGang_noMid_abort-1_1pzhiyi" # 从哪个模型开始恢复训练
MODEL_TO_SAVE_PATH = "./save_model"  # 保存模型的为位置
SAVEMODEL_REWARD = 0  # 从多少reward开始保存模型


MODEL_PATH = "../train_with_raw/save_model/ppo2_policyResnet101_tf_Nfea1274_updateStep2048_lr0.00029527772159999995_" \
             "gamma0.925_entCoef0.004_vfCoef0.5_reward0.9370618404322107_GangIsFalse_MidRewardTrue_illegaA-3_" \
             "abort_reward-1_nsteps_1751040" # 从哪个模型开始恢复训练


INFO_ON_LOG = {"lr": INIT_LR, "lr_info": "LinearSchedule; end_lr:"+str(END_LR),"n_fea": 1274,
               "clip":INIT_CLIPRANGE, "clip_info":"LinearSchedule; end_clip:"+str(INIT_CLIPRANGE),
               "nminibatches": NMINIBATCHES, "noptepochs": NOPTEPOCHS, "n_steps": N_STEPS,
               "total_timesteps": TOTAL_TIMESTEPS, "gamma": GAMMA, "ent_coef": ENT_COEF, "vf_coef": VF_COEF,
               "policy": POLICY, "gpu": GPU, "dynEnt": DynEnt,"model_path": MODEL_PATH,
               "GameInfo": {"opponent": ["zhiyi", "random", "random"], "midReward": MidReward, "sqrtReward": True,
                            "isGang_score": False, "abort_reward": -1, "illegaA": IllegaA},
               "model_save_path": MODEL_TO_SAVE_PATH,"best_model": SAVEMODEL_REWARD}

TENSORBOARD_LOG = TENSORBOARD_LOG + "/n_fea" + str(INFO_ON_LOG["n_fea"])

if not os.path.exists(TENSORBOARD_LOG):
    os.mkdir(TENSORBOARD_LOG)

env = mahEnv.MahEnv()

model = PPO2(Resnet101_policy, env, gamma=GAMMA, ent_coef=ENT_COEF, verbose=VERBOSE, vf_coef=VF_COEF,
             learning_rate=LEARNING_RATE, cliprange=INIT_CLIPRANGE,
             nminibatches=NMINIBATCHES, noptepochs=NOPTEPOCHS, n_steps=N_STEPS, tensorboard_log=TENSORBOARD_LOG,
             full_tensorboard_log=FULL_LOG)

model = PPO2.load(MODEL_PATH, model.get_env())
# 重新加载配置  如果重新加载，会使预训练模型出问题
# model.learning_rate = LEARNING_RATE
# model.cliprange = CLIPRANGE
model.tensorboard_log = TENSORBOARD_LOG

model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=TENSORBOARD_LOG_NAME, info=INFO_ON_LOG)

model.save("./save_model/ppo2_"+POLICY+"_lr"+str(LEARNING_RATE)+"_gamma"+str(GAMMA)+"_entCoef"+str(ENT_COEF) +
           "_vfCoef"+str(VF_COEF)+"_end_steps_"+str(TOTAL_TIMESTEPS)+"_1pzhiyi_sqrtR_1p_mid"+str(MidReward)+DynEnt,True)
