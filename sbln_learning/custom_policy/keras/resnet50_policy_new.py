#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : resnet50_policy_new.py
# @Description: resnet50网络结构
import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy

from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor

def conv2d_bn(input, kernel_num, kernel_size=(3, 1), strides=(1, 1), layer_name='', padding_mode='same',data_format="channels_first"):
    """
    卷积模块
    @param input: 输入
    @param kernel_num: 卷积核数目
    @param kernel_size: 卷积核大小
    @param strides: 步长
    @param layer_name: 卷积层名称
    @param padding_mode: 填充方式
    @param data_format: 数据格式
    @return: 卷积后输出
    """
    conv1 = Conv2D(kernel_num, kernel_size, strides=strides, padding=padding_mode, data_format=data_format,
                   kernel_regularizer=l2(0.001), name=layer_name + '_conv')(input)
    # # 如果training不置为False，会出现训练偏差  reference: https://www.cnblogs.com/Fosen/p/11419930.html
    # # 关于BatchNormalization的说明 参考：https://blog.csdn.net/macair123/article/details/79511179
    # batch1 = BatchNormalization(axis=1, name=layer_name + '_bn1', training=False)(conv1)
    batch1 = BatchNormalization(axis=1, name=layer_name + '_bn')(conv1)

    return batch1

def shortcut(fx, x, padding_mode='same', layer_name=''):
    """
    残差模块
    @param fx: 卷积核输出
    @param x: 输入
    @param padding_mode: 填充方式
    @param layer_name: 隐藏层名称
    @return: 残差后的输出
    """
    layer_name += '_shortcut'
    if x.shape[1] != fx.shape[1]:
        k = fx.shape[1]
        k = int(k)
        identity = conv2d_bn(x, kernel_num=k, kernel_size=1, padding_mode=padding_mode, layer_name=layer_name)
    else:
        identity = x

    return Add(name=layer_name + '_add')([identity, fx])

def basic_block(input, kernel_num=128, strides=(1, 1), layer_name='basic', padding_mode='same'):
    """
    残差快
    @param input: 输入
    @param kernel_num: 卷积核数目
    @param strides: 步长
    @param layer_name: 层名
    @param padding_mode: 填充方式
    @return: 残差块输出
    """
    # k1, k2 = kernel
    conv1 = conv2d_bn(input, kernel_num=kernel_num, kernel_size=(3, 1), strides=strides,
                      layer_name=layer_name + '_1', padding_mode=padding_mode)
    relu1 = Activation("relu", name=layer_name + '_relu1')(conv1)

    conv2 = conv2d_bn(relu1, kernel_num=kernel_num, kernel_size=(3, 1), strides=strides,
                      layer_name=layer_name + '_2', padding_mode=padding_mode)
    relu2 = Activation("relu", name=layer_name + '_relu2')(conv2)

    shortcut_add = shortcut(fx=relu2, x=input, layer_name=layer_name)
    relu3 = Activation("relu", name=layer_name + '_relu3')(shortcut_add)
    return relu3

def make_layer(input, block, block_num, kernel_num, layer_name=''):
    """
    多层残差模块
    @param input: 输入
    @param block: 块
    @param block_num: 块的数目
    @param kernel_num: 卷积核数目
    @param layer_name: 层名
    @return: 多层残差后的输出
    """
    x = input
    for i in range(1, block_num + 1):
        x = block(x, kernel_num=kernel_num, strides=(1, 1), layer_name=layer_name + str(i), padding_mode='same')
    return x

class Resnet50_policy_byZengw(ActorCriticPolicy):
    '''
        改进的ResNet50残差网络
        className:Resnet50_policy_byZengw
        fileName:resnet50_policy_new.py
    '''
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        """
        构造器
        @param sess: 会话
        @param ob_space: 状态空间
        @param ac_space: 动作空间
        @param n_env: 环境数目
        @param n_steps: 步数
        @param n_batch: batch数目
        @param reuse: 是否复用
        @param kwargs: 其他参数
        """
        super(Resnet50_policy_byZengw, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            print("self.processed_obs的shape为：", self.processed_obs.shape)
            conv0 = conv2d_bn(self.processed_obs, kernel_num=128, kernel_size=(3, 1), strides=(1, 1),
                              layer_name="basic" + '_0', padding_mode="same", data_format="channels_first")
            relu0 = Activation("relu", name="basic" + '_relu0')(conv0)

            # 100层卷积
            resNet_100 = make_layer(relu0, basic_block, block_num=50, kernel_num=128, layer_name="basic_block")

            # 最后actor网络输出
            conv_actor = conv2d_bn(resNet_100, kernel_num=1, kernel_size=(1, 1), strides=(1, 1),
                                   layer_name='actor', padding_mode="same", data_format="channels_first")
            relu_actor = Activation("relu", name="relu_actor")(conv_actor)

            action_h = Flatten(name="flatten_actor")(relu_actor)
            pi_latent = Dense(34, activation="relu", name="pi_latent_dense_relu")(action_h)
            # action_pi = Dense(34, activation="softmax", kernel_initializer='he_uniform', name="softmax_action")(
            #     action_h)
            action_latent = pi_latent


            # 最后value值网络的输出
            conv_value = conv2d_bn(resNet_100, kernel_num=32, kernel_size=(3, 1), strides=(1, 1),
                                   layer_name="value", padding_mode="valid", data_format="channels_first")
            value_h = Flatten(name="flatten_value")(conv_value)
            value_h = Dense(1024, activation='relu', kernel_initializer='he_uniform', name="fc1_value")(value_h)
            value_h = Dense(256, activation='relu', kernel_initializer='he_uniform', name="fc2_value")(value_h)

            value_fn = Dense(1, activation='linear', kernel_initializer='he_uniform', name="out_value")(value_h)
            value_latent = value_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(action_latent, value_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        对应与gym的step方法
        @param obs: 观测
        @param state: 状态
        @param mask: 掩膜
        @param deterministic: 是否开启deterministic
        @return: a,S,S0,neglogp
        """
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp, logits, acProb = self.sess.run([self.action, self.value_flat, self.neglogp,
                                                                    self.policy, self.policy_proba], {self.obs_ph: obs})

        print("动作概率：{}, 选择动作：{}, value:{}, neglogp:{}, logits:{}".format(acProb, action, value, neglogp, logits))
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        """
        概率分布step方法
        @param obs: 观测
        @param state: 状态
        @param mask: 掩膜
        @return: 概率分布
        """
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        """
        Q值，即价值评估结果
        @param obs: 观测
        @param state: 状态
        @param mask: 掩膜
        @return: 价值
        """
        return self.sess.run(self.value_flat, {self.obs_ph: obs})