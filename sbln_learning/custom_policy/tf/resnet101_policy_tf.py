#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : resnet101_policy_tf.py
# @Description: ResNet101网络
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from stable_baselines.common.policies import ActorCriticPolicy

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


def flatten(x, data_format="channels_first"):
    """
    平铺层
    @param x: 输入
    @param data_format: 数据格式
    @return: 平铺层输出
    """
    return tf.layers.flatten(x, data_format=data_format)


def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    """
    全连接层
    @param x: 输入
    @param units: 单元数
    @param use_bias: 是否开启偏向
    @param scope: 命名
    @return: 全连接层输出
    """
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x


def batch_norm(x, is_training=True, scope='batch_norm'):
    """
    批标准化层
    @param x: 输入
    @param is_training: 是否训练
    @param scope: 层名
    @return: 批标准化层的输出
    """
    return tf_contrib.layers.batch_norm(x, decay=0.99, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope, data_format="NCHW")
    # return tf.layers.batch_normalization(x, axis=1, momentum=0.99, epsilon=1e-05, trainable=is_training, name=scope)


def relu(x, name=None):
    """
    RELU激活函数
    @param x: 输入
    @param name: 层名
    @return: 输出
    """
    return tf.nn.relu(x, name)


def conv2d(x, kernel_num, kernel_size=(3, 1), stride=(1, 1), padding='SAME',
           use_bias=True, scope='conv2d', dataformat="channels_first"):
    """
    卷积层
    @param x: 输入
    @param kernel_num: 卷积核数目
    @param kernel_size: 卷积核大小
    @param stride: 步长
    @param padding: 填充模式
    @param use_bias: 是否开启偏差
    @param scope: 层名
    @param dataformat: 数据格式
    @return: 输出
    """
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=kernel_num,
                             kernel_size=kernel_size, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding, data_format=dataformat)
        return x


def resblock(x_init, kernel_num, kernel_size=(3, 1), stride=(1, 1), is_training=True,
             use_bias=True, downsample=False, layer_name='resblock', padding_mode='same'):
    """
    残差块
    @param x_init: 输入
    @param kernel_num:卷积核数目
    @param kernel_size: 卷积核大小
    @param stride: 步长
    @param is_training:是否训练
    @param use_bias: 是否使用偏差
    @param downsample: 是否采用下采样
    @param layer_name: 层名
    @param padding_mode: 填充模式
    @return: 输出
    """
    with tf.variable_scope(layer_name):

        x = batch_norm(x_init, is_training, scope=layer_name + "_bn0")
        x = relu(x, name=layer_name + "_bn0_relu")

        if downsample:
            x = conv2d(x, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                       use_bias=use_bias, scope=layer_name + "_conv2d0")
            x_init = conv2d(x_init, kernel_num, kernel_size=1, stride=stride, use_bias=use_bias,
                            scope=layer_name + '_conv_init')

        else:
            x = conv2d(x, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                       use_bias=use_bias, scope=layer_name + "_conv2d0")

        x = batch_norm(x, is_training, scope=layer_name + "_bn1")
        x = relu(x, name=layer_name + "_bn1_relu")
        x = conv2d(x, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                   use_bias=use_bias, scope=layer_name + 'conv2d1')

        short_cut = tf.add(x, x_init, name=layer_name + "short_cut")
        return short_cut


def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock'):
    """
    下采样残差模块
    @param x_init:输入
    @param channels: 通道数
    @param is_training: 是否训练
    @param use_bias: 是否开启偏差
    @param downsample: 下采样
    @param scope: 命名
    @return: 输出
    """

    with tf.variable_scope(scope):
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv2d(shortcut, channels, kernel_size=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample:
            x = conv2d(x, channels, kernel_size=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv2d(shortcut, channels * 4, kernel_size=1, stride=2, use_bias=use_bias, scope='conv_init')

        else:
            x = conv2d(x, channels, kernel_size=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv2d(shortcut, channels * 4, kernel_size=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv2d(x, channels * 4, kernel_size=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut


def make_layers(x, block, block_num=50, kernel_num=128, stride=(1, 1), layer_name="basic_resblock", is_training=True):
    """
    残差模块外层封装
    @param x: 输入
    @param block:残差块
    @param block_num: 块的数目
    @param kernel_num: 卷积核数目
    @param stride: 步长
    @param layer_name:层名
    @param is_training: 是否训练
    @return: 输出
    """
    for i in range(1, block_num + 1):
        x = block(x, kernel_num=kernel_num, kernel_size=(3, 1), stride=stride, is_training=is_training,
                  layer_name=layer_name + str(i), padding_mode='same')
    return x


class Resnet101_policy(ActorCriticPolicy):
    '''
        ResNet101策略
        className:Resnet101_policy
        fileName:resnet101_policy_tf.py
    '''
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        """
        构造器
        @param sess: 会话
        @param ob_space:状态空间
        @param ac_space: 动作空间
        @param n_env: 环境数目
        @param n_steps: 步数
        @param n_batch: 批次数
        @param reuse: 是否复用
        @param kwargs: 其他参数
        """
        super(Resnet101_policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                               scale=True)

        with tf.variable_scope("model", reuse=reuse):
            print("self.processed_obs的shape为：", self.processed_obs.shape)
            bn0 = batch_norm(self.processed_obs, scope="basic_block0_bn_0")
            relu0 = relu(bn0, name="bn0_relu")
            conv0 = conv2d(relu0, kernel_num=128, kernel_size=(3, 1), stride=(1, 1), padding='SAME', use_bias=True,
                           scope='basic_block0_bn_0_conv2d_0', dataformat="channels_first")

            # 100层卷积
            resNet_100 = make_layers(conv0, resblock, block_num=50, kernel_num=128, layer_name="basic_block")

            # 最后actor网络输出
            bn_actor = batch_norm(resNet_100, scope="bn_actor")
            relu_actor = relu(bn_actor)
            action_h = conv2d(relu_actor, 1, kernel_size=(1, 1), padding="same", scope="conv2d_actor")

            # action_h = Flatten(name="flatten_actor")(relu_actor)
            # pi_latent = Dense(34, activation="relu", name="pi_latent_dense_relu")(action_h)
            pi_latent = fully_conneted(action_h, 34, scope="pi_latent")
            # action_pi = Dense(34, activation="softmax", kernel_initializer='he_uniform', name="softmax_action")(
            #     action_h)
            action_latent = pi_latent

            # 最后value值网络的输出
            bn_value = batch_norm(resNet_100, scope="bn_value")
            relu_value = relu(bn_value)
            conv_value = conv2d(relu_value, 32, padding='valid', scope='conv2d_value')

            value_h = fully_conneted(conv_value, 1024, scope="fc1_value")
            value_h = tf.layers.dense(value_h, units=256, kernel_initializer=weight_init, name="value_h",
                                      kernel_regularizer=weight_regularizer, use_bias=True, activation=tf.nn.relu)
            # value_h = Dense(1024, activation='relu', kernel_initializer='he_uniform', name="fc1_value")(value_h)
            # value_h = Dense(256, activation='relu', kernel_initializer='he_uniform', name="fc2_value")(value_h)

            value_fn = tf.layers.dense(value_h, 1, name="vf")
            value_latent = value_h
            # [34*1]或者[1*34]的概率分布，policy是没有softmax映射的34维的数值向量
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(action_latent, value_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    # deterministic=true，取最大概率的动作，结论的性质；false，按概率分布取动作，探索的性质
    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        对应gym的step方法，走一步
        @param obs: 观测
        @param state: 状态
        @param mask: 掩膜
        @param deterministic:确定化
        @return:a,v,s0,neglogp
        """
        if deterministic:  # 概率对数似然，neglogp
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        # action_prob = self.proba_step(obs, state, mask)  # 胡牌概率
        # print("动作概率：{}, 选择动作：{}, value:{}, neglogp:{}".format(action_prob, action, value, neglogp))
        return action, value, self.initial_state, neglogp

    # 出牌前34张牌出每张牌的概率
    def proba_step(self, obs, state=None, mask=None):
        """
        概率分布
        @param obs: 观测
        @param state: 状态
        @param mask: 掩膜
        @return: 概率分布
        """
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    # step方法返回的动作的价值，[1]
    def value(self, obs, state=None, mask=None):
        """
        对动作的价值评估
        @param obs: 观测
        @param state: 状态
        @param mask: 掩膜
        @return: 价值
        """
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
