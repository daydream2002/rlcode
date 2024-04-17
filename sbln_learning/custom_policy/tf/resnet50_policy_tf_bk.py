import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        # tf.contrib.layers.fully_connected()
        x = conv_to_fc(x)
        x = relu(linear(x, scope+"_fc0", n_hidden=units, init_scale=np.sqrt(2)))
        return x

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x, decay=0.99, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope, data_format="NCHW")
    # return tf.layers.batch_normalization(x, axis=1, momentum=0.99, epsilon=1e-05, trainable=is_training, name=scope)

def relu(x,name=None):
    return tf.nn.relu(x, name)

def conv2d(x, kernel_num, kernel_size=(3, 1), stride=(1, 1), padding="SAME",
           use_bias=True, scope="conv2d", dataformat="channels_first"):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=kernel_num,
                             kernel_size=kernel_size, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding, data_format=dataformat)
        return x
    # x = conv(x, scope, n_filters=kernel_num, filter_size=kernel_size, stride=stride,
    #          pad=padding, init_scale=np.sqrt(2), data_format=dataformat, use_bias=use_bias)
    # return x

def resblock(x_init, kernel_num, kernel_size=(3, 1), stride=(1, 1), is_training=True,
             one_dim_bias=False, downsample=False, layer_name='resblock', padding_mode='SAME'):
    with tf.variable_scope(layer_name):

        if downsample :
            x = conv2d(x_init, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                       use_bias=one_dim_bias, scope=layer_name + "_conv2d0")
            x = batch_norm(x, scope=layer_name+"bn_0")
            x = relu(x, name=layer_name + "_conv2d0_relu0")

            x_init = conv2d(x_init, kernel_num, kernel_size=1, stride=stride, use_bias=one_dim_bias, scope=layer_name + '_conv_init')

        else:
            x = conv2d(x_init, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                       use_bias=one_dim_bias, scope=layer_name + "_conv2d0")
            x = batch_norm(x, scope=layer_name + "bn_0")
            x = relu(x, name=layer_name+"_relu0")

        x = conv2d(x, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                   use_bias=one_dim_bias, scope=layer_name + 'conv2d1')
        x = batch_norm(x, scope=layer_name + "bn_1")
        x = relu(x, name=layer_name + "_relu1")

        short_cut = tf.add(x, x_init, name=layer_name+"short_cut")
        return short_cut

def make_layers(x, block, block_num=50, kernel_num=128, stride=(1, 1), layer_name="basic_resblock", is_training=True):
    for i in range(1, block_num + 1):
        x = block(x, kernel_num=kernel_num, kernel_size=(3, 1), stride=stride, is_training=is_training,
                  layer_name=layer_name + str(i), padding_mode='SAME')
    return x

class Resnet50_policy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(Resnet50_policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            print("self.processed_obs的shape为：", self.processed_obs.shape)

            conv0 = conv2d(self.processed_obs, kernel_num=128, kernel_size=(3, 1), stride=(1, 1), padding='SAME', use_bias=True,
                           scope='conv2d_0')
            bn0 = batch_norm(conv0, scope='batch_norm_0')
            relu0 = relu(bn0, name="bn0_relu0")
            print("relu0_shape:",relu0.shape)
            # 50层卷积
            resNet_50 = make_layers(relu0, resblock, block_num=25, kernel_num=128, layer_name="basic_block")

            # 最后actor网络输出
            action_conv = conv2d(resNet_50, 1, kernel_size=(1, 1), padding="SAME", scope="conv2d_actor")
            action_bn = batch_norm(action_conv, scope="actor_bn")
            relu_actor = relu(action_bn, "actor_relu")

            pi_latent = fully_conneted(relu_actor, 34, scope="pi_latent")
            # action_pi = Dense(34, activation="softmax", kernel_initializer='he_uniform', name="softmax_action")(
            #     action_h)
            action_latent = pi_latent

            # 最后value值网络的输出
            conv_value = conv2d(resNet_50, 32, padding='SAME', scope='conv2d_value')
            bn_value = batch_norm(conv_value, scope='bn_value')
            relu_value = relu(bn_value, "relu_value")


            value_h = fully_conneted(relu_value, 1024, scope="fc1_value")
            value_h = relu(linear(value_h, scope="value_h", n_hidden=256, init_scale=np.sqrt(2)))

            value_fn = linear(value_h, scope="vf", n_hidden=1)
            value_latent = value_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(action_latent, value_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})