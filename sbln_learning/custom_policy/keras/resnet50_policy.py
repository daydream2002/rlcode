#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : resnet50_policy.py
# @Description: resnet50网络结构
from stable_baselines.common.policies import MlpPolicy,CnnPolicy,FeedForwardPolicy
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
import numpy as np

import warnings
from itertools import zip_longest


from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.merge import Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2



def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=2, stride=1, init_scale=1, **kwargs))
    #layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    #layer_3 = activ(conv(layer_2, 'c3', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_1)
    temp = activ(linear(layer_3, 'fc1', n_hidden=34, init_scale=1))
    return temp

    # activ = tf.nn.relu
    # layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    # layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    # layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    # layer_3 = conv_to_fc(layer_3)
    # return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))



def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    print("flat_observations",flat_observations)
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value

def _build_residual_block( x, index):
        """Fetches rows from a Bigtable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            x: An open Bigtable Table instance.
            index: A sequence of strings representing the key of each table row
                to fetch.

        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:

            {'Serak': ('Rigel VII', 'Preparer'),
             'Zim': ('Irk', 'Invader'),
             'Lrrr': ('Omicron Persei 8', 'Emperor')}

            If a key from the keys argument is missing from the dictionary,
            then that row was not found in the table.

        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """
        in_x = x
        res_name = "res" + str(index)
        x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=128, kernel_size=[3, 1], padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
                   name=res_name + "_conv1-" + str(3 + index) + "-" + str(3))(x)

        x = BatchNormalization(axis=1, name=res_name + "_batchnorm2")(x)
        x = Activation("relu", name=res_name + "_relu2")(x)
        x = Conv2D(filters=128, kernel_size=[3, 1], padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
                   name=res_name + "_conv2-" + str(3 + index) + "-" + str(3))(x)

        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu3")(x)

        return x



class Resnet50_policy(FeedForwardPolicy):
    '''
        ResNet50残差网络源码
        className:Resnet50_policy
        fileName:resnet50_policy.py
    '''
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="resnet", **kwargs):

        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)


        with tf.variable_scope("modeld", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                print("self.processed_obs",self.processed_obs)

                x = Conv2D(filters=128, kernel_size=[3, 1], padding="same",
                           data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
                           name="input_conv-" + str(5) + "-" + str(256))(self.processed_obs)
                x = BatchNormalization(axis=1, name="input_batchnorm")(x)
                x = Activation("relu", name="input_relu")(x)

                for i in range(50):
                    x = _build_residual_block(x, i + 1)

                latent_policy = Conv2D(filters=1, kernel_size=[1, 1], padding="same",
                           data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
                           name="policy_conv")(x)
                latent_policy = BatchNormalization(axis=1, name="output_batchnorm")(latent_policy)
                pi_latent = Flatten(name="pi_latent_flatten")(latent_policy)
                pi_latent = Dense(34, activation="relu", name="pi_latent_dense_relu")(pi_latent)


                latent_value = Conv2D(filters=32, kernel_size=[3, 1], padding="valid",
                           data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
                           name="output_conv")(x)

                fc1 = Flatten()(latent_value)
                fc2 = Dense(1024, activation='relu')(fc1)
                vf_latent = Dense(256, activation='relu')(fc2)


            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
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
        print(self.sess.run(self.policy_proba))
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

# from stable_baselines import PPO1
# from mahEnv import MahEnv
# env = MahEnv()
# obs = env.reset()
#
# model = PPO1(selfPolicy, env, verbose=1)
# # model.learn(total_timesteps=1000000)
# # model.save("./s-v0-1000000")
#
#
#
# # model = model.load("./mj-v0-10000.zip", env)
# model.learn(total_timesteps=10000)
# model.save("./s-v0-10000")
#
# reward_sum = 0
# for i in range(100000):
#     action, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(action)
#     reward_sum += reward
#     env.render()
#     if done:
#         print(reward_sum)
#         reward_sum = 0.0
#         obs = env.reset()