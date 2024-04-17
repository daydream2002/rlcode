import tensorflow as tf
import tensorflow.contrib as tf_contrib
from stable_baselines.common.policies import ActorCriticPolicy
weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


def flatten(x, data_format="channels_first"):
    return tf.layers.flatten(x, data_format=data_format)

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x, decay=0.99, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope, data_format="NCHW")
    # return tf.layers.batch_normalization(x, axis=1, momentum=0.99, epsilon=1e-05, trainable=is_training, name=scope)

def relu(x,name=None):
    return tf.nn.relu(x, name)

def conv2d(x, kernel_num, kernel_size=(3, 1), stride=(1, 1),padding='SAME',
           use_bias=True, scope='conv2d', dataformat="channels_first"):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=kernel_num,
                             kernel_size=kernel_size, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding, data_format=dataformat)
        return x

def resblock(x_init, kernel_num, kernel_size=(3, 1), stride=(1, 1), is_training=True,
             use_bias=True, downsample=False, layer_name='resblock', padding_mode='same'):
    with tf.variable_scope(layer_name):

        x = batch_norm(x_init, is_training, scope=layer_name+"_bn0")
        x = relu(x, name=layer_name+"_bn0_relu")

        if downsample :
            x = conv2d(x, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                       use_bias=use_bias, scope=layer_name+"_conv2d0")
            x_init = conv2d(x_init, kernel_num, kernel_size=1, stride=stride, use_bias=use_bias, scope=layer_name+ '_conv_init')

        else:
            x = conv2d(x, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                       use_bias=use_bias, scope=layer_name+"_conv2d0")

        x = batch_norm(x, is_training, scope=layer_name+"_bn1")
        x = relu(x, name=layer_name+"_bn1_relu")
        x = conv2d(x, kernel_num, kernel_size=kernel_size, stride=stride, padding=padding_mode,
                   use_bias=use_bias, scope=layer_name+'conv2d1')

        short_cut = tf.add(x, x_init, name=layer_name+"short_cut")
        return short_cut

def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope):
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv2d(shortcut, channels, kernel_size=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
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
    for i in range(1, block_num + 1):
        x = block(x, kernel_num=kernel_num, kernel_size=(3, 1), stride=stride, is_training=is_training,
                  layer_name=layer_name + str(i), padding_mode='same')
    return x

class Resnet18_policy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(Resnet18_policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            print("self.processed_obs的shape为：", self.processed_obs.shape)
            bn0 = batch_norm(self.processed_obs, scope="basic_block0_bn_0")
            relu0 = relu(bn0, name="bn0_relu")
            conv0 = conv2d(relu0, kernel_num=128, kernel_size=(3, 1), stride=(1, 1), padding='SAME', use_bias=True,
                           scope='basic_block0_bn_0_conv2d_0', dataformat="channels_first")

            # 18层卷积
            resNet_18 = make_layers(conv0, resblock, block_num=9, kernel_num=128, layer_name="basic_block")

            # 最后actor网络输出
            bn_actor = batch_norm(resNet_18, scope="bn_actor")
            relu_actor = relu(bn_actor)
            action_h = conv2d(relu_actor, 1, kernel_size=(1, 1), padding="same", scope="conv2d_actor")

            pi_latent = fully_conneted(action_h, 34, scope="pi_latent")
            pi_latent = relu(pi_latent, name="relu_pi_latent")
            # action_pi = Dense(34, activation="softmax", kernel_initializer='he_uniform', name="softmax_action")(
            #     action_h)
            action_latent = pi_latent

            # 最后value值网络的输出
            bn_value = batch_norm(resNet_18, scope="bn_value")
            relu_value = relu(bn_value)
            conv_value = conv2d(relu_value, 32, padding='same', scope='conv2d_value')

            value_h = fully_conneted(conv_value, 1024, scope="fc1_value")
            value_h = relu(value_h, name="relu_fc1_value")

            value_h = tf.layers.dense(value_h, units=256, kernel_initializer=weight_init, name="value_h",
                                      kernel_regularizer=weight_regularizer, use_bias=True, activation=tf.nn.relu)
            value_h = relu(value_h, name="relu_value_h")

            # value_h = Dense(1024, activation='relu', kernel_initializer='he_uniform', name="fc1_value")(value_h)
            # value_h = Dense(256, activation='relu', kernel_initializer='he_uniform', name="fc2_value")(value_h)

            # value_h = pi_latent
            value_fn = tf.layers.dense(value_h, 1, name="vf")
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