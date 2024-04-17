# -*- coding: utf-8 -*-
# @Time    : 20-5-21 星期四
# @Author  : zengw
# @Site    :
# @File    : model.py
# @Software: PyCharm Community Edition
# @Function:
# 去掉代码冗余部分
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from tensorflow.contrib import layers

import numpy as np
import logging
import os

current_path = os.path.split(os.path.realpath(__file__))[0]
print(current_path)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print("model-智一")
# 综合决策模型构建
g1 = tf.Graph()
with g1.as_default():
    sess1 = tf.Session(config=config, graph=g1)
    # sess1.run(tf.global_variables_initializer())
    # model running
    print('---testing---')
    import os

    print(os.getcwd())  # 获得当前工作目录
    try:
        saver = tf.compat.v1.train.import_meta_graph("interface/interface_v1/save/20180918_sys_noking_100.ckpt.meta")
        saver.restore(sess1, "interface/interface_v1/save/20180918_sys_noking_100.ckpt")
    except:
        saver = tf.train.import_meta_graph(current_path + "/save/20180918_sys_noking_100.ckpt.meta")
        saver.restore(sess1, current_path + "/save/20180918_sys_noking_100.ckpt")
    print('model1 restore !')

    x1 = g1.get_tensor_by_name("inputs/x:0")

# 动作决策模型构建
g2 = tf.Graph()
with g2.as_default():
    sess2 = tf.Session(config=config, graph=g2)

    # model running
    print('---testing---')
    try:
        saver = tf.train.import_meta_graph("interface/interface_v1/save/20180918_action_noking_100.ckpt.meta")
        # saver = tf.train.import_meta_graph("interface/interface_v1/save/20180918_sys_noking_100.ckpt.meta")
        saver.restore(sess2, "interface/interface_v1/save/20180918_action_noking_100.ckpt")
    except:

        saver = tf.train.import_meta_graph(current_path + "/save/20180918_action_noking_100.ckpt.meta")
        # saver = tf.train.import_meta_graph("C:/Users/adminstor/Desktop/Zonst/RL/shangrao_rl_v1/interface"
        #                                    "/interface_v1/save/20180918_sys_noking_100.ckpt.meta")
        saver.restore(sess2, current_path + "/save/20180918_action_noking_100.ckpt")
    print('model2 restore !')
    x2 = g2.get_tensor_by_name("inputs/x:0")


def model_choose1(ismyturn, list, hand_cards,
                  allow_hand_cards=[1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37,
                                    38, 39, 40, 41, 49, 50, 51, 52, 53, 54, 55]
                  , allow_op=[0, 1, 2, 3, 4, 5, 6, 7]):
    """
    模型选择
    Args:
        ismyturn: 是否轮到我的回合
        list: 列表
        hand_cards: 手牌
        allow_hand_cards: 允许操作的手牌
        allow_op: 允许的动作操作

    Returns: 动作

    """
    testData = np.array(list)
    # data input
    x_data = np.array(testData[0:-1])
    sign = 1
    test_data = x_data
    # print(test_data.shape)
    if ismyturn:
        L1_1 = g1.get_tensor_by_name("Softmax:0")
        # tf.argmax(L1, 1)
        ret1 = sess1.run(L1_1, feed_dict={x1: test_data.reshape(1, test_data.shape[0])})

        hand_filter = card_filter(hand_cards, 34)
        allow_hand_filter = card_filter(allow_hand_cards, 34)

        ret1 = ret1 * hand_filter * allow_hand_filter  ##过滤推荐手牌

        temp = ret1.argmax() + 1

        result = translate(temp)

        # print('our decision:%d | ' % result + 'best decision:%d | ' % result)

    else:
        L1_2 = g2.get_tensor_by_name("Softmax:0")
        # tf.argmax(L1, 1)
        ret2 = sess2.run(L1_2, feed_dict={x2: test_data.reshape(1, test_data.shape[0])})

        op = op_filter(allow_op, 8)  # 过滤推荐操作
        ret2 = ret2 * op
        decision = ret2.argmax()
        op_out_table = {0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'}
        print('our decision:%s | ' % op_out_table[decision] + 'best decision:%s | ' % op_out_table[decision])
        result = decision
    return result


def translate(i):
    if 1 <= i <= 9:
        return i
    elif 10 <= i <= 18:
        return i + 1
    elif 19 <= i <= 27:
        return i + 2
    elif 28 <= i <= 34:
        return i + 3
    else:
        logging.info('Error !')


def translate2(i):  # 转换十进制到cards 1-34
    if 10 <= i <= 18:
        i = i + 7
    elif 19 <= i <= 27:
        i = i + 14
    elif 28 <= i <= 34:
        i = i + 21
    return i


def translate3(op_card):  # 16进制op_card转换到 0-33 34转换    #######################################
    if 1 <= op_card <= 9:
        op_card = op_card - 1
    elif 17 <= op_card <= 25:
        op_card = op_card - 8
    elif 33 <= op_card <= 41:
        op_card = op_card - 15
    elif 49 <= op_card <= 55:
        op_card = op_card - 22
    elif op_card == 255:
        op_card = 34
    return op_card


def card_filter(cards, shape):
    """
    过滤推荐手牌
    Args:
        cards: 手牌
        shape: 维度

    Returns:

    """
    filter = np.ones(shape)
    card_translate = [translate3(i) for i in cards]
    for i in range(shape):
        if i not in card_translate:
            filter[i] = 0
    return filter


def op_filter(op, shape):
    """
    过滤推荐操作
    Args:
        op: 动作
        shape: 维度

    Returns:

    """
    filter = np.ones(shape)

    for i in range(shape):
        if i not in op:
            filter[i] = 0
    return filter


# 2019.1.24
# 调用监督模型，为强化学习缩减搜索范围

import interface.interface_v1.feature_extract_v8 as feature_extract


def model_based_rl(handCards, actions, allow_hand_cards, ):
    """
    强化学习版出牌模型选择
    Args:
        handCards: 手牌
        actions: 副露
        allow_hand_cards: 允许操作的手牌

    Returns:出牌推荐

    """
    feature_noking = feature_extract.calculate_noking_sys(handCards, actions)
    feature_noking.append(35)

    list = feature_noking
    hand_cards = handCards
    allow_hand_cards = allow_hand_cards

    testData = np.array(list)
    # data input
    x_data = np.array(testData[0:-1])
    sign = 1
    test_data = x_data
    print(test_data.shape)

    L1_1 = g1.get_tensor_by_name("Softmax:0")
    # tf.argmax(L1, 1)
    ret1 = sess1.run(L1_1, feed_dict={x1: test_data.reshape(1, test_data.shape[0])})

    hand_filter = card_filter(hand_cards, 34)
    allow_hand_filter = card_filter(allow_hand_cards, 34)

    ret1 = ret1 * hand_filter * allow_hand_filter  ##过滤推荐手牌
    # temp = ret1.argmax() + 1
    # result = translate(temp)
    # print('our decision:%d | ' % result + 'best decision:%d | ' % result)

    return ret1
