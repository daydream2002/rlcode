# -*- coding: utf-8 -*-
# @Time    : 18-9-29 星期六
# @Author  : Lei
# @Site    :
# @File    : interface_3.py
# @Software: PyCharm Community Edition


from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from tensorflow.contrib import layers
import numpy as np
import os, json, copy
import sys
import interface.interface_v1.feature_extract_v8 as feature_extract
import interface.interface_v1.model as model
import datetime
import time


# 将1-34的牌值转换为16进制数的形式
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
        print('Error !')


# 转换十进制到cards
def translate2(i):
    if 10 <= i <= 18:
        i = i + 7
    elif 19 <= i <= 27:
        i = i + 14
    elif 28 <= i <= 34:
        i = i + 21
    return i


# 推荐出牌
# handCards:当前手牌
# actions:当前副露
# allow_hand_cards:允许操作的手牌
# discarded_hands:弃牌
# round:轮数
def recommand_card(handCards, actions, allow_hand_cards, discarded_hands=None, round=None):
    t1 = datetime.datetime.now()
    feature_noking = feature_extract.calculate_noking_sys(handCards, actions)
    feature_noking.append(35)

    temp = model.model_choose1(ismyturn=True, list=feature_noking, hand_cards=handCards,
                               allow_hand_cards=allow_hand_cards)
    op_card = (int(temp // 10)) * 16 + (temp % 10)
    # handCards.remove(op_card)
    # t2 = datetime.datetime.now()
    # print("recommend_card_time:", t2-t1)
    # return handCards
    return op_card


# 推荐操作
# handCards:当前手牌
# actions:当前副露
# allow_op:允许进行的操作(吃、碰、杠）
# discarded_hands:弃牌
# round:轮数
def recommand_op(handCards, actions, op_card, allow_op, discarded_hands=None, round=None):
    t1 = datetime.datetime.now()
    feature_op = feature_extract.calculate_noking_op(handCards, actions, op_card)
    feature_op.append(8)
    op = model.model_choose1(ismyturn=False, list=feature_op, hand_cards=handCards, allow_op=allow_op)
    t2 = datetime.datetime.now()
    print("recommend_op_time:", t2 - t1)
    print("op=%d\n" % op)
    return op


if __name__ == '__main__':
    pass
    # print("#" * 50)
    #
    # # 出牌
    # handCards = [0x03, 0x04, 0x05, 0x11, 0x12, 0x16, 0x17, 0x18, 0x25, 0x26, 0x27,0x09, 0x09, 0x09]
    # # handCards = [0x16, 0x17, 0x18, 0x25, 0x26, 0x27, 0x03, 0x04, 0x05, 0x11, 0x12]
    # actions = []
    #
    # discarded_hands = [0,0,1,1,1,0,0,0,3,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]
    # allow_hand_cards=handCards
    # round=1
    #
    # print(recommand_card(handCards,actions,allow_hand_cards, discarded_hands, round))
    #
    #
    #
    #
    #
    # # 操作
    # handCards2 = [9,9,20,22,23,24,25,35,35,35]
    #
    # # handCards = [0x03, 0x04, 0x05, 0x09, 0x09, 0x11, 0x12, 0x16, 0x17, 0x18, 0x25, 0x26, 0x27]
    # actions2 = [[6,7,8]]
    #
    # outCard = 35
    # allow_op=[0,4,5]#允许执行的操作     0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'     初始化为：[0,4,5,6,7]（新疆麻将没有吃牌）
    # discarded_hands2 = [0, 0, 1, 1, 1, 0, 0, 0, 3, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,0, 0]
    # round2=1
    # # for i in range(100):
    # #     t = time.time()
    # #     print(RecommandOprateV2(handCards, actions, kingCard, outCard, fei_king, [0] * 34, 0, ))
    # #     print(time.time() - t)
    # print(recommand_op(handCards2,  actions, outCard, allow_op, discarded_hands2, round2))
