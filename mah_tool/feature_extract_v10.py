#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : feature_extract_v10.py
# @Description:特征工程
import json
import sys
import copy
import torch

from mah_tool import tool
import os
from mah_tool.so_lib.sr_xt_ph import pinghu
from mah_state import RL_state
import numpy as np
import random
import time
from mah_tool.suphx_extract_features.search_tree_ren import SearchInfo
from mah_tool.so_lib.srmj_v5.shangraoMJ_v5 import get_xts_list, get_xts_fan_list

self_l = []


# 1通常手向听数
def wait_types_comm(tile_list, suits):
    """
    通常手向听数
    @param tile_list: 牌墙
    @param suits: 副露
    @return: 平胡向听数
    """
    common_waiting = {'common_waiting0': 0,
                      'common_waiting1': 0,
                      'common_waiting2': 0,
                      'common_waiting3': 0,
                      'common_waiting4': 0,
                      'common_waiting5': 0,
                      'common_waiting6': 0,
                      'common_waiting7': 0,
                      'common_waiting8': 0,
                      'common_waiting9': 0,
                      'common_waiting10': 0,
                      'common_waiting11': 0,
                      'common_waiting12': 0,
                      'common_waiting13': 0,
                      }
    tempList = tool.deepcopy(tile_list)
    # print tempList
    wait_num = 14

    # 考虑副露的情况
    suits_len = len(suits)
    wait_num -= suits_len * 3

    sz = 0  # 顺子数
    kz = 0  # 刻子数
    dzk = 0  # 搭子 aa
    dzs12 = 0  # 搭子ab
    dzs13 = 0  # 搭子ac
    # 判断顺子数
    i = 0
    while i <= len(tempList) - 3:
        if tempList[i] & 0xF0 != 0x30 and tempList[i] + 1 in tempList and tempList[i] + 2 in tempList:
            # print(tempList[i], "i")
            wait_num -= 3
            sz += 1
            card0 = tempList[i]
            card1 = tempList[i] + 1
            card2 = tempList[i] + 2
            tempList.remove(card0)
            tempList.remove(card1)
            tempList.remove(card2)
        else:
            i += 1
    # print(tempList)

    # 判断刻子数
    j = 0
    while j <= len(tempList) - 3:
        if tempList[j + 1] == tempList[j] and tempList[j + 2] == tempList[j]:
            # print(tempList[j], "j")
            wait_num -= 3
            kz += 1
            card = tempList[j]
            tempList.remove(card)
            tempList.remove(card)
            tempList.remove(card)
        else:
            j += 1
    # print(tempList)

    # 判断搭子aa
    x = 0
    while x <= len(tempList) - 2:
        # print(tempList[x], "x")
        if tempList[x + 1] == tempList[x]:
            dzk += 1
            # wait_num -=2
            card = tempList[x]
            tempList.remove(card)
            tempList.remove(card)
        else:
            x += 1

    # 判断搭子ab ac
    k = 0
    while k <= len(tempList) - 2:
        if tempList[k] & 0xF0 != 0x30:
            # print(tempList[k], "k")
            if tempList[k] + 1 in tempList:
                # wait_num -= 2
                dzs12 += 1
                card0 = tempList[k]
                card1 = tempList[k] + 1
                tempList.remove(card0)
                tempList.remove(card1)
            elif tempList[k] + 2 in tempList:
                # wait_num -= 2
                dzs13 += 1
                card0 = tempList[k]
                card2 = tempList[k] + 2
                tempList.remove(card0)
                tempList.remove(card2)
            else:
                k += 1
        else:
            k += 1
    if dzk > 0:  # 如果搭子aa>0 ,取其中一个作为将牌，并且向听数-2
        wait_num -= 2
        if dzk - 1 + dzs12 + dzs13 - (4 - sz - kz - suits_len) <= 0:  # 如果搭子加面子<=4,向听数再减去搭子数*2
            wait_num -= (dzk - 1 + dzs12 + dzs13) * 2
        else:
            wait_num -= (4 - sz - kz - suits_len) * 2  # 否则 向听数只减去多余的，即向听数减到为0
    else:  # 如果搭子aa=0，取一张单牌作为将的候选，向听数-1
        wait_num -= 1
        if dzk + dzs12 + dzs13 - (4 - sz - kz - suits_len) <= 0:  # 向上同理
            wait_num -= (dzk + dzs12 + dzs13) * 2
        else:
            wait_num -= (4 - sz - kz - suits_len) * 2
    # print(tempList)

    common_waiting['common_waiting' + str(wait_num)] = 1
    return wait_num
    # print(common_waiting)


# 1 平胡向听数  任航师兄版本
def wait_types_comm_king(tile_list, suits, jing_card=0):
    """
    平胡向听数  任航师兄版本
    @param tile_list: 牌墙
    @param suits: 副露
    @param jing_card: 宝牌
    @return: 平胡向听数
    """
    xt_ph = pinghu(tile_list, suits, jing_card).get_xts()
    # 向听数此时为0，为胡牌情况
    if xt_ph == 0:
        return 0
    elif jing_card not in tile_list:
        return xt_ph
    else:
        # 当xt_ph 为1时，宝又在手牌中，需要考虑宝还原的情况
        xt_ph_no_king = pinghu(tile_list, suits, 0).get_xts()
        return min(xt_ph, xt_ph_no_king)


# 2 七对的向听数判断
def wait_types_7(tile_list, suits=[], jing_card=0):
    """
    七对的向听数判断
    @param tile_list: 牌墙
    @param suits: 副露
    @param jing_card: 宝牌
    @return: 七对向听数
    """
    _tile_list = tool.deepcopy(tile_list)

    jing_count = _tile_list.count(jing_card)
    for i in range(jing_count):
        _tile_list.remove(jing_card)

    if suits != []:
        wait_num = 7  # 如果副露有牌，则不能做七对
        return wait_num
    else:
        wait_num = 7  # 表示向听数
        _tile_list.sort()  # L是临时变量，传递tile_list的值
        L = set(_tile_list)
        for i in L:
            # print("the %d has %d in list" % (i, tile_list.count(i)))
            if _tile_list.count(i) >= 2:
                wait_num -= 1

        return max(0, wait_num - jing_count)


# 返回去精牌后的手牌，四个牌的数量，三个牌数量，两个牌数量，精牌数量
def get_four_three_two_card_jing_nums(tile_list, jing_card=0):
    """
    返回去精牌后的手牌，四个牌的数量，三个牌数量，两个牌数量，精牌数量
    @param tile_list: 牌墙
    @param jing_card: 宝牌
    @return: 去精后
    """
    _tile_list = tool.deepcopy(tile_list)
    jing_count = _tile_list.count(jing_card)

    for i in range(jing_count):
        _tile_list.remove(jing_card)

    si_card_num = 0
    san_card_num = 0
    er_card_num = 0
    L = list(set(_tile_list))
    L.sort(key=_tile_list.index)

    for i in L:
        _count = _tile_list.count(i)
        if _count == 4:
            si_card_num += 1
        if _count == 3:
            san_card_num += 1
        if _count == 2:
            er_card_num += 1

    return _tile_list, si_card_num, san_card_num, er_card_num, jing_count


# 2-2 豪华七对的向听数判断
def wait_types_haohua7(tile_list, suits=[], jing_card=0):
    """
    豪华七对的向听数判断
    @param tile_list: 牌墙
    @param suits: 副露
    @param jing_card: 宝牌
    @return: 豪华七对向听数
    """
    _tile_list = tool.deepcopy(tile_list)

    if len(suits) > 0 or len(_tile_list) != 14:  # 当副露不为空时,不是七对
        return 7

    wait_nums = 7
    _tile_list, si_card_num, san_card_num, er_card_num, jing_count = get_four_three_two_card_jing_nums(_tile_list,
                                                                                                       jing_card)
    wait_nums -= (si_card_num * 2 + san_card_num + er_card_num)  # 减去向听数

    signal_nums = len(_tile_list) - si_card_num * 4 - san_card_num * 3 - er_card_num * 2 + max((san_card_num - 1),
                                                                                               0) + jing_count  # 精牌也算单张

    # 如果没有四个相同的牌，需要增加向听数
    if si_card_num == 0:
        if san_card_num == 0:  # 只有aa， 需要增加1个向听
            if signal_nums < 2:  # 单张不满足2张，需要拆对， 向听+1
                wait_nums += 2
            else:
                wait_nums += 1
        else:  # 有刻子时
            if signal_nums < 1:  # 也需要拆对 +1
                wait_nums += 1
    return max(0, wait_nums - jing_count)


# 3 十三浪的向听数判断
def wait_types_13(tile_list, suits=[], jing_card=0):  # 十三烂中仅作宝还原
    """
    十三烂的向听数判断，手中十四张牌中，序数牌间隔大于等于3，字牌没有重复所组成的牌形
    先计算0x0,0x1,0x2中的牌，起始位a，则a+3最多有几个，在wait上减，0x3计算不重复最多的数
    @param tile_list: 牌墙
    @param suits: 副露
    @param jing_card:宝牌
    @return: 十三烂向听数
    """
    # 十三浪的向听数判断，手中十四张牌中，序数牌间隔大于等于3，字牌没有重复所组成的牌形
    # 先计算0x0,0x1,0x2中的牌，起始位a，则a+3最多有几个，在wait上减，0x3计算不重复最多的数
    wait_13lan = {
        'thirteen_waiting0': 0,
        'thirteen_waiting1': 0,
        'thirteen_waiting2': 0,
        'thirteen_waiting3': 0,
        'thirteen_waiting4': 0,
        'thirteen_waiting5': 0,
        'thirteen_waiting6': 0,
        'thirteen_waiting7': 0,
        'thirteen_waiting8': 0,
        'thirteen_waiting9': 0,
        'thirteen_waiting10': 0,
        'thirteen_waiting11': 0,
        'thirteen_waiting12': 0,
        'thirteen_waiting13': 0,
        'thirteen_waiting14': 0,
    }
    wait_num = 14  # 表示向听数
    max_num_wait = 0
    if suits != []:
        wait_num = 14
        return wait_num
    else:
        L = set(tile_list)  # 去除重复手牌
        L_num0 = []  # 万数牌
        L_num1 = []  # 条数牌
        L_num2 = []  # 筒数牌
        for i in L:
            if i & 0xf0 == 0x30:
                # 计算字牌的向听数
                wait_num -= 1
            if i & 0xf0 == 0x00:
                L_num0.append(i & 0x0f)
            if i & 0xf0 == 0x10:
                L_num1.append(i & 0x0f)
            if i & 0xf0 == 0x20:
                L_num2.append(i & 0x0f)
        wait_num -= calculate_13(L_num0)
        # 减去万数牌的向听数
        wait_num -= calculate_13(L_num1)
        # 减去条数牌的向听数
        wait_num -= calculate_13(L_num2)
        # 减去筒数牌的向听数
        # print(L)
        # print(L_num0)
        # print(L_num1)
        # print(L_num2)
        # print(wait_num)
        wait_13lan['thirteen_waiting' + str(wait_num)] = 1
        # print(wait_13lan)
        return wait_num


# 4 九幺的向听数判断
def wait_types_19(tile_list, suits, jing_card=0):
    """
    九幺的向听数判断
    @param tile_list: 牌墙
    @param suits: 副露
    @param jing_card: 宝牌
    @return: 九幺向听数
    """
    # 九幺的向听数判断，由一、九这些边牌、东、西、南、北、中、发、白这些风字牌中的任意牌组成的牌形。以上这些牌可以重复
    wait_19 = {
        'one_nine_waiting0': 0,
        'one_nine_waiting1': 0,
        'one_nine_waiting2': 0,
        'one_nine_waiting3': 0,
        'one_nine_waiting4': 0,
        'one_nine_waiting5': 0,
        'one_nine_waiting6': 0,
        'one_nine_waiting7': 0,
        'one_nine_waiting8': 0,
        'one_nine_waiting9': 0,
        'one_nine_waiting10': 0,
        'one_nine_waiting11': 0,
        'one_nine_waiting12': 0,
        'one_nine_waiting13': 0,
        'one_nine_waiting14': 0,
    }
    wait_num = 14  # 表示向听数
    _suits = tool.deepcopy(suits)
    for i in _suits:
        if i[0] != i[1]:
            return 14
        else:
            if i[0] & 0xf0 == 0x30 or i[0] & 0x0f == 0x01 or i[0] & 0x0f == 0x09:
                wait_num -= 3
            else:
                return 14  # 如果非1和9及字牌的刻子

    for i in tile_list:
        if i & 0x0f == 0x01 or i & 0x0f == 0x09 or i & 0xf0 == 0x30:
            wait_num -= 1
    wait_19['one_nine_waiting' + str(wait_num)] = 1
    # print(wait_19)
    return wait_num


def calculate_13(tiles):
    """
    计算十三烂的数牌最大向听数
    @param tiles: 牌墙
    @return: 十三烂的数牌最大向听数
    """
    # 计算十三浪的数牌最大向听数
    if len(tiles) == 0:
        return 0
    if len(tiles) == 1:
        return 1
    if len(tiles) == 2:
        if tiles[0] + 3 <= tiles[1]:
            return 2
        else:
            return 1
    if len(tiles) >= 3:
        return max((tiles.count(1) + tiles.count(4) + tiles.count(7)),
                   (tiles.count(1) + tiles.count(4) + tiles.count(8)),
                   (tiles.count(1) + tiles.count(4) + tiles.count(9)),
                   (tiles.count(1) + tiles.count(5) + tiles.count(8)),
                   (tiles.count(1) + tiles.count(5) + tiles.count(9)),
                   (tiles.count(1) + tiles.count(6) + tiles.count(9)),
                   (tiles.count(2) + tiles.count(5) + tiles.count(8)),
                   (tiles.count(2) + tiles.count(5) + tiles.count(9)),
                   (tiles.count(2) + tiles.count(6) + tiles.count(9)),
                   (tiles.count(3) + tiles.count(6) + tiles.count(9)))


# 5 操作牌值
def get_op_cards(op_cards=0):
    """
    操作牌值
    @param op_cards: 操作牌
    @return:
    """
    return op_cards


# 6 7左吃操作后普通手向听数减少
def waitnums_chi_left(tile_list=[], suits=[], op_cards=0):
    """
    Return 左吃操作后普通手向听数减少
    :param tile_list: input hand cards
    :param suits: input suits cards
    :param op_cards: op_cards
    :return: chi dec and inc numbers
    """
    _suits = tool.deepcopy(suits)
    __suits = tool.deepcopy(suits)
    _tile_list = tool.deepcopy(tile_list)
    __tile_list = tool.deepcopy(tile_list)
    if ((op_cards + 1) in tile_list) and ((op_cards + 2) in tile_list):
        _suits.append([op_cards, op_cards + 1, op_cards + 2])
        _tile_list.remove(op_cards + 1)
        _tile_list.remove(op_cards + 2)
        chi_left_nums = wait_types_comm(_tile_list, _suits) - wait_types_comm(__tile_list, __suits)
        if chi_left_nums >= 0:
            # 向听数减少
            return [0, chi_left_nums]
        else:
            # 向听数增加
            return [abs(chi_left_nums), 0]
    else:
        return [0, 0]


# 8 9右吃操作后普通手向听数减少
def waitnums_chi_right(tile_list=[], suits=[], op_cards=0):
    """
    Return 右吃操作后普通手向听数减少
    :param tile_list: input hand cards
    :param suits: input suits cards
    :param op_cards: op_cards
    :return: chi dec and inc numbers
    """
    # 判断是否过
    _suits = tool.deepcopy(suits)
    __suits = tool.deepcopy(suits)
    _tile_list = tool.deepcopy(tile_list)
    __tile_list = tool.deepcopy(tile_list)
    if ((op_cards - 1) in tile_list) and ((op_cards - 2) in tile_list):
        _suits.append([op_cards, op_cards - 1, op_cards - 2])
        _tile_list.remove(op_cards - 1)
        _tile_list.remove(op_cards - 2)
        chi_right_nums = wait_types_comm(_tile_list, _suits) - wait_types_comm(__tile_list, __suits)
        if chi_right_nums >= 0:
            # 向听数减少
            return [0, chi_right_nums]
        else:
            # 向听数增加
            return [abs(chi_right_nums), 0]
    else:
        return [0, 0]


# 10 11中吃操作后普通手向听数减少
def waitnums_chi_mid(tile_list=[], suits=[], op_cards=0):
    """
    Return 中吃操作后普通手向听数减少
    :param tile_list: input hand cards
    :param suits: input suits cards
    :param op_cards: op_cards
    :return: chi dec and inc numbers
    """
    _suits = tool.deepcopy(suits)
    __suits = tool.deepcopy(suits)
    _tile_list = tool.deepcopy(tile_list)
    __tile_list = tool.deepcopy(tile_list)
    if ((op_cards + 1) in tile_list) and ((op_cards - 1) in tile_list):
        _suits.append([op_cards, op_cards + 1, op_cards - 1])
        _tile_list.remove(op_cards + 1)
        _tile_list.remove(op_cards - 1)
        chi_mid_nums = wait_types_comm(_tile_list, _suits) - wait_types_comm(__tile_list, __suits)
        if chi_mid_nums >= 0:
            # 向听数减少
            return [0, chi_mid_nums]
        else:
            # 向听数增加
            return [abs(chi_mid_nums), 0]
    else:
        return [0, 0]


# 12 碰操作后普通手向听数减少
def waitnums_dec_peng_gang(tile_list=[], suits=[], op_cards=0):
    """
    Return 中吃操作后普通手向听数减少
    :param tile_list: input hand cards
    :param suits: input suits cards
    :param op_cards: op_cards
    :return: peng gang dec numbers
    """
    _suits = tool.deepcopy(suits)
    __suits = tool.deepcopy(suits)
    _tile_list = tool.deepcopy(tile_list)
    __tile_list = tool.deepcopy(tile_list)

    if _tile_list.count(op_cards) == 2:
        _suits.append([op_cards, op_cards, op_cards])
        _tile_list.remove(op_cards)
        _tile_list.remove(op_cards)
        peng_nums = wait_types_comm(_tile_list, _suits) - wait_types_comm(__tile_list, __suits)
        return [abs(peng_nums), 0]

    elif _tile_list.count(op_cards) == 3:
        _suits.append([op_cards, op_cards, op_cards, op_cards])
        _tile_list.remove(op_cards)
        _tile_list.remove(op_cards)
        _tile_list.remove(op_cards)
        gang_nums = wait_types_comm(_tile_list, _suits) - wait_types_comm(__tile_list, __suits)
        return [0, abs(gang_nums)]
    else:
        return [0, 0]


# 13 碰杠操作后91的向听数减少
def waitnums_19_change(tile_list=[], suits=[], op_cards=0):
    """
    Return 碰杠操作后91的向听数减少
    :param tile_list: input hand cards
    :param suits: input suits cards
    :param op_cards: op_cards
    :return: peng gang 91 [dec numbers,inc numbers]

    """
    _suits = tool.deepcopy(suits)
    __suits = tool.deepcopy(suits)
    _tile_list = tool.deepcopy(tile_list)
    __tile_list = tool.deepcopy(tile_list)
    # print(op_cards,_tile_list,_tile_list.count(op_cards))
    # 判断是否过
    if _tile_list.count(op_cards) == 2:
        _suits.append([op_cards, op_cards, op_cards])
        _tile_list.remove(op_cards)
        _tile_list.remove(op_cards)
        wait91_change = wait_types_19(_tile_list, _suits) - wait_types_19(__tile_list, __suits)
        if wait91_change <= 0:
            return [abs(wait91_change), 0]
        else:
            return [0, abs(wait91_change)]
    else:
        return [0, 0]


def translate(op_card):  # 转换op_card函数    #######################################
    """
    转换op_card函数，十六进制->十进制0到34
    @param op_card: 麻将牌
    @return: 十进制表示的麻将牌
    """
    if op_card >= 1 and op_card <= 9:
        op_card = op_card
    elif op_card >= 17 and op_card <= 25:
        op_card = op_card - 7
    elif op_card >= 33 and op_card <= 41:
        op_card = op_card - 14
    elif op_card >= 49 and op_card <= 55:
        op_card = op_card - 21
    else:
        print("error")
    return op_card


# 5-9 万，条，筒，风，箭个数

def numOfCards(handcards):
    """
    万，条，筒，风，箭个数
    @param handcards: 手牌
    @return: 一维列表[万，条，筒，风，箭个数]
    """
    feature = [0, 0, 0, 0, 0]
    for d in handcards:
        if (d & 0xF0) / 16 < 3:
            feature[int((d & 0xF0) / 16)] += 1
        elif (d & 0xF0) / 16 == 3 and (d & 0x0F) < 5:
            feature[3] += 1
        else:
            feature[4] += 1
    # print feature
    return feature


# # 10 -43 万的AA
def numofcard(handcards):
    """
    万的AA
    @param handcards: 手牌
    @return: 万的AA列表
    """
    feature = [0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               ]
    for d in handcards:
        num = d & 0x0F
        # 1-9万 【0-8】
        if (d & 0xF0) / 16 == 0:
            feature[num - 1] += 1
        # 1-9条  【9-17】
        elif (d & 0xF0) / 16 == 1:
            feature[8 + num] += 1
        # 1-9饼 【18-26】
        elif (d & 0xF0) / 16 == 2:
            feature[17 + num] += 1
        # 字牌 【27-33】
        elif (d & 0xF0) / 16 == 3:
            # print handcards
            # print num
            feature[26 + num] += 1
    # print feature
    return feature


# 44-67
def ab(handcards):
    """
    AB搭子
    @param handcards: 手牌
    @return: AB搭子列表
    """
    feature = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    k = 0
    while k <= len(handcards) - 2:
        num = handcards[k] & 0x0F
        # 万牌 ab 【0-7】
        if handcards[k] & 0xF0 == 0x00 and handcards[k] + 1 in handcards:
            feature[num - 1] += 1
        # 条 ab 【8-15】
        elif handcards[k] & 0xF0 == 0x10 and handcards[k] + 1 in handcards:
            feature[7 + num] += 1
        # 筒 ab 【16-23】
        elif handcards[k] & 0xF0 == 0x20 and handcards[k] + 1 in handcards:
            feature[15 + num] += 1
        k += 1
    # print feature
    return feature


# 68-88
def ac(handcards):
    """
    AC搭子
    @param handcards: 手牌
    @return: AC搭子列表
    """
    feature = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    k = 0
    while k <= len(handcards) - 2:
        num = handcards[k] & 0x0F
        # 万 ac【0-6】
        if handcards[k] & 0xF0 == 0x00 and handcards[k] + 2 in handcards:
            feature[num - 1] += 1
        # 条 【7-13】

        elif handcards[k] & 0xF0 == 0x10 and handcards[k] + 2 in handcards:
            # print handcards
            feature[6 + num] += 1
        # 筒 【14-20】
        elif handcards[k] & 0xF0 == 0x20 and handcards[k] + 2 in handcards:
            feature[13 + num] += 1
        k += 1
    # print  feature
    return feature


# 89-109
def abc(handcards):
    """
    顺子
    @param handcards: 手牌
    @return: 顺子特征矩阵
    """
    feature = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    k = 0
    # print feature
    while k <= len(handcards) - 3:
        num = handcards[k] & 0x0F
        # wan abc [0-6]
        if handcards[k] & 0xF0 == 0x00 and handcards[k] + 2 in handcards and handcards[k] + 1 in handcards:
            feature[num - 1] += 1
        # tiao abc [7-13]
        elif handcards[k] & 0xF0 == 0x10 and handcards[k] + 2 in handcards and handcards[k] + 1 in handcards:
            feature[6 + num] += 1
        # tong abc [14-20]
        elif handcards[k] & 0xF0 == 0x20 and handcards[k] + 2 in handcards and handcards[k] + 1 in handcards:
            feature[13 + num] += 1
        k += 1

    # print feature
    return feature


def a2_a3_a4(handcards, an):
    """
    an张相同牌的筛选
    @param handcards: 手牌
    @param an: 牌数
    @return: an张相同牌的特征矩阵
    """
    features = [0] * 34
    L = set(handcards)
    for card in L:
        if handcards.count(card) == an:
            index = tool.translate3(card)
            features[index] = 1
    return features


# 110
def kindOfKing(kingcard):
    """
    确定宝牌的编号
    @param kingcard: 宝牌
    @return: 宝牌编号
    """
    color = (kingcard & 0xF0) / 16
    num = kingcard & 0x0F
    result = 0
    if color == 0:
        result = num
    elif color == 1:
        result = 9 + num
    elif color == 2:
        result = 18 + num
    elif color == 3:
        result = 27 + num
    # print  result
    return result


def kindOfKing_list(kingcard):
    """
    宝牌种类
    @param kingcard: 宝牌
    @return: 宝牌类型
    """
    feature = [0, 0, 0, 0]
    color = int((kingcard & 0xF0) / 16)  # 类别
    feature[color] = 1
    return feature


def is_fei_king(fei):
    """
    是否飞宝
    @param fei: 飞宝数
    @return: 特征矩阵
    """
    feature = [0] * 2
    feature[fei - 1] = 1
    return feature


# 111

# 112-115
def fun_king(kingList):
    """
    四个玩家的飞宝数列表
    @param kingList: 飞宝列表
    @return: 飞宝列表
    """
    king = [0, 0, 0, 0]
    i = 0
    for kingcard in kingList:
        king[i] = kingcard
        i = i + 1
    # print king
    return king


# 116 飞宝数
# 117-129
def fun_19(handCards):
    """
    飞宝数-幺九牌
    @param handCards: 手牌
    @return: 飞宝数
    """
    num = numofcard(handCards)
    num_19 = [num[0], num[8], num[9], num[17], num[18]]
    num_19 = num_19 + num[26:]
    return num_19


def cal_13lan_funciton(tiles, result):
    """
    十三烂
    @param tiles: 牌墙
    @param result: 十三烂
    @return:
    """
    num = len(tiles)
    # print('num=%d' % num)
    # if num == 2:
    #     # pre precess
    #     if tiles[0] not in (1, 6) or tiles[1] - tiles[0] < 3:
    #         return result
    #     if tiles[0] == 1:
    #         result[tiles[1] - 4] = 1
    #     if tiles[0] == 2:
    #         result[tiles[1] + 1] = 1
    #     if tiles[0] == 3:
    #         result[tiles[1] + 5] = 1
    #     if tiles[0] == 4:
    #         result[tiles[1] + 8] = 1
    #     if tiles[0] == 5:
    #         result[tiles[1] + 10] = 1
    #     if tiles[0] == 6:
    #         result[tiles[1] + 11] = 1
    for i in range(num - 1):
        for j in range(i + 1, num):
            if tiles[i] not in range(1, 6) or tiles[j] - tiles[i] < 3:
                # print(i, j)
                continue
            if tiles[i] == 1:
                result[tiles[j] - 4] = 1
            if tiles[i] == 2:
                result[tiles[j] + 1] = 1
            if tiles[i] == 3:
                result[tiles[j] + 5] = 1
            if tiles[i] == 4:
                result[tiles[j] + 8] = 1
            if tiles[i] == 5:
                result[tiles[j] + 10] = 1
            if tiles[i] == 6:
                result[tiles[j] + 11] = 1

    if num >= 3:
        if (tiles.count(1) + tiles.count(4) + tiles.count(7)) == 3:
            result[0] = result[3] = result[15] = result[21] = 1

        if (tiles.count(1) + tiles.count(4) + tiles.count(8)) == 3:
            result[0] = result[4] = result[16] = result[22] = 1

        if (tiles.count(1) + tiles.count(4) + tiles.count(9)) == 3:
            result[0] = result[5] = result[17] = result[23] = 1

        if (tiles.count(1) + tiles.count(5) + tiles.count(8)) == 3:
            result[1] = result[4] = result[18] = result[24] = 1

        if (tiles.count(1) + tiles.count(5) + tiles.count(9)) == 3:
            result[1] = result[5] = result[19] = result[25] = 1

        if (tiles.count(1) + tiles.count(6) + tiles.count(9)) == 3:
            result[2] = result[5] = result[20] = result[26] = 1

        if (tiles.count(2) + tiles.count(5) + tiles.count(8)) == 3:
            result[6] = result[9] = result[18] = result[27] = 1

        if (tiles.count(2) + tiles.count(5) + tiles.count(9)) == 3:
            result[6] = result[10] = result[19] = result[28] = 1

        if (tiles.count(2) + tiles.count(6) + tiles.count(9)) == 3:
            result[7] = result[10] = result[20] = result[29] = 1

        if (tiles.count(3) + tiles.count(6) + tiles.count(9)) == 3:
            result[11] = result[14] = result[20] = result[30] = 1

    return result


# 130-222 十三烂 2张的可能 真值
def cal_13lan_2tiles(tile_list):
    """
    十三烂2张的可能真值
    @param tiles: 牌墙
    @param result: 十三烂
    @return:
    """
    tmp = tile_list
    result1 = [0] * 31
    result2 = [0] * 31
    result3 = [0] * 31
    L = set(tmp)  # 去除重复手牌
    L_num0 = []  # 万数牌
    L_num1 = []  # 条数牌
    L_num2 = []  # 筒数牌
    for i in L:
        if i & 0xf0 == 0x00:
            L_num0.append(i & 0x0f)
        if i & 0xf0 == 0x10:
            L_num1.append(i & 0x0f)
        if i & 0xf0 == 0x20:
            L_num2.append(i & 0x0f)

    # 万字 烂牌 14-19,25-29,36-39,47-49,58-59,69
    result1 = cal_13lan_funciton(L_num0, result1)
    # print(L_num0)
    # 条  烂牌 14-19,25-29,36-39,47-49,58-59,69
    result2 = cal_13lan_funciton(L_num1, result2)
    # print(L_num1)
    # 筒字 烂牌 14-19,25-29,36-39,47-49,58-59,69
    result3 = cal_13lan_funciton(L_num2, result3)
    # print(L_num2)
    return result1 + result2 + result3


# 副露数
def fuluNum(actions):
    """
    副露数
    @param actions: 副露
    @return: 副露数
    """
    return len(actions)


# 某个顺子个数
def shunziNum(actions):
    """
    某个顺子个数
    @param actions: 副露
    @return: 顺子个数
    """
    feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wan = []
    tiao = []
    tong = []

    # wan
    for i in range(len(actions)):
        if actions[i][0] != actions[i][1] and actions[i][0] < 8:
            wan.append(actions[i])

    for j in range(len(wan)):
        feature[wan[j][0] - 1] += 1

    # tiao
    for i in range(len(actions)):
        if actions[i][0] != actions[i][1] and 16 < actions[i][0] < 26:
            tiao.append(actions[i])

    for j in range(len(tiao)):
        feature[tiao[j][0] - 10] += 1

    # tong
    for i in range(len(actions)):
        if actions[i][0] != actions[i][1] and actions[i][0] > 32:
            tong.append(actions[i])

    for j in range(len(tong)):
        feature[tong[j][0] - 19] += 1

    return feature


# 刻子判断
def keziJudge(actions):
    """
    刻子判断
    @param actions: 副露
    @return: 刻字特征矩阵
    """
    feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wan = []
    tiao = []
    tong = []
    zi = []

    # wan
    for i in range(len(actions)):
        if len(actions[i]) == 3:
            if actions[i][0] == actions[i][1] and actions[i][0] < 10:
                wan.append(actions[i])

    for j in range(len(wan)):
        feature[wan[j][0] - 1] += 1

    # tiao
    for i in range(len(actions)):
        if len(actions[i]) == 3:
            if actions[i][0] == actions[i][1] and 16 < actions[i][0] < 26:
                tiao.append(actions[i])

    for j in range(len(tiao)):
        feature[tiao[j][0] - 8] += 1

    # tong
    for i in range(len(actions)):
        if len(actions[i]) == 3:
            if actions[i][0] == actions[i][1] and 32 < actions[i][0] < 49:
                tong.append(actions[i])

    for j in range(len(tong)):
        feature[tong[j][0] - 15] += 1

    # zi
    for i in range(len(actions)):
        if len(actions[i]) == 3:
            if actions[i][0] == actions[i][1] and actions[i][0] > 48:
                zi.append(actions[i])

    for j in range(len(zi)):
        feature[zi[j][0] - 22] += 1

    return feature


# 杠判断
def gangJudge(actions):
    """
    杠子判断
    @param actions: 副露
    @return: 杠子特征矩阵
    """
    feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wan = []
    tiao = []
    tong = []
    zi = []

    # wan
    for i in range(len(actions)):
        if len(actions[i]) == 4:
            if actions[i][0] == actions[i][1] and actions[i][0] < 10:
                wan.append(actions[i])

    for j in range(len(wan)):
        feature[wan[j][0] - 1] += 1

    # tiao
    for i in range(len(actions)):
        if len(actions[i]) == 4:
            if actions[i][0] == actions[i][1] and 16 < actions[i][0] < 26:
                tiao.append(actions[i])

    for j in range(len(tiao)):
        feature[tiao[j][0] - 8] += 1

    # tong
    for i in range(len(actions)):
        if len(actions[i]) == 4:
            if actions[i][0] == actions[i][1] and 32 < actions[i][0] < 49:
                tong.append(actions[i])

    for j in range(len(tong)):
        feature[tong[j][0] - 15] += 1

    # zi
    for i in range(len(actions)):
        if len(actions[i]) == 4:
            if actions[i][0] == actions[i][1] and actions[i][0] > 48:
                zi.append(actions[i])

    for j in range(len(zi)):
        feature[zi[j][0] - 22] += 1

    return feature


def get_single_cards_code(hand_cards, jing_card, hu_type):
    '''
    返回对应胡牌类型的单张
    :param hand_cards: 手牌 16进制
    :param jing_card:   jing_card 16进制
    :param hu_type:  胡牌类型[0,1,2,3] -> 平胡、七对、九幺、十三烂
    :return:特征
    '''
    if hu_type == 0:  # pinghu
        single_cards = tool.get_comm_single_card(hand_cards)  # 考虑是否需要把精牌从手牌中剔除

    elif hu_type == 1:  # qidui
        single_cards = tool.get_qidui_single_card(hand_cards)

    elif hu_type == 2:  # 九幺
        single_cards = tool.get_91_single_card(hand_cards)
    elif hu_type == 3:
        single_cards = tool.get_13_single_card(hand_cards)
    else:
        single_cards = [0] * 14
        raise IndexError

    # 精牌不算孤张
    if jing_card in single_cards:
        single_cards.remove(jing_card)
    cards = numofcard(single_cards)
    feature = tool.batch_one_hot(5, cards)
    return feature


def get_hu_type_code(hand_cards, jing_card, com_xt, type_7xt, type_91xt, type_13xt):  # 对胡牌方向及该方向剩余牌编码
    """
    # 对胡牌方向及该方向剩余牌编码
    @param hand_cards: 手牌
    @param jing_card: 宝牌
    @param com_xt: 平胡向听数
    @param type_7xt: 七对向听数
    @param type_91xt: 九幺向听数
    @param type_13xt: 十三烂向听数
    @return: 胡牌方向、剩余牌编码
    """
    xt_list = [com_xt, type_7xt + 3, type_91xt + 1, type_13xt + 1]
    min_xt = min(xt_list)
    hu_direction_index = xt_list.index(min_xt)  # 胡牌方向
    hu_code = [0] * 4
    hu_code[hu_direction_index] = 1

    # single_cards_code = None  # 1*170
    single_cards_code = get_single_cards_code(hand_cards, jing_card, hu_direction_index)

    return hu_code, single_cards_code


def cards16_to_index34(handcards):
    """
    麻将牌格式转换：十六进制->十进制0-34
    @param handcards: 手牌
    @return: 麻将牌十进制0-34特征矩阵
    """
    features = [0] * 34
    for card in handcards:
        features[tool.translate3(card)] += 1
    return features


# 门清no_king
def calculate1(handCards):
    """
    门清no_king
    @param handCards: 手牌
    @return: 无宝牌的门前编码
    """
    feature = []

    # 1 通常手
    hand_cards = tool.deepcopy(handCards)
    f1 = wait_types_comm(hand_cards)
    feature.append(14 - f1)

    # 2 七对
    hand_cards = tool.deepcopy((handCards))
    f2 = wait_types_7(hand_cards)
    feature.append(14 - f2)

    # 3 十三浪
    hand_cards = tool.deepcopy((handCards))
    f3 = wait_types_13(hand_cards)
    feature.append(14 - f3)

    # 4 91
    hand_cards = tool.deepcopy((handCards))
    f4 = wait_types_19(hand_cards)
    feature.append(14 - f4)

    # 5-9 花色数
    hand_cards = tool.deepcopy((handCards))
    f5_9 = numOfCards(hand_cards)
    feature = feature + f5_9

    # 10-43 每张牌的数
    hand_cards = tool.deepcopy((handCards))
    f10_43 = numofcard(hand_cards)
    feature = feature + f10_43

    # 44-67 ab
    hand_cards = tool.deepcopy((handCards))
    f44_67 = ab(hand_cards)
    feature = feature + f44_67

    # 68_88 ac
    hand_cards = tool.deepcopy((handCards))
    f68_88 = ac(hand_cards)
    feature = feature + f68_88

    # 89-109 abc
    hand_cards = tool.deepcopy((handCards))
    f89_109 = abc(hand_cards)
    feature = feature + f89_109

    # 114-206 十三浪牌数
    hand_cards = tool.deepcopy((handCards))
    f114_206 = cal_13lan_2tiles(hand_cards)
    feature = feature + f114_206

    return feature


# 门清king
def calculate2(handCards, king_card, king_num, fei_king, fei):
    """
    门清（带宝牌）
    @param handCards: 手牌
    @param king_card: 宝牌
    @param king_num: 宝牌数
    @param fei_king: 飞宝数
    @param fei: 本手是否飞宝
    @return: 门清特征矩阵
    """
    feature = []

    # 1 通常手
    hand_cards = tool.deepcopy(handCards)
    f1 = wait_types_comm(hand_cards)
    feature.append(14 - f1)

    # 2 七对
    hand_cards = tool.deepcopy((handCards))
    f2 = wait_types_7(hand_cards)
    feature.append(14 - f2)

    # 3 十三浪
    hand_cards = tool.deepcopy((handCards))
    f3 = wait_types_13(hand_cards)
    feature.append(14 - f3)

    # 4 91
    hand_cards = tool.deepcopy((handCards))
    f4 = wait_types_19(hand_cards)
    feature.append(14 - f4)

    # 5-9 花色数
    hand_cards = tool.deepcopy((handCards))
    f5_9 = numOfCards(hand_cards)
    feature = feature + f5_9

    # 10-43 每张牌的数
    hand_cards = tool.deepcopy((handCards))
    f10_43 = numofcard(hand_cards)
    feature = feature + f10_43

    # 44-67 ab
    hand_cards = tool.deepcopy((handCards))
    f44_67 = ab(hand_cards)
    feature = feature + f44_67

    # 68_88 ac
    hand_cards = tool.deepcopy((handCards))
    f68_88 = ac(hand_cards)
    feature = feature + f68_88

    # 89-109 abc
    hand_cards = tool.deepcopy((handCards))
    f89_109 = abc(hand_cards)
    feature = feature + f89_109

    # 110 kingcard
    f110 = kindOfKing(king_card)
    feature.append(f110)

    # 111 num_kingcards
    feature.append(king_num)

    # 111 飞宝数
    feature.append(fei_king)

    # 112 本手是否飞宝
    feature.append(fei)

    # 114-206 十三浪牌数
    hand_cards = tool.deepcopy((handCards))
    f114_206 = cal_13lan_2tiles(hand_cards)
    feature = feature + f114_206

    return feature


# 吃碰杠决策 non_king_op
def calculate_noking_op(handCards, actions, opcard):
    """
    吃碰杠决策（不带宝牌）
    @param handCards: 手牌
    @param actions: 副露
    @param opcard: 操作牌
    @return: 动作矩阵
    """
    feature = []

    # 1 通常手
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f1 = wait_types_comm(hand_cards, t_actions)
    feature.append(14 - f1)

    # 2 七对
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f2 = wait_types_7(hand_cards, t_actions)
    feature.append(14 - f2)

    # 3 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f4 = wait_types_19(hand_cards, t_actions)
    feature.append(14 - f4)

    # 4 十三浪
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f3 = wait_types_13(hand_cards, t_actions)
    feature.append(14 - f3)

    # 5 操作牌
    f5 = translate(opcard)
    feature.append(f5)

    # 6-7_左吃
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f6_7 = waitnums_chi_left(hand_cards, t_actions, opcard)
    feature = feature + f6_7

    # 8-9 右吃
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f8_9 = waitnums_chi_right(hand_cards, t_actions, opcard)
    feature = feature + f8_9

    # 10-11 中吃
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f10_11 = waitnums_chi_mid(hand_cards, t_actions, opcard)
    feature = feature + f10_11

    # 12-13 碰杠
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f12_13 = waitnums_dec_peng_gang(hand_cards, t_actions, opcard)
    feature = feature + f12_13

    # 14-15 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f14_15 = waitnums_19_change(hand_cards, t_actions, opcard)
    feature = feature + f14_15

    # 16-20 万，条，筒，风，箭个数
    hand_cards = tool.deepcopy(handCards)
    f16_20 = numOfCards(hand_cards)
    feature = feature + f16_20

    # 21-54
    hand_cards = tool.deepcopy(handCards)
    f21_54 = numofcard(hand_cards)
    feature = feature + f21_54

    # 55-78 ab
    hand_cards = tool.deepcopy(handCards)
    f55_78 = ab(hand_cards)
    feature = feature + f55_78

    # 79-99 ac
    hand_cards = tool.deepcopy(handCards)
    f79_99 = ac(hand_cards)
    feature = feature + f79_99

    # 100-120 abc
    hand_cards = tool.deepcopy(handCards)
    f100_120 = abc(hand_cards)
    feature = feature + f100_120

    # 121 副露数
    f121 = fuluNum(actions)
    feature.append(f121)

    # 122-142 顺子数
    f122_142 = shunziNum(actions)
    feature = feature + f122_142

    # 143-176 刻子
    f143_176 = keziJudge(actions)
    feature = feature + f143_176

    # 177-210 杠
    f177_210 = gangJudge(actions)
    feature = feature + f177_210

    return feature


# 吃碰杠决策 king
def calculate_king_op(handCards, actions, opcard, king_card, fei_king, king_num):
    """
    吃碰杠决策（带宝牌）
    @param handCards: 手牌
    @param actions: 副露
    @param opcard: 操作牌
    @param king_card: 宝牌
    @param fei_king: 飞宝数
    @param king_num: 宝牌数
    @return: 动作矩阵
    """
    feature = []

    # 1 通常手
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f1 = wait_types_comm(hand_cards, t_actions)
    feature.append(14 - f1)

    # 2 七对
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f2 = wait_types_7(hand_cards, t_actions)
    feature.append(14 - f2)

    # 3 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f4 = wait_types_19(hand_cards, t_actions)
    feature.append(14 - f4)

    # 4 十三浪
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f3 = wait_types_13(hand_cards, t_actions)
    feature.append(14 - f3)

    # 5 操作牌
    f5 = translate(opcard)
    feature.append(f5)

    # 6-7_左吃
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f6_7 = waitnums_chi_left(hand_cards, t_actions, opcard)
    feature = feature + f6_7

    # 8-9 右吃
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f8_9 = waitnums_chi_right(hand_cards, t_actions, opcard)
    feature = feature + f8_9

    # 10-11 中吃
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f10_11 = waitnums_chi_mid(hand_cards, t_actions, opcard)
    feature = feature + f10_11

    # 12-13 碰杠
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f12_13 = waitnums_dec_peng_gang(hand_cards, t_actions, opcard)
    feature = feature + f12_13

    # 14-15 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f14_15 = waitnums_19_change(hand_cards, t_actions, opcard)
    feature = feature + f14_15

    # 16-20 万，条，筒，风，箭个数
    hand_cards = tool.deepcopy(handCards)
    f16_20 = numOfCards(hand_cards)
    feature = feature + f16_20

    # 21-54
    hand_cards = tool.deepcopy(handCards)
    f21_54 = numofcard(hand_cards)
    feature = feature + f21_54

    # 55-78 ab
    hand_cards = tool.deepcopy(handCards)
    f55_78 = ab(hand_cards)
    feature = feature + f55_78

    # 79-99 ac
    hand_cards = tool.deepcopy(handCards)
    f79_99 = ac(hand_cards)
    feature = feature + f79_99

    # 100-120 abc
    hand_cards = tool.deepcopy(handCards)
    f100_120 = abc(hand_cards)
    feature = feature + f100_120

    # 121 副露数
    f121 = fuluNum(actions)
    feature.append(f121)

    # 122-142 顺子数
    f122_142 = shunziNum(actions)
    feature = feature + f122_142

    # 143 宝牌种类
    f143 = kindOfKing(king_card)
    feature.append(f143)

    # 144 宝牌数量
    feature.append(king_num)

    # 145 飞了几个宝
    feature.append(fei_king)

    # 146-179 刻子
    f146_179 = keziJudge(actions)
    feature = feature + f146_179

    # 180-213 杠
    f180_213 = gangJudge(actions)
    feature = feature + f180_213

    return feature


# 无宝牌综合特征
def calculate_noking_sys(handCards, actions):
    """
    无宝牌综合特征
    @param handCards: 手牌
    @param actions: 副露
    @return: 无宝牌综合特征
    """
    feature = []

    # 1 通常手
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f1 = wait_types_comm(hand_cards, t_actions)
    feature.append(14 - f1)

    # 2 七对
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f2 = wait_types_7(hand_cards, t_actions)
    feature.append(14 - f2)

    # 3 十三浪
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f3 = wait_types_13(hand_cards, t_actions)
    feature.append(14 - f3)

    # 4 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f4 = wait_types_19(hand_cards, t_actions)
    feature.append(14 - f4)

    # 5-9 万，条，筒，风，箭个数
    hand_cards = tool.deepcopy(handCards)  # 14,14,14,14,12
    f5_9 = numOfCards(hand_cards)
    feature = feature + f5_9

    # 10-43
    hand_cards = tool.deepcopy(handCards)
    f10_43 = numofcard(hand_cards)
    feature = feature + f10_43

    # 44-67 ab
    hand_cards = tool.deepcopy(handCards)
    f44_67 = ab(hand_cards)
    feature = feature + f44_67

    # 68-88 ac
    hand_cards = tool.deepcopy(handCards)
    f68_88 = ac(hand_cards)
    feature = feature + f68_88

    # 89-109 abc
    hand_cards = tool.deepcopy(handCards)
    f89_109 = abc(hand_cards)
    feature = feature + f89_109

    # 110 副露数
    f110 = fuluNum(actions)
    feature.append(f110)

    # 111-131 顺子数
    f111_131 = shunziNum(actions)
    feature = feature + f111_131

    # 132-165 刻子
    f132_165 = keziJudge(actions)
    feature = feature + f132_165

    # 166-199 杠
    f166_199 = gangJudge(actions)
    feature = feature + f166_199

    # 200-292 烂牌
    hand_cards = tool.deepcopy(handCards)
    f200_292 = cal_13lan_2tiles(hand_cards)
    feature = feature + f200_292

    return feature


# 有宝牌综合特征
def calculate_king_sys(handCards, actions, king_card, fei_king, king_num, fei):
    """
    有宝牌综合特征
    @param handCards: 手牌
    @param actions: 副露
    @param king_card: 宝牌
    @param fei_king: 飞宝数
    @param king_num: 宝牌数
    @param fei: 本手是否飞宝
    @return: 有宝牌综合特征
    """
    feature = []

    # 1 通常手
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f1 = wait_types_comm_king(hand_cards, t_actions, king_card)
    feature.append(14 - f1)

    # 2 七对
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f2 = wait_types_7(hand_cards, t_actions, king_card)
    feature.append(14 - f2)

    # 3 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f4 = wait_types_19(hand_cards, t_actions)
    feature.append(14 - f4)

    # 4 十三浪
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f3 = wait_types_13(hand_cards, t_actions)
    feature.append(14 - f3)

    # 5-9 万，条，筒，风，箭个数
    hand_cards = tool.deepcopy(handCards)
    f5_9 = numOfCards(hand_cards)
    feature = feature + f5_9

    # 10-43
    hand_cards = tool.deepcopy(handCards)
    f10_43 = numofcard(hand_cards)
    feature = feature + f10_43

    # 44-67 ab
    hand_cards = tool.deepcopy(handCards)
    f44_67 = ab(hand_cards)
    feature = feature + f44_67

    # 68-88 ac
    hand_cards = tool.deepcopy(handCards)
    f68_88 = ac(hand_cards)
    feature = feature + f68_88

    # 89-109 abc
    hand_cards = tool.deepcopy(handCards)
    f89_109 = abc(hand_cards)
    feature = feature + f89_109

    # 110 副露数
    f110 = fuluNum(actions)
    feature.append(f110)

    # 111-131 顺子数
    f111_131 = shunziNum(actions)
    feature = feature + f111_131

    # 132 宝牌种类
    f132 = kindOfKing(king_card)
    feature.append(f132)

    # 133 宝牌数量
    feature.append(king_num)

    # 134 飞了几个宝
    feature.append(fei_king)

    # 135 本手是否飞宝
    feature.append(fei)

    # 136-169 刻子
    f133_168 = keziJudge(actions)
    feature = feature + f133_168

    # 170-203 杠
    f169_202 = gangJudge(actions)
    feature = feature + f169_202

    # 204-296 烂牌
    hand_cards = tool.deepcopy(handCards)
    f203_295 = cal_13lan_2tiles(hand_cards)
    feature = feature + f203_295

    return feature


# 2020.6.3 dqn中提取特征，取消顺子  shape 542
def calculate_noking_sys_2_dqn(handCards, actions):
    """
    dqn中提取特征，取消顺子  shape 542
    @param handCards: 手牌
    @param actions: 副露
    @return: DQN中提取特征
    """
    feature = []

    # 1 通常手
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f1 = wait_types_comm(hand_cards, t_actions)
    # feature.append(14 - f1)

    # 2 七对
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f2 = wait_types_7(hand_cards, t_actions)
    # feature.append(14 - f2)

    # 3 十三浪
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f3 = wait_types_13(hand_cards, t_actions)
    # feature.append(14 - f3)

    # 4 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f4 = wait_types_19(hand_cards, t_actions)
    # feature.append(14 - f4)

    # 5-9 万，条，筒，风，箭个数
    hand_cards = tool.deepcopy(handCards)  # 14,14,14,12
    f5_9 = numOfCards(hand_cards)
    feature = feature + f5_9
    feature = tool.batch_one_hot(14, feature)  #############为前面的特征进行onehot编码
    # 10-43
    hand_cards = tool.deepcopy(handCards)
    f10_43 = numofcard(hand_cards)
    f10_43 = tool.translate34_to_136(f10_43)
    feature = feature + f10_43

    # 44-67 ab
    feature2 = []
    hand_cards = tool.deepcopy(handCards)  # 4
    f44_67 = ab(hand_cards)
    feature2 = feature2 + f44_67

    # 68-88 ac
    hand_cards = tool.deepcopy(handCards)  # 4
    f68_88 = ac(hand_cards)
    feature2 = feature2 + f68_88

    # 89-109 abc
    hand_cards = tool.deepcopy(handCards)  # 4
    f89_109 = abc(hand_cards)
    feature2 = feature2 + f89_109

    # 110 副露数
    f110 = fuluNum(actions)  # 4
    feature2.append(f110)
    feature2 = tool.batch_one_hot(4, feature2)  # jjggghhhhhgg#jjthhetbtwbyebyenyef
    feature = feature + feature2  # ffghjhdhhgnhrnyrbfwbfwnmgbfnryhwrhethtjt

    # 111-131 顺子数
    # f111_131 = shunziNum(actions)
    # feature = feature + f111_131

    # 132-165 刻子
    f132_165 = keziJudge(actions)
    feature = feature + f132_165

    # 166-199 杠
    f166_199 = gangJudge(actions)
    feature = feature + f166_199

    # 200-292 烂牌
    hand_cards = tool.deepcopy(handCards)
    f200_292 = cal_13lan_2tiles(hand_cards)
    # feature = feature + f200_292

    return feature


# 2019.1.3
def calculate_noking_sys_2(handCards, actions):
    """
    特征编码，取消顺子  shape 542
    @param handCards: 手牌
    @param actions: 副露
    @return: 特征编码
    """
    feature = []

    # 1 通常手
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f1 = wait_types_comm(hand_cards, t_actions)
    # feature.append(14 - f1)

    # 2 七对
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f2 = wait_types_7(hand_cards, t_actions)
    # feature.append(14 - f2)

    # 3 十三浪
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f3 = wait_types_13(hand_cards, t_actions)
    # feature.append(14 - f3)

    # 4 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f4 = wait_types_19(hand_cards, t_actions)
    # feature.append(14 - f4)

    # 5-9 万，条，筒，风，箭个数
    hand_cards = tool.deepcopy(handCards)  # 14,14,14,12
    f5_9 = numOfCards(hand_cards)
    feature = feature + f5_9
    feature = tool.batch_one_hot(14, feature)  #############为前面的特征进行onehot编码  # 14 * 5个特征
    # 10-43
    hand_cards = tool.deepcopy(handCards)
    f10_43 = numofcard(hand_cards)
    f10_43 = tool.translate34_to_136(f10_43)
    feature = feature + f10_43

    # 44-67 ab
    feature2 = []
    hand_cards = tool.deepcopy(handCards)  # 4
    f44_67 = ab(hand_cards)
    feature2 = feature2 + f44_67

    # 68-88 ac
    hand_cards = tool.deepcopy(handCards)  # 4
    f68_88 = ac(hand_cards)
    feature2 = feature2 + f68_88

    # 89-109 abc
    hand_cards = tool.deepcopy(handCards)  # 4
    f89_109 = abc(hand_cards)
    feature2 = feature2 + f89_109

    # 110 副露数
    f110 = fuluNum(actions)  # 4
    feature2.append(f110)
    feature2 = tool.batch_one_hot(4, feature2)  # jjggghhhhhgg#jjthhetbtwbyebyenyef
    feature = feature + feature2  # ffghjhdhhgnhrnyrbfwbfwnmgbfnryhwrhethtjt

    # 111-131 顺子数
    f111_131 = shunziNum(actions)
    feature = feature + f111_131

    # 132-165 刻子
    f132_165 = keziJudge(actions)
    feature = feature + f132_165

    # 166-199 杠
    f166_199 = gangJudge(actions)
    feature = feature + f166_199

    # 200-292 烂牌
    hand_cards = tool.deepcopy(handCards)
    f200_292 = cal_13lan_2tiles(hand_cards)
    # feature = feature + f200_292
    return feature


# 有宝牌综合特征
# 2020 8 17
def calculate_king_sys_2(handCards, all_discards, actions, king_card, fei_king, king_num, fei, round_):
    """
    有宝牌综合特征
    @param handCards: 手牌
    @param all_discards: 所有玩家的弃牌
    @param actions: 副露
    @param king_card: 宝牌
    @param fei_king: 飞宝数
    @param king_num: 宝牌数
    @param fei: 本手是否飞宝
    @param round_: 回合数
    @return: 有宝牌综合特征
    """
    feature = []

    # 1 通常手  向听数
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f1 = wait_types_comm_king(hand_cards, t_actions, king_card)
    feature.append(f1)

    # 2 七对
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f2 = wait_types_7(hand_cards, t_actions, king_card)
    feature.append(f2)

    # 3 91
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f3 = wait_types_19(hand_cards, t_actions)
    feature.append(f3)

    # 4 十三浪
    hand_cards = tool.deepcopy(handCards)
    t_actions = tool.deepcopy(actions)
    f4 = wait_types_13(hand_cards, t_actions)
    feature.append(f4)

    # 5-9 万，条，筒，风，箭个数
    hand_cards = tool.deepcopy(handCards)
    f5_9 = numOfCards(hand_cards)
    feature = feature + f5_9
    # 0-134
    feature135 = tool.batch_one_hot(15, feature)  # 此处特征有 9 × 15 = 135
    feature = feature135
    # 手牌信息
    # 135-305
    # 添加手牌
    hand_cards = tool.deepcopy(handCards)
    num_cards = numofcard(hand_cards)
    feature136_305 = tool.batch_one_hot(5, num_cards)  # 包含数量特征的  1000-》1个  0100-》2个 。。。类推

    feature = feature + feature136_305  # 305

    # f_handcards_34 = cards16_to_index34(hand_cards)
    # 手牌顺子等信息
    # 0-23 ab
    feature2 = []
    hand_cards = tool.deepcopy(handCards)
    f_ab_24 = ab(hand_cards)
    feature2 = feature2 + f_ab_24  # 24

    # 24-45 ac
    hand_cards = tool.deepcopy(handCards)
    f_ac_21 = ac(hand_cards)
    feature2 = feature2 + f_ac_21  # 24 + 21 = 45

    # 46-67 abc
    hand_cards = tool.deepcopy(handCards)
    f_abc_21 = abc(hand_cards)
    feature2 = feature2 + f_abc_21  # 45 + 21 = 66

    feature2 = tool.batch_one_hot(5, feature2)  # jjggghhhhhgg#jjthhetbtwbyebyenyef 66 * 5 = 330
    feature = feature + feature2  # 305+330=635

    # 手牌aa， aaa， aaaa信息
    feature = feature + a2_a3_a4(hand_cards, 2)  # aa * 34
    feature = feature + a2_a3_a4(hand_cards, 3)  # aaa * 34
    feature = feature + a2_a3_a4(hand_cards, 4)  # aaaa * 34
    # features_len = 737

    # 737-830 烂牌
    hand_cards = tool.deepcopy(handCards)
    f_13lan_93 = cal_13lan_2tiles(hand_cards)
    feature = feature + f_13lan_93  # 737 + 93 = 830

    feature2 = []
    # 副露数
    f_fn_1 = fuluNum(actions)
    feature2.append(f_fn_1)  # 1

    # 111-131 顺子数
    f_fs2_22 = shunziNum(actions)
    feature2 = feature2 + f_fs2_22  # 21

    # 136-169 刻子
    f_f23_57 = keziJudge(actions)
    feature2 = feature2 + f_f23_57  # 34

    # 170-203 杠
    f_f58_92 = gangJudge(actions)
    feature2 = feature2 + f_f58_92  # 34

    # 副露信息

    # 133 宝牌数量
    feature2.append(king_num)  # 1

    # 134 飞了几个宝
    feature2.append(fei_king)  # 1

    feature2 = tool.batch_one_hot(5, feature2)  # 92* 5 = 460
    feature = feature + feature2  # 副露及部分宝牌信息 830 + 460 = 1290

    # 135 本手是否飞宝
    f_king = is_fei_king(fei)

    feature = feature + f_king  # 1290 + 2 = 1292

    # 132 宝牌种类
    f132 = kindOfKing_list(king_card)
    feature += f132  # 1091 + 4 = 1095

    # 添加所有弃牌 263 -398
    fall_discards = numofcard(all_discards)
    fall_discards = tool.batch_one_hot(5, fall_discards)
    feature = feature + fall_discards  # 1292 + 136 = 1466

    feature_round = tool.batch_one_hot(69, [round_])

    feature = feature + feature_round  # 1466 + 68 = 1535

    # 添加胡牌方向
    hand_cards = tool.deepcopy(handCards)
    feature_hu, feature_single_cards = get_hu_type_code(hand_cards, king_card, f1, f2, f3,
                                                        f4)  # feature_hu 1*4, feature_single: 1*170
    feature = feature + feature_hu + feature_single_cards  # 4 + 170 = 1709
    return feature


# 模仿suphx对牌进行编码
def suphx_cards_feature_code(cards_, channels):
    '''
    对牌集进行特征编码
    :param cards_:  牌或者牌集
    :param channels: 通道数
    :return:特征编码
    '''
    cards = copy.deepcopy(cards_)

    if not isinstance(cards, list):  # 如果是一张牌
        cards = [cards]

    features = []
    for channel in range(channels):
        S = set(cards)
        feature = [0] * 34
        for card in S:
            card_index = tool.translate3(card)
            cards.remove(card)
            feature[card_index] = 1
        features.append(feature)
    return features


def suphx_data_feature_code(data, channels=4, data_type="cards_set"):
    '''
    返回对数据按数据类型编码的特征
    :param data: 数据
    :param channel： 通道数
    :param type: 数据类型  optional ["cards_set", "seq_discards", "dummy"]
    :return:特征编码
    '''

    # cards 为16进制  此处的data可能不全是list
    data_copy = copy.deepcopy(data)
    features = []
    if data_type == "cards_set":
        features.extend(suphx_cards_feature_code(data_copy, channels))
    elif data_type == "seq_discards":
        seq_discards_features = []  # 弃牌的features,四个玩家的弃牌顺序，
        seq_len = 30  # 每个玩家弃牌的最大手数为30手
        for i in range(4):
            player_discard_seq = data_copy[i]
            cur_seq_discards_features = []  # 当前玩家的弃牌序列
            for i in range(len(player_discard_seq)):
                cur_seq_discards_features.extend(suphx_cards_feature_code(player_discard_seq[i], channels))

            seq_discards_features.extend(cur_seq_discards_features)  # 把当前已有的序列添加到features中
            need_pad_len = seq_len - len(player_discard_seq)  # 需要填充的长度

            # pad_features = [[0]*34 for _ in range(need_pad_len)]
            pad_features = [[0] * 34 for _ in range(need_pad_len * 4)]
            seq_discards_features.extend(pad_features)
        features.extend(seq_discards_features)
    elif data_type == "dummy":  # 哑变量编码  此时的data为整数
        assert isinstance(data_copy, int)
        dummy_features = [[0] * 34 for _ in range(channels)]
        if 0 < data_copy <= channels:
            # ---------------------------modified by xwc 2022.2.16----------#
            dummy_features[data_copy - 1] = [1] * 34
            # for i in range(data_copy):
            #     dummy_features[i] = [1] * 34
            # --------------------------------------------------------------#
        elif data_copy == 0:
            # pass  当为0时，哑变量全为零
            pass
        else:
            print("xts_fan_list", self_l)
            print("channers:", channels)
            print("data_copy:", data_copy)
            print("INFO[ERROR]")
        features.extend(dummy_features)
    elif data_type == "look_ahead":  # 暂时空着
        pass

    return features


def suphx_extract_feature_params(state, seat_id=0):
    '''
    处理特征抽取所需要的参数
    手牌，4个人的副露，宝牌，对手的手牌（第一个4×34全为零），牌墙中的牌，弃牌的顺序4×30×34,牌墙剩余牌， 拥有的宝牌数，所有玩家飞宝数，当前手数
    获取特征里的参数，模块重用
    需要把10进制转换成16进制
    :param state: 集成的状态信息
    :param seat_id: agent的座位id
    :return:特征抽取所需要的参数
    '''

    # 确保当前编码玩家的信息在第一个
    seat_seq = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]  # 座位顺序表 如果玩家座位为0 ， 则其他玩家顺序为seat_seq[seat_id]

    # # ---牌集特征 私有手牌、所有副露、宝牌-----
    handCards0 = tool.list10_to_16(state.players[seat_id].handcards)

    # 所有副露
    fulu_ = []
    for s_id in seat_seq[seat_id]:  # 按顺序排列
        cur_player_fulu = []  # 当前玩家副露表
        for i in state.player_fulu[s_id]:
            cur_player_fulu.append(tool.list10_to_16(i))
        fulu_.append(cur_player_fulu)

    # 宝牌
    king_card = tool.f10_to_16(state.jing_card)

    # # --- 隐藏特征-----
    # 所有玩家手牌
    all_player_handcards = []
    # 对手手中的宝牌数
    all_palyer_king_nums = []
    for s_id in seat_seq[seat_id]:  # 按顺序排列
        cur_player_handcards = tool.list10_to_16(state.players[s_id].handcards)  # 当前玩家的手牌
        all_palyer_king_nums.append(cur_player_handcards.count(king_card))
        all_player_handcards.append(cur_player_handcards)
    # 牌墙中的牌
    card_library = tool.list10_to_16_2(state.card_library)  # 无需sort

    # # --- 各玩家弃牌顺序，及飞宝个数-----
    discards_seq = []
    # 飞宝数
    fei_king_nums = []  # 四个玩家分别飞宝数

    for s_id in seat_seq[seat_id]:  # 按顺序排列
        cur_player_dis = tool.list10_to_16(state.player_discards[s_id])
        discards_seq.append(cur_player_dis)
        # 增加飞宝数
        fei_king_nums.append(cur_player_dis.count(king_card))

    # # --- 整数特征，包括·牌墙剩余数，拥有宝牌数，当前轮数-----
    round_ = state.round + 1
    # 牌墙剩余数
    remain_card_num = len(state.card_library)
    self_king_num = all_palyer_king_nums[0]

    # # --- 分类特征，包括·庄家标志-----
    dealer_flag_ = [0, 0, 0, 0]
    dealer_flag_[state.dealer_seat_id] = 1
    dealer_flag = []
    for s_id in seat_seq[seat_id]:  # 按顺序排列
        dealer_flag.append(dealer_flag_[s_id])

    return handCards0, fulu_, king_card, all_player_handcards, card_library, all_palyer_king_nums, discards_seq, \
           remain_card_num, self_king_num, fei_king_nums, round_, dealer_flag


# 模仿suphx对牌进行编码
def calculate_king_sys_suphx(state, seat_id=0, search=False, global_state=False, dropout_prob=0):
    '''
    牌都是用16进制进行表示，参数需要预先处理好
    返回不加前瞻特征及隐藏特征的特征
    :param state: 集成的状态信息
    :param seat_id: agent的座位id
    :param search: 开启前瞻搜索特征
    :param global_state: 是否编码隐藏信息特征
    :param dropout_prob: 对隐藏信息特征的dropout的概率
    :return:不加前瞻特征及隐藏特征的特征
    '''

    handCards0, fulu_, king_card, all_player_handcards, card_library, all_palyer_king_nums, discards_seq, \
    remain_card_num, self_king_num, fei_king_nums, round_, dealer_flag = \
        suphx_extract_feature_params(state, seat_id)

    # 所有特征
    features = []

    # 手牌特征
    handcards_features = suphx_data_feature_code(handCards0, 4)
    features.extend(handcards_features)

    # 副露特征
    fulu_features = []
    for fulu in fulu_:
        action_features = []
        fulu_len = len(fulu)  # 当前玩家副露的长度
        for action in fulu:
            action_features.extend(suphx_data_feature_code(action, 4))
        # 需要padding
        action_padding_features = [[0] * 34 for _ in range(4) for _ in range(4 - fulu_len)]
        action_features.extend(action_padding_features)

        fulu_features.extend(action_features)
    features.extend(fulu_features)

    # 宝牌特征
    # king_features = suphx_data_feature_code(king_card, 1)
    king_features = suphx_data_feature_code(king_card, 4)
    features.extend(king_features)

    # # 隐藏信息特征  20.12.9版本
    # hiding_info_features = []
    # if global_state and dropout_prob < 1:  # 开启隐藏特征
    #     # 对手手牌
    #     for player_handcards in all_player_handcards:
    #         hiding_info_features.extend(suphx_data_feature_code(player_handcards, 4))
    #
    #     # 牌墙中的牌
    #     hiding_info_features.extend(suphx_data_feature_code(card_library, 4))
    #
    #     # 对手手中的宝牌数
    #     for player_king_nums in all_palyer_king_nums:
    #         hiding_info_features.extend(suphx_data_feature_code(player_king_nums, 4, data_type="dummy"))
    #
    #     # 对隐藏信息特征进行dropout
    #     # 转换成np格式
    #     hiding_info_features = np.array(hiding_info_features, dtype=np.int)
    #     hiding_info_features_size = hiding_info_features.shape[0] * hiding_info_features.shape[1]
    #
    #     index_list = [index for index in range(hiding_info_features_size)]
    #     drop_indexs = random.sample(index_list, int(hiding_info_features_size * dropout_prob))
    #     drop_matrix = np.ones([hiding_info_features_size], dtype=np.int)
    #
    #     for dropout_index in drop_indexs:  drop_matrix[dropout_index] = 0
    #
    #     drop_matrix = drop_matrix.reshape([-1, 34])
    #     hiding_info_features = hiding_info_features * drop_matrix
    #
    #     # 转换成list格式
    #     hiding_info_features = hiding_info_features.tolist()
    # else:
    #     hiding_info_features = [[0] * 34 for _ in range(36)]
    #
    # features.extend(hiding_info_features)

    # 所有弃牌的顺序信息
    # seq_discards_features = suphx_data_feature_code(discards_seq, 1, data_type="seq_discards")
    seq_discards_features = suphx_data_feature_code(discards_seq, 4, data_type="seq_discards")
    features.extend(seq_discards_features)

    # 剩余牌数特征
    remian_cardsnums_features = suphx_data_feature_code(remain_card_num, 120, data_type="dummy")
    features.extend(remian_cardsnums_features)

    # 自己拥有的宝牌数
    self_king_num_features = suphx_data_feature_code(self_king_num, 4, data_type="dummy")
    features.extend(self_king_num_features)

    # 所有玩家飞宝数
    all_palyer_fei_king_num_features = []
    for fei_king_num in fei_king_nums:
        all_palyer_fei_king_num_features.extend(suphx_data_feature_code(fei_king_num, 4, data_type="dummy"))
    features.extend(all_palyer_fei_king_num_features)

    # 当前手数
    cur_round_features = suphx_data_feature_code(round_, 30, data_type="dummy")
    features.extend(cur_round_features)

    # 庄家特征
    dealer_features = []
    for flag in dealer_flag:
        dealer_features.extend(suphx_data_feature_code(flag, 1, data_type="dummy"))

    features.extend(dealer_features)

    # 开启搜索特征
    search_features = [[0] * 34 for _ in range(52)]
    if search:
        # paixing -> [平胡 九幺　七对 十三烂]
        # fanList -> [清一色、门清、碰碰胡、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
        # 判断remain_card_num是否为0 为0时搜索树会报错
        if remain_card_num <= 0: remain_card_num = 1

        recommend_card, paixing, fanList = SearchInfo.getSearchInfo(handCards0, fulu_[0], king_card, discards_seq,
                                                                    fulu_,
                                                                    fei_king_nums[0], remain_card_num, round_ - 1)

        search_features[0] = suphx_data_feature_code(recommend_card, 1)[0]

        # search_features[1 + paixing * 12] = [1] * 34
        # for fan_index in range(len(fanList)):
        #     if fanList[fan_index] == 1:
        #         search_features[paixing * 12 + 2 + fan_index] = [1] * 34

        search_features[4 + paixing * 12] = [1] * 34
        for fan_index in range(len(fanList)):
            if fanList[fan_index] == 1:
                search_features[paixing * 12 + 5 + fan_index] = [1] * 34
    features.extend(search_features)

    # 隐藏信息特征  21.2.23版本
    hiding_info_features = []
    if global_state and dropout_prob < 1:  # 开启隐藏特征
        # 对手手牌
        for player_handcards in all_player_handcards:
            hiding_info_features.extend(suphx_data_feature_code(player_handcards, 4))

        # 牌墙中的牌 长度 4×120 = 480
        for card in card_library:
            hiding_info_features.extend(suphx_data_feature_code(card, 4))

        # 为使长度一样，需要补零
        pad_len = 480 - len(card_library) * 4
        pad_features = [[0] * 34 for _ in range(pad_len)]
        hiding_info_features.extend(pad_features)
        # # 对手手中的宝牌数
        # for player_king_nums in all_palyer_king_nums:
        #     hiding_info_features.extend(suphx_data_feature_code(player_king_nums, 4, data_type="dummy"))

        # 对隐藏信息特征进行dropout
        # 转换成np格式
        hiding_info_features = np.array(hiding_info_features, dtype=np.int)
        hiding_info_features_size = hiding_info_features.shape[0] * hiding_info_features.shape[1]

        index_list = [index for index in range(hiding_info_features_size)]
        drop_indexs = random.sample(index_list, int(hiding_info_features_size * dropout_prob))
        drop_matrix = np.ones([hiding_info_features_size], dtype=np.int)

        for dropout_index in drop_indexs:  drop_matrix[dropout_index] = 0

        drop_matrix = drop_matrix.reshape([-1, 34])
        hiding_info_features = hiding_info_features * drop_matrix

        # 转换成list格式
        hiding_info_features = hiding_info_features.tolist()
    else:
        hiding_info_features = [[0] * 34 for _ in range(496)]

    features.extend(hiding_info_features)

    return features


def calculate_king_sys_suphx_1330(state, seat_id=0, search=False, global_state=False, dropout_prob=0):
    handCards0, fulu_, king_card, all_player_handcards, card_library, all_palyer_king_nums, discards_seq, \
    remain_card_num, self_king_num, fei_king_nums, round_, dealer_flag = \
        suphx_extract_feature_params(state, seat_id)
    # print("aaaa")
    # print(f"handCards0 = {handCards0}")
    # print(f"fulu_ = {fulu_}")
    # print(f"king_card = {king_card}")
    # print(f"all_player_handcards = {all_player_handcards}")
    # print(f"card_library = {card_library}")
    # print(f"all_palyer_king_nums = {all_palyer_king_nums}")
    # print(f"discards_seq = {discards_seq}")
    # print(f"remain_card_num = {remain_card_num}")
    # print(f"self_king_num = {self_king_num}")
    # print(f"fei_king_nums = {fei_king_nums}")
    # print(f"round_ = {round_}")
    # print(f"dealer_flag = {dealer_flag}")
    return calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                     discards_seq, remain_card_num, self_king_num, fei_king_nums, round_, dealer_flag,
                                     search,
                                     global_state, dropout_prob)


# 模仿suphx对牌进行编码(牌的编码是50多的哪个范围)
def calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                              discards_seq, remain_card_num, self_king_num, fei_king_nums, round_, dealer_flag,
                              search=True, global_state=False, dropout_prob=0):
    '''
    牌都是用16进制进行表示，参数需要预先处理好
    返回不加前瞻特征及隐藏特征的特征
    :param handCards0: 当前要编码玩家的手牌 -> [] 1维list
    :param fulu_: 四个玩家的副露 -> [[[7,8,9],[17,17,17]], [], [], []] 3维list   位置参考说明2
    :param king_card:  宝牌 -> 1 int
    :param all_player_handcards:四个玩家的手牌 -> [[],[],[],[]]  2维list   位置参考说明2  当后面三个玩家为空时，隐藏完美信息
    :param card_library:  牌库的牌 -> [] 1维list 当为空时，隐藏完美信息
    :param discards_seq:  四个玩家真实弃牌顺序-> [[], [], [], []] 2维list   位置参考说明2
    :param remain_card_num: 牌墙剩余牌 -> int
    :param self_king_num: 当前玩家的宝牌数 -> int
    :param fei_king_nums: 所有玩家的飞宝数 -> [0,0,0,0] 长度为4的一维list 位置参考说明2
    :param round_: 当前轮（手）数 -> int
    :param dealer_flag: 庄家flag，默认当前玩家为庄家 -> [1,0,0,0]
    :param search: 是否采用搜索树 默认关闭
    :param global_state:是否开启隐藏信息特征，默认关闭
    :param dropout_prob: 对隐藏信息的dropout率，默认为0
    '''

    # 所有特征
    features = []

    # 手牌特征
    handcards_features = suphx_data_feature_code(handCards0, 4)
    features.extend(handcards_features)

    # 副露特征
    fulu_features = []
    for i in range(4):
        fulu = fulu_[i]
        action_features = []
        fulu_len = len(fulu)  # 当前玩家副露的长度
        for action in fulu:
            action_features.extend(suphx_data_feature_code(action, 4))
        # 需要padding
        action_padding_features = [[0] * 34 for _ in range(4) for _ in range(4 - fulu_len)]
        action_features.extend(action_padding_features)

        fulu_features.extend(action_features)
    features.extend(fulu_features)

    # 宝牌特征
    # king_features = suphx_data_feature_code(king_card, 1)
    king_features = suphx_data_feature_code(king_card, 4)
    features.extend(king_features)

    # 所有弃牌的顺序信息
    # seq_discards_features = suphx_data_feature_code(discards_seq, 1, data_type="seq_discards")
    seq_discards_features = suphx_data_feature_code(discards_seq, 4, data_type="seq_discards")
    features.extend(seq_discards_features)

    # 剩余牌数特征
    remian_cardsnums_features = suphx_data_feature_code(remain_card_num, 120, data_type="dummy")
    features.extend(remian_cardsnums_features)

    # 自己拥有的宝牌数
    self_king_num_features = suphx_data_feature_code(self_king_num, 4, data_type="dummy")
    features.extend(self_king_num_features)

    # 所有玩家飞宝数
    all_palyer_fei_king_num_features = []
    for fei_king_num in fei_king_nums:
        all_palyer_fei_king_num_features.extend(suphx_data_feature_code(fei_king_num, 4, data_type="dummy"))
    features.extend(all_palyer_fei_king_num_features)

    # 当前手数
    cur_round_features = suphx_data_feature_code(round_, 30, data_type="dummy")
    features.extend(cur_round_features)

    # 庄家特征
    dealer_features = []
    for flag in dealer_flag:
        dealer_features.extend(suphx_data_feature_code(flag, 1, data_type="dummy"))
    features.extend(dealer_features)

    # 向听数特征[平胡，九幺，七对，十三烂] 增加维度56*34*1 add by xwc 2022.2.16
    xts_features = []
    for xts in get_xts_list(handCards0, fulu_[0], king_card):
        xts_features.extend(suphx_data_feature_code(xts, 14, data_type="dummy"))
    features.extend(xts_features)

    # # 各个番型的差牌数 [清一色、门清、碰碰胡、宝吊、宝还原、单吊、七星、飞宝1234] 增加维度 (7*14+4)102*34*1
    # # ------------------------------------add by xwc 2022.2.23--------------------------------------
    # fan_features = []
    # l = get_xts_fan_list(handCards0, fulu_[0], king_card, fei_king_nums[0])
    # for i in range(7):
    #     fan_features.extend(suphx_data_feature_code(l[i], 14, data_type="dummy"))
    # fan_features.extend(suphx_data_feature_code(l[-1], 4, data_type="dummy"))
    # features.extend(fan_features)
    # # ----------------------------------------------------------------------------------------------

    # 开启搜索特征
    search_features = [[0] * 34 for _ in range(52)]
    if search:
        # paixing -> [平胡 九幺　七对 十三烂]
        # fanList -> [清一色、门清、碰碰胡、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
        # 判断remain_card_num是否为0 为0时搜索树会报错
        if remain_card_num <= 0:
            remain_card_num = 1

        recommend_card, paixing, fanList = SearchInfo.getSearchInfo(handCards0, fulu_[0], king_card, discards_seq,
                                                                    fulu_,
                                                                    fei_king_nums[0],
                                                                    remain_card_num,
                                                                    round_ - 1)
        # print(f"recommend_card = {recommend_card}")
        # print(f"paixing = {paixing}")
        # print(f"fanList = {fanList}")
        search_features[0] = suphx_data_feature_code(recommend_card, 1)[0]

        # search_features[1 + paixing * 12] = [1] * 34
        # for fan_index in range(len(fanList)):
        #     if fanList[fan_index] == 1:
        #         search_features[paixing * 12 + 2 + fan_index] = [1] * 34

        search_features[4 + paixing * 12] = [1] * 34
        for fan_index in range(len(fanList)):
            if fanList[fan_index] == 1:
                search_features[paixing * 12 + 5 + fan_index] = [1] * 34
    # print(f"search_features = {search_features}")
    # print(f"search_features_s = {torch.tensor(search_features).sum()}")
    features.extend(search_features)

    # 隐藏信息特征  21.2.23版本
    hiding_info_features = []
    if global_state and dropout_prob < 1:  # 开启隐藏特征
        # 对手手牌
        for player_handcards in all_player_handcards:
            hiding_info_features.extend(suphx_data_feature_code(player_handcards, 4))

        # 牌墙中的牌 长度 4×120 = 480
        for card in card_library:
            hiding_info_features.extend(suphx_data_feature_code(card, 4))

        # 为使长度一样，需要补零
        pad_len = 480 - len(card_library) * 4
        pad_features = [[0] * 34 for _ in range(pad_len)]
        hiding_info_features.extend(pad_features)
        # # 对手手中的宝牌数
        # for player_king_nums in all_palyer_king_nums:
        #     hiding_info_features.extend(suphx_data_feature_code(player_king_nums, 4, data_type="dummy"))

        # 对隐藏信息特征进行dropout
        # 转换成np格式
        hiding_info_features = np.array(hiding_info_features, dtype=np.int)
        hiding_info_features_size = hiding_info_features.shape[0] * hiding_info_features.shape[1]

        index_list = [index for index in range(hiding_info_features_size)]
        drop_indexs = random.sample(index_list, int(hiding_info_features_size * dropout_prob))
        drop_matrix = np.ones([hiding_info_features_size], dtype=np.int)

        for dropout_index in drop_indexs:
            drop_matrix[dropout_index] = 0

        drop_matrix = drop_matrix.reshape([-1, 34])
        hiding_info_features = hiding_info_features * drop_matrix

        # 转换成list格式
        hiding_info_features = hiding_info_features.tolist()
    else:
        hiding_info_features = [[0] * 34 for _ in range(496)]
    features.extend(hiding_info_features)

    return features


# 1432*34*1(原始维度：1274；向听数：56；番型：102) add by xwc 2022.2.23
def card_preprocess_sr_suphx(handCards0, fulu_, king_card, all_player_handcards, card_library,
                             discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                             dealer_flag=[1, 0, 0, 0], search=True, global_state=False, dropout_prob=0):
    '''
    上饶麻将特征提取,模仿suphx
    说明:
    1.牌都是用16进制进行表示，参数需要预先处理好
    2.当前玩家的位置放在第一位  eg  当前玩家座位为0时:[0,1,2,3], 当前玩家座位为1时:[1,2,3,0], 当前玩家座位为2时:[2,3,0,1], ..[3,0,1,2]
    :param handCards0: 当前要编码玩家的手牌 -> [] 1维list
    :param fulu_: 四个玩家的副露 -> [[[7,8,9],[17,17,17]], [], [], []] 3维list   位置参考说明2
    :param king_card:  宝牌 -> 1 int
    :param all_player_handcards:四个玩家的手牌 -> [[],[],[],[]]  2维list   位置参考说明2  当后面三个玩家为空时，隐藏完美信息
    :param card_library:  牌库的牌 -> [] 1维list 当为空时，隐藏完美信息
    :param discards_seq:  四个玩家真实弃牌顺序-> [[], [], [], []] 2维list   位置参考说明2
    :param remain_card_num: 牌墙剩余牌 -> int
    :param self_king_num: 当前玩家的宝牌数 -> int
    :param fei_king_nums: 所有玩家的飞宝数 -> [0,0,0,0] 长度为4的一维list 位置参考说明2
    :param round_: 当前轮（手）数 -> int
    :param dealer_flag: 庄家flag，默认当前玩家为庄家 -> [1,0,0,0]
    :param search: 是否采用搜索树 默认开启
    :param global_state:是否开启隐藏信息特征，默认关闭
    :param dropout_prob: 对隐藏信息的dropout率，默认为0
    :return: 编码好的三维特征 1274×34×1
    '''
    # start = time.time()
    features = calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                         discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                                         dealer_flag, search, global_state, dropout_prob)
    # print("耗时：", time.time() - start)
    # start = time.time()
    features = np.array(features)
    # print("array:", time.time() - start)
    # start = time.time()
    features = features.T
    # print("T:", time.time() - start)
    # start = time.time()
    features = np.expand_dims(features, 0)
    # print("增加维度：", time.time() - start)
    # start = time.time()
    features = features.transpose([2, 1, 0])  # 更换位置  转换成c × 34 × 1的格式
    # print("更换位置：", time.time() - start)
    # features = np.zeros((1274, 34, 1))
    return features


def get_feature(json_data):
    """
    从json格式的数据中提取特征
    Args:
        json_data:

    Returns: (1330, 34, 1)的特征编码(tensor格式)

    """
    handCards0 = json_data["handcards"]
    fulu_ = [[], [], [], []]
    for idx in range(4):
        fulu_[idx] += json_data["fulu_chow"][idx]
        fulu_[idx] += json_data["fulu_peng"][idx]
        fulu_[idx] += json_data["fulu_kong"][idx]
    king_card = json_data["king_card"]
    all_player_handcards = [handCards0, [], [], []]
    card_library = []
    all_palyer_king_nums = [0, 0, 0, 0]
    discards_seq = json_data["discards_real"]
    remain_card_num = 0
    self_king_num = 0
    fei_king_nums = [0, 0, 0, 0]
    round_ = json_data["round"]
    featrues = calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                         discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                                         [1, 0, 0, 0])

    return torch.tensor(featrues, dtype=torch.float).view(1330, 34, 1)


def get_feature_global(json_data):
    """
    从json格式的数据中提取特征，为全局预测模型训练
    Args:
        json_data:

    Returns:(15, 1330*34*1)的特征编码(tensor格式)
    """
    king_card = json_data["king_card"]
    dealer_flag = json_data["dealer_flag"]
    featrues = []

    handCards0 = json_data["handcards"][0]
    fulu_ = json_data["discards_op"]
    all_player_handcards = json_data["handcards"]
    card_library = json_data["card_library"]
    discards_seq = json_data["discards_real"]
    remain_card_num = len(card_library)
    self_king_num = json_data["self_king_num"]
    fei_king_nums = json_data["fei_king_nums"]
    round_ = json_data["round"]
    feature_item = calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                             discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                                             dealer_flag, search=False, global_state=True, dropout_prob=1)
    featrues = torch.tensor(feature_item, dtype=torch.float)
    featrues = featrues.view(1330, 34, 1)
    # print(featrues.shape)
    # 30 * 1330, 34, 1
    # print(featrues.shape)
    return featrues

def get_feature_total(json_data):
    """
    从json格式的数据中提取特征，为全局预测模型训练
    Args:
        json_data:

    Returns:(15, 1330*34*1)的特征编码(tensor格式)
    """
    king_card = json_data["king_card"]
    dealer_flag = json_data["dealer_flag"]
    discard_step = json_data["discard_step"]
    featrues = []
    for item in discard_step:
        handCards0 = item["handcards"][0]
        fulu_ = item["discards_op"]
        all_player_handcards = item["handcards"]
        card_library = item["card_library"]
        discards_seq = item["discards_real"]
        remain_card_num = len(card_library)
        self_king_num = item["self_king_num"]
        fei_king_nums = item["fei_king_nums"]
        round_ = item["round"]
        feature_item = calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                                 discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                                                 dealer_flag, search=False, global_state=True, dropout_prob=1)
        featrues.append(feature_item)

    featrues = torch.tensor(featrues, dtype=torch.float)
    featrues = featrues.view(featrues.shape[0], 1330*34*1)
    arr = np.zeros((30 - len(discard_step), 1330*34*1))
    tmp = torch.tensor(arr, dtype=torch.float)
    featrues = torch.cat((featrues, tmp), dim=0)
    featrues = featrues.view(30 * 1330, 34, 1)
    # 30 * 1330, 34, 1
    # print(featrues.shape)
    return featrues, len(discard_step)


def get_feature_king(json_data):
    """
    从json格式的数据中提取特征
    Args:
        json_data:

    Returns: (1330, 34, 1)的特征编码(tensor格式)

    """
    handCards0 = json_data["handcards"]
    fulu_ = [[], [], [], []]
    for idx in range(4):
        fulu_[idx] += json_data["fulu_chow"][idx]
        fulu_[idx] += json_data["fulu_peng"][idx]
        fulu_[idx] += json_data["fulu_kong"][idx]
    king_card = json_data["king_card"]
    all_player_handcards = [handCards0, [], [], []]
    card_library = []
    all_palyer_king_nums = [0, 0, 0, 0]
    discards_seq = json_data["discards_real"]
    remain_card_num = 0
    self_king_num = 0
    fei_king_nums = [0, 0, 0, 0]
    round_ = json_data["round"]
    featrues = calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                         discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                                         [1, 0, 0, 0])
    return torch.tensor(featrues, dtype=torch.float)


if __name__ == '__main__':
    # test
    # handCards0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 19]
    # fulu_ = [[[18, 18, 18]], [], [[41, 41, 41], [20, 20, 20], [21, 22, 23]], []]
    # king_card = 5
    # all_player_handcards = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 19], [2, 2, 2, 6, 6, 6, 7, 8, 9, 49, 49, 50, 51, 52],
    #                         [33, 35, 38, 39, 40], []]
    # card_library = [53, 53, 2, 2, 9, 9, 9]
    # all_palyer_king_nums = [0, 0, 0, 0]
    # discards_seq = [[2, 3], [3, 2], [1, 4], [4, 1]]
    # remain_card_num = 83
    # self_king_num = 0
    # fei_king_nums = [1, 0, 0, 0]
    # round_ = 2
    # discards_real_list = [2, 3, 1, 4, 3, 2, 4, 1]
    # dealer_flag = [1, 0, 0, 0]
    #
    # featrues = card_preprocess_sr_suphx(handCards0, fulu_, king_card, all_player_handcards, card_library,
    #                                     discards_seq, remain_card_num, self_king_num, fei_king_nums, round_)
    # print(featrues.shape)
    # print(get_xts_fan_list(handCards0, fulu_[0], king_card, 1))

    data_root = "/home/tonnn/.nas/xy/output/globel/1"
    # json_path = os.path.join(data_root, "_4/0.json")
    df = open(os.path.join(data_root, "9.json"), encoding="utf-8")
    data = json.load(df)
    print(data)
    featrues = get_feature_global(data)
    print(featrues)
    print(featrues.shape)
