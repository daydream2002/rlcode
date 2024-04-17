#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : shangrao_hu.py
# @Description:上饶麻将胡牌判断
import copy
import mah_tool.feature_extract_v10 as feature
import mah_tool.tool2 as tool2

'''
上饶麻将胡牌牌型集合
'''

def qidui(handcards, suit=[], jing_card=0, catch_card=0):
    """
    判断是否胡了七对
    Args:
        handcards: 手牌
        suit: 副露
        jing_card: 宝牌
        catch_card: 摸到的牌

    Returns:bool值

    """
    qidui_xt = feature.wait_types_7(handcards, suit, jing_card)
    if qidui_xt == 0:
        if catch_card == jing_card or catch_card not in handcards:
            return False
        else:
            return True
    else:
        return False


def haohua_qidui(handcards, suit, jing_card, catch_card):
    """
    判断是否胡了豪华七对
    Args:
        handcards: 手牌
        suit: 副露
        jing_card: 宝牌
        catch_card: 摸到的牌

    Returns:bool值

    """
    _tile_list, si_card_num, san_card_num, er_card_num, jing_count = feature.get_four_three_two_card_jing_nums(
        handcards, jing_card)
    haohua_qidui_xt = feature.wait_types_haohua7(handcards, suit, jing_card)

    if haohua_qidui_xt == 0:
        if jing_count == 0:  # 无精情况，必胡
            return True
        else:
            # 判断是否需要精牌补足豪华七对条件
            if si_card_num > 0:
                need_xt_to_four = 0
            elif san_card_num > 0:
                need_xt_to_four = 1
            else:
                need_xt_to_four = 2

            if jing_count > need_xt_to_four:

                if jing_count == 1:  # 判断精为1

                    if si_card_num == 0:  # 刚好作两个对相同的部分
                        return False
                    elif jing_card == catch_card:  # 摸到宝牌与单张牌凑一对
                        return False
                    else:  # 非现在摸到的宝牌
                        return True
                else:  # 宝牌不止一张
                    return True
            else:  # 精牌都去作豪七胡牌条件
                return False
    else:
        return False


def ssl_hu(handcards, fulu):  # 十三烂胡牌型
    """
    判断是否胡了十三烂
    Args:
        handcards: 手牌
        fulu: 副露

    Returns:bool值

    """
    ssl_xt = feature.wait_types_13(handcards, fulu)
    return ssl_xt == 0


def nine_one_hu(handcards, fulu):  # 91胡牌类型
    """
    判断是否可胡九幺牌
    Args:
        handcards: 手牌
        fulu: 副露

    Returns:bool值

    """
    xt_91 = feature.wait_types_19(handcards, fulu)
    return xt_91 == 0


def normal_hu(handcards, fulu, jing_card, catch_card):  # 普通胡法
    """
    判断是否可胡平胡
    Args:
        handcards: 手牌
        fulu: 副露
        jing_card: 宝牌
        catch_card: 摸到的牌

    Returns: bool值

    """
    xt_normal = feature.wait_types_comm_king(handcards, fulu, jing_card)
    return xt_normal == 0


def transform_params(handcards, fulu, jing_card=0, catch_card=0):
    """
    参数格式转换：10进制—>16进制
    Args:
        handcards: 手牌
        fulu: 副露
        jing_card: 宝牌
        catch_card: 摸到的牌

    Returns:16进制的参数

    """
    # 十进制转换成16进制参数集合
    jing_card_ = tool2.f10_to_16(jing_card)
    catch_card_ = tool2.f10_to_16(catch_card)
    handcards_ = tool2.list10_to_16(tool2.deepcopy(handcards))
    fulu_ = tool2.fulu_translate(tool2.deepcopy(fulu))

    return handcards_, fulu_, jing_card_, catch_card_


def is_hu(handcards, fulu, jing_card, catch_card):  # 加入新抓手牌_在上饶麻将中
    '''
    判断是否胡牌，加入精牌,此处的牌需要转换成16进制表示
    :param handcards:  手牌
    :param fulu: 副露
    :param jing_card: 精牌
    :return: 返回是否胡牌
    '''

    # 把牌型都做成16进制
    handcards_, fulu_, jing_card_, catch_card_ = transform_params(handcards, fulu, jing_card, catch_card)
    hu_type = ""  # 胡牌类型
    flag = False

    # 豪华七对
    # if haohua_qidui(handcards_,fulu_, jing_card_,catch_card_) == True:
    #     flag = True
    #     hu_type = "豪华七对"

    # 七对
    if qidui(handcards_, fulu_, jing_card_, catch_card_) == True:
        flag = True
        hu_type = "七对"

    elif nine_one_hu(handcards_, fulu_):
        flag = True
        hu_type = "九幺"

    elif ssl_hu(handcards_, fulu_):
        flag = True
        hu_type = "十三烂"

    elif normal_hu(handcards_, fulu_, jing_card_, catch_card_) == True:
        flag = True
        hu_type = "平胡"  # 暂不考虑精吊问题

    return flag, hu_type


def min_xt(handcards, fulu, jing_card):
    """
    计算向听数
    Args:
        handcards: 手牌
        fulu: 副露
        jing_card: 宝牌

    Returns: 向听数

    """
    handcards_, fulu_, jing_card_, _ = transform_params(handcards, fulu, jing_card)

    xt_hhqd = feature.wait_types_haohua7(handcards_, fulu_, jing_card_)
    xt_qd = feature.wait_types_7(handcards_, fulu_, jing_card_)
    xt_91 = feature.wait_types_19(handcards_, fulu_)
    xt_13 = feature.wait_types_13(handcards_, fulu_)
    xt_normal = feature.wait_types_comm_king(handcards_, fulu_, jing_card_)

    return min(xt_hhqd, xt_qd, xt_91, xt_13, xt_normal)


def min_xt_add_weight(handcards, fulu, jing_card):
    """
    计算加权向听数
    Args:
        handcards: 手牌
        fulu: 副露
        jing_card:宝牌

    Returns:加权向听数

    """
    handcards_, fulu_, jing_card_, _ = transform_params(handcards, fulu, jing_card)

    # xt_hhqd = feature.wait_types_haohua7(handcards_, fulu_, jing_card_)
    xt_qd = feature.wait_types_7(handcards_, fulu_, jing_card_)
    xt_91 = feature.wait_types_19(handcards_, fulu_)
    xt_13 = feature.wait_types_13(handcards_, fulu_)
    xt_normal = feature.wait_types_comm_king(handcards_, fulu_, jing_card_)

    return min(xt_qd + 1, xt_91 - 2, xt_13 + 1, xt_normal)


if __name__ == '__main__':
    # 测试函数
    handcards = [3, 4, 5, 7, 7]
    fulu = [[34, 34, 34], [14, 14, 14, 14], [23, 24, 25]]
    jing_card = 13
    print(is_hu(handcards, fulu, jing_card, 32))
    min_num = min_xt(handcards, fulu, jing_card)
    print(min_num)
