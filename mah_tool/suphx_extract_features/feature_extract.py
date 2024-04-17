# -*- coding:utf-8 -*-
import copy
import sys
from mah_tool import tool
from mah_tool.so_lib.sr_xt_ph import pinghu
import numpy as np
import random
from mah_tool.suphx_extract_features.search_tree_ren import SearchInfo


# 1 平胡向听数  任航师兄版本
def wait_types_comm_king(tile_list, suits, jing_card=0):
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
    _tile_list = copy.deepcopy(tile_list)

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
    _tile_list = copy.deepcopy(tile_list)
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
    _tile_list = copy.deepcopy(tile_list)

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
    _suits = copy.deepcopy(suits)
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


# 模仿suphx对牌进行编码
def suphx_cards_feature_code(cards_, channels):
    '''
    对牌集进行特征编码
    :param cards_:  牌或者牌集
    :param channels: 通道数
    :return:
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
    :return:
    '''

    # cards 为16进制
    data_copy = copy.deepcopy(data)
    features = []
    if data_type == "cards_set":
        features.extend(suphx_cards_feature_code(data_copy, channels))
    elif data_type == "seq_discards":
        seq_discards_features = []  # 弃牌的features,四个玩家的弃牌顺序，
        seq_len = 30  # 每个玩家弃牌的最大手数为30手
        for player_discard_seq in data_copy:
            cur_seq_discards_features = []  # 当前玩家的弃牌序列
            for i in range(len(player_discard_seq)):
                cur_seq_discards_features.extend(suphx_cards_feature_code(player_discard_seq[i], channels))

            seq_discards_features.extend(cur_seq_discards_features)  # 把当前已有的序列添加到features中
            need_pad_len = seq_len - len(cur_seq_discards_features)  # 需要填充的长度

            pad_features = [[0] * 34 for _ in range(need_pad_len)]
            seq_discards_features.extend(pad_features)
        features.extend(seq_discards_features)
    elif data_type == "dummy":  # 哑变量编码  此时的data为整数
        assert isinstance(data_copy, int)
        dummy_features = [[0] * 34 for _ in range(channels)]
        if 0 < data_copy <= channels:
            dummy_features[data_copy - 1] = [1] * 34
        elif data_copy == 0:
            # pass  当为0时，哑变量全为零
            pass
        else:
            print("INFO[ERROR]")
        features.extend(dummy_features)
    elif data_type == "look_ahead":  # 暂时空着
        pass

    return features


def calculate_king_sys_suphx(handCards0, fulu_, king_card, all_player_handcards, card_library, all_palyer_king_nums,
                             discards_seq, remain_card_num, self_king_num, fei_king_nums, round_, dealer_flag,
                             search=False, global_state=False, dropout_prob=0):
    '''
    返回不加前瞻特征及隐藏特征的特征
    :param handCards0: 当前要编码玩家的手牌 -> [] 1维list
    :param fulu_: 四个玩家的副露 -> [[[7,8,9],[17,17,17]], [], [], []] 3维list   位置参考说明2
    :param king_card:  宝牌 -> 1 int
    :param all_player_handcards:四个玩家的手牌 -> [[],[],[],[]]  2维list   位置参考说明2  当后面三个玩家为空时，隐藏完美信息
    :param card_library:  牌库的牌 -> [] 1维list 当为空时，隐藏完美信息
    :param all_palyer_king_nums: 四个玩家手中的宝牌数 -> [0,0,0,0] 长度为4的一维list 位置参考说明2  当后面三个玩家为0时，隐藏完美信息
    :param discards_seq:  四个玩家真实弃牌顺序-> [[], [], [], []] 2维list   位置参考说明2
    :param remain_card_num: 牌墙剩余牌 -> int
    :param self_king_num: 当前玩家的宝牌数 -> int
    :param fei_king_nums: 所有玩家的飞宝数 -> [0,0,0,0] 长度为4的一维list 位置参考说明2
    :param round_: 当前轮（手）数 -> int
    :param dealer_flag: 庄家flag，默认当前玩家为庄家 -> [1,0,0,0]
    :param search: 是否采用搜索树 默认开启
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
    king_features = suphx_data_feature_code(king_card, 1)
    features.extend(king_features)

    # 隐藏信息特征
    hiding_info_features = []
    if global_state and dropout_prob < 1:  # 开启隐藏特征
        # 对手手牌
        for player_handcards in all_player_handcards:
            hiding_info_features.extend(suphx_data_feature_code(player_handcards, 4))

        # 牌墙中的牌
        hiding_info_features.extend(suphx_data_feature_code(card_library, 4))

        # 对手手中的宝牌数
        for player_king_nums in all_palyer_king_nums:
            hiding_info_features.extend(suphx_data_feature_code(player_king_nums, 4, data_type="dummy"))

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
        hiding_info_features = [[0] * 34 for _ in range(36)]
    features.extend(hiding_info_features)

    # 所有弃牌的顺序信息
    seq_discards_features = suphx_data_feature_code(discards_seq, 1, data_type="seq_discards")
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
    search_features = [[0] * 34 for _ in range(56)]
    if search:
        # paixing -> [平胡  碰碰胡 九幺　七对 十三烂]
        # fanList -> [清一色、门清、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
        # 判断remain_card_num是否为0 为0时搜索树会报错
        if remain_card_num <= 0: remain_card_num = 1
        # print(handCards0, fulu_[0], king_card, discards_seq, fulu_, fei_king_nums[0], remain_card_num, round_ - 1, 0)
        recommend, paixing, fanList = SearchInfo.getSearchInfo(handCards0, fulu_[0], king_card, discards_seq, fulu_,
                                                               fei_king_nums[0], remain_card_num, round_ - 1, 0)
        search_features[paixing * 11] = [1] * 34
        for fan_index in range(len(fanList)):
            if fanList[fan_index] == 1:
                search_features[paixing * 11 + 1 + fan_index] = [1] * 34
    features.extend(search_features)
    return features


# zengw 20.11.11
def card_preprocess_sr_suphx(handCards0, fulu_, king_card, all_player_handcards, card_library, all_palyer_king_nums,
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
    :param all_palyer_king_nums: 四个玩家手中的宝牌数 -> [0,0,0,0] 长度为4的一维list 位置参考说明2  当后面三个玩家为0时，隐藏完美信息
    :param discards_seq:  四个玩家真实弃牌顺序-> [[], [], [], []] 2维list   位置参考说明2
    :param remain_card_num: 牌墙剩余牌 -> int
    :param self_king_num: 当前玩家的宝牌数 -> int
    :param fei_king_nums: 所有玩家的飞宝数 -> [0,0,0,0] 长度为4的一维list 位置参考说明2
    :param round_: 当前轮（手）数 -> int
    :param dealer_flag: 庄家flag，默认当前玩家为庄家 -> [1,0,0,0]
    :param search: 是否采用搜索树 默认开启
    :param global_state:是否开启隐藏信息特征，默认关闭
    :param dropout_prob: 对隐藏信息的dropout率，默认为0
    :return: 编码好的三维特征 455×34×1
    '''

    features = calculate_king_sys_suphx(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                        all_palyer_king_nums, discards_seq, remain_card_num, self_king_num,
                                        fei_king_nums, round_, dealer_flag, search, global_state, dropout_prob)

    features = np.array(features)
    features = features.T
    features = np.expand_dims(features, 0)
    features = features.transpose([2, 1, 0])  # 更换位置  转换成c × 34 × 1的格式

    return features


if __name__ == '__main__':
    # test
    handCards0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 19]
    fulu_ = [[[18, 18, 18]], [], [[41, 41, 41], [20, 20, 20], [21, 22, 23]], []]
    king_card = 5
    all_player_handcards = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 19], [2, 2, 2, 6, 6, 6, 7, 8, 9, 49, 49, 50, 51, 52],
                            [33, 35, 38, 39, 40], []]
    card_library = [53, 53, 2, 2, 9, 9, 9]
    all_palyer_king_nums = [0, 0, 0, 0]
    discards_seq = [[2, 3], [3, 2], [1, 4], [4, 1]]
    remain_card_num = 83
    self_king_num = 0
    fei_king_nums = [1, 0, 0, 0]
    round_ = 2
    discards_real_list = [2, 3, 1, 4, 3, 2, 4, 1]
    dealer_flag = [1, 0, 0, 0]

    featrues = card_preprocess_sr_suphx(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                        all_palyer_king_nums,
                                        discards_seq, remain_card_num, self_king_num, fei_king_nums, round_)
    print(featrues.shape)
