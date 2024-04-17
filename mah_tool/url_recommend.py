#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : url_recommend.py
# @Description: 推荐出牌、动作的接口封装
import requests
import json
import mah_tool.tool2 as tool2
from mah_tool.so_lib.shangraoMJ_v5 import recommend_op
from mah_tool.suphx_extract_features.search_tree_ren import SearchInfo
import time

headers = {
    "Content-Type": "application/json; charset=UTF-8",
    # "Referer": "http://jinbao.pinduoduo.com/index?page=5",
    # "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
}

outcard_shangrao_url = "http://172.81.238.92:8082/shangraoMJ/v6/outcard"
operate_shangrao_url = "http://172.81.238.92:8082/shangraoMJ/v6/operate"


def trans_discards(player_discards_display):
    """
    弃牌信息的格式转化，10进制->16进制
    @param player_discards_display:四个玩家的弃牌（不包括被吃、碰、杠的牌）
    @return: 16进制表示的弃牌列表
    """
    discards = [[], [], [], []]
    discards[0] = tool2.list10_to_16_2(player_discards_display[0])
    discards[1] = tool2.list10_to_16_2(player_discards_display[1])
    discards[2] = tool2.list10_to_16_2(player_discards_display[2])
    discards[3] = tool2.list10_to_16_2(player_discards_display[3])
    return discards


def trans_discards_op(player_fulu):
    """
    副露的格式转换，10进制->16进制
    @param player_fulu:玩家副露
    @return: 16进制的玩家副露
    """
    discards_op = [[], [], [], []]
    discards_op[0] = tool2.fulu_translate(player_fulu[0])
    discards_op[1] = tool2.fulu_translate(player_fulu[1])
    discards_op[2] = tool2.fulu_translate(player_fulu[2])
    discards_op[3] = tool2.fulu_translate(player_fulu[3])
    return discards_op


def state_transfer(state, seat_id, type="outcards"):
    """
    把状态对象转换成数据
    :param state:状态
    :param seat_id:玩家座位号
    :param type:出牌 or 动作
    :return:url出牌（动作）接口所需要的参数
    """
    seat_id = seat_id  # 玩家座位
    player = state.players[seat_id]
    king_card = state.jing_card  # 全场精牌
    discards = state.player_discards[seat_id]  # 默认1为agent
    catch_card = player.catch_card  # 抓牌
    handcards = player.handcards  # 玩家1手牌

    fulu_ = player.fulu  # 玩家副露

    fei_king = discards.count(king_card)  # 飞宝数量
    remain_num = len(state.card_library)  # 牌库剩余牌数
    round_ = state.round  # 轮数

    # 四个玩家的手牌(二维列表)
    hands = [e.handcards for e in state.players]

    if type == "outcards":
        # return catch_card, state.player_discards, state.player_fulu, fei_king, \
        #        king_card, round_, seat_id, handcards, fulu_, remain_num
        return hands, state.player_discards, catch_card, state.player_discards_display, state.player_fulu, fei_king, \
               king_card, round_, seat_id, handcards, fulu_, remain_num, state.card_library, state.dealer_seat_id
    else:
        return hands, state.player_discards, state.outcard, state.player_discards_display, state.player_fulu, fei_king, king_card, round_, \
               seat_id, state.out_seat_id, handcards, fulu_, remain_num, state.card_library, state.dealer_seat_id


def trans_result2Op(operate_result, handcards, isHu, out_card):
    '''
    允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
    将模型推荐的result—op结果转换成op，与动作对应
    :param operate_result: 推荐结果，十六进制
    :param handcards: 十六进制手牌
    :param isHu: 胡牌标志
    :param out_card: 操作牌，十六进制
    :return: 对应操作的index
    '''
    if not operate_result:
        operate = 0
    else:
        if len(handcards) % 3 == 2:  # 自己的回合
            # 暗杠或者补杠,或者胡牌
            if isHu:
                operate = 8
            else:
                operate = 6 if handcards.count(operate_result[0]) == 4 else 7
        else:
            # 左、中、右吃、碰、明杠
            if operate_result[0] != operate_result[1]:  # 吃
                operate = 1 + operate_result.index(out_card)
            else:  # 碰或者明杠
                operate = 4 if len(operate_result) == 3 else 5
    return operate

def outcard_shangrao(hands, discards_real, catch_card, player_discards_display, player_fulu, fei_king, king_card, round,
                     seat_id, handcards, fulu, remain_num, card_library=[], dealer_id=0):
    """
    上饶麻将推荐出牌
    Args:
        hands: 四个玩家的手牌
        discards_real:真实弃牌（包括被吃碰杠的牌）
        catch_card: 摸到的牌
        player_discards_display:弃牌
        player_fulu: 副露
        fei_king: 飞宝数
        king_card: 宝牌数
        round: 回合数
        seat_id: 座位号
        handcards: 当前玩家的手牌
        fulu: 当前玩家的副露
        remain_num: 牌墙牌数
        card_library: 牌墙
        dealer_id: 庄家座位号

    Returns: 推荐出牌

    """
    json_data = {}
    json_data["catch_card"] = tool2.f10_to_16(catch_card)  # 摸牌
    json_data["discards"] = trans_discards(player_discards_display)  # 丢牌(不包括吃碰杠)
    json_data["discards_op"] = trans_discards_op(player_fulu)  # 四个玩家的副露
    json_data["discards_real"] = trans_discards(discards_real)  # 真实丢牌(包括吃碰杠)
    json_data["fei_king"] = fei_king  # 飞宝数
    json_data["hands"] = tool2.fulu_translate(hands)  # 四个玩家的手牌
    json_data["king_card"] = tool2.f10_to_16(king_card)  # 宝牌
    json_data["round"] = round  # 回合数
    json_data["seat_id"] = seat_id  # 座位号
    json_data["dealer_id"] = dealer_id  # 庄家
    json_data["remain_num"] = remain_num  # 牌墙剩余牌数

    user_cards = {}
    user_cards["hand_cards"] = tool2.list10_to_16_2(handcards)  # 当前玩家手牌
    user_cards["operate_cards"] = tool2.fulu_translate(fulu)  # 当前玩家副露
    json_data["user_cards"] = user_cards
    json_data["wall"] = tool2.list10_to_16_2(card_library)  # 牌墙

    # print(type(json_data["king_card"]))
    # print(json_data)
    # print(json.dumps(json_data))
    data = json.dumps(json_data)
    while True:
        try:
            response = requests.post(outcard_shangrao_url, data=data, headers=headers)
        except requests.exceptions.ConnectionError:
            print('Timeout, try again_132')
            time.sleep(1)
        else:
            # 成功获取
            # print('ok')
            break
    # print(response.text)
    try:
        result = json.loads(response.text)["out_card"]
    except:
        result = user_cards["hand_cards"][0]
        print(result)

    if result == None:
        print("result == None!!!!!!!!!")
        result = user_cards["hand_cards"][0]
    if result not in user_cards["hand_cards"]:
        print(handcards, result)
    return result


def outcard_shangrao_v5(hands, discards_real, catch_card, player_discards_display, player_fulu, fei_king, king_card,
                        round,
                        seat_id, handcards, fulu, remain_num):
    """
    上饶麻将v5版推荐出牌
    Args:
        hands: 四个玩家的手牌
        discards_real: 四个玩家的真实弃牌
        catch_card: 摸到的牌
        player_discards_display: 四个玩家的弃牌
        player_fulu: 四个玩家的副露
        fei_king: 飞宝数
        king_card: 宝牌
        round: 回合数
        seat_id: 座位号
        handcards: 当前玩家的手牌
        fulu: 当前玩家的副露
        remain_num: 牌墙牌数

    Returns: 推荐打出的牌

    """
    discards = trans_discards(player_discards_display)
    discards_op = trans_discards_op(player_fulu)

    king_card = tool2.f10_to_16(king_card)
    cards = tool2.list10_to_16_2(handcards)
    suits = tool2.fulu_translate(fulu)

    recommend_card, _, _ = SearchInfo.getSearchInfo(cards, suits, king_card, discards, discards_op,
                                                    fei_king, remain_num, round - 1)
    return recommend_card


def operate_shangrao(hands, discards_real, out_card, player_discards_display, player_fulu, fei_king, king_card, round,
                     seat_id, out_seat_id, handcards, fulu, remain_num, card_library, dealer_id):
    """
    上饶麻将推荐动作
    Args:
        hands: 四个玩家的手牌
        discards_real: 真实弃牌
        out_card: 打出的牌
        player_discards_display: 弃牌
        player_fulu: 四个玩家的副露
        fei_king: 飞宝数
        king_card: 宝牌
        round: 回合数
        seat_id: 当前玩家座位号
        out_seat_id: 出牌玩家座位号
        handcards: 手牌
        fulu: 副露
        remain_num:牌墙牌数
        card_library: 牌墙
        dealer_id: 庄家座位号

    Returns:上饶麻将推荐动作

    """
    # 获取推荐操作


    # 允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
    json_data = {}
    json_data["discards"] = trans_discards(player_discards_display)
    json_data["king_card"] = tool2.f10_to_16(king_card)
    json_data["isHu"] = False
    json_data["wall"] = tool2.list10_to_16_2(card_library)
    json_data["out_seat_id"] = out_seat_id
    json_data["discards_op"] = trans_discards_op(player_fulu)
    user_cards = {}
    user_cards["hand_cards"] = tool2.list10_to_16_2(handcards)
    user_cards["operate_cards"] = tool2.fulu_translate(fulu)
    json_data["user_cards"] = user_cards
    json_data["discards_real"] = trans_discards(discards_real)
    json_data["hands"] = tool2.fulu_translate(hands)
    json_data["out_card"] = tool2.f10_to_16(out_card)
    json_data["seat_id"] = seat_id
    json_data["round"] = round
    json_data["fei_king"] = fei_king

    json_data["dealer_id"] = dealer_id  # 庄家
    json_data["remain_num"] = remain_num

    data = json.dumps(json_data)
    while True:
        try:
            response = requests.post(operate_shangrao_url, data=data, headers=headers)
        except requests.exceptions.ConnectionError:
            print('Timeout, try again_217')
            time.sleep(1)
        else:
            # 成功获取
            # print('ok')
            break
    # print(response.text)
    operate_result = []
    isHu = False
    try:
        operate_result = json.loads(response.text)["operate_cards"]
        # isHu = json.loads(response.text)["isHu"]
    except:
        print("[INFO]: url_recommend operate appear except")
        # operate = user_cards["operate_cards"]
        # print(result)

    operate = trans_result2Op(operate_result, user_cards["hand_cards"], isHu, json_data["out_card"])

    #
    # if not operate_result:
    #     operate = 0
    # else:
    #     if len(handcards) % 3 == 2:  # 自己的回合
    #         # 暗杠或者补杠,或者胡牌
    #         if isHu:
    #             operate = 8
    #         else:
    #             operate = 6 if user_cards["hand_cards"].count(operate_result[0]) == 4 else 7
    #     else:
    #         # 左、中、右吃、碰、明杠
    #         if operate_result[0] != operate_result[1]: # 吃
    #             operate = 1 + operate_result.index(json_data["out_card"])
    #         else:  # 碰或者明杠
    #             operate = 4 if len(operate_result) == 3 else 5
    return operate


def operate_shangrao_v5(out_card, player_discards_display, player_fulu, fei_king, king_card, round,
                        seat_id, out_seat_id, handcards, fulu, remain_num):
    '''
    调用v5的接口进行决策
    :param out_card:打出的牌
    :param player_discards_display:弃牌
    :param player_fulu:四个玩家的副露
    :param fei_king:飞宝数
    :param king_card:宝牌
    :param round:回合数
    :param seat_id:座位号
    :param out_seat_id:出牌玩家座位号
    :param handcards:手牌
    :param fulu:副露
    :param remain_num:牌墙牌数
    :return:推荐动作
    '''
    op_card = tool2.f10_to_16(out_card)
    discards = trans_discards(player_discards_display)
    discards_op = trans_discards_op(player_fulu)

    king_card = tool2.f10_to_16(king_card)
    cards = tool2.list10_to_16_2(handcards)
    suits = tool2.fulu_translate(fulu)

    self_turn = (len(handcards) % 3 == 2)  # 判断是否是自己的轮次

    canchi = isHu = False  # 吃和胡牌默认为False
    if not self_turn:  # 不是自己的轮次，才能判断是否可以吃
        canchi = ((out_seat_id + 1) % 4 == seat_id)  # 只可以吃上家

    op_result, _ = recommend_op(op_card, cards, suits, king_card, discards,
                                discards_op, canchi, self_turn, fei_king, isHu, round)
    operate = trans_result2Op(op_result, cards, isHu, op_card)
    return operate


def get_url_recommend(state, seat_id, local_v5=False):
    """
    推荐出牌，接口封装
    Args:
        state: 状态
        seat_id: 座位号
        local_v5: 是否搜索树v5版本

    Returns:推荐打出的牌

    """
    params = state_transfer(state, seat_id)
    if local_v5:
        recommend_card = outcard_shangrao_v5(*params)
    else:
        recommend_card = outcard_shangrao(*params)
    result = tool2.f16_to_10(recommend_card)  # 十六进制转换成十进制
    return result


def get_url_recommend_op(state, seat_id, allow_op, local_v5=False):
    '''
    允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
    获取URL推荐出牌，当local_v5 为True时，调用本地接口
    :param state: state状态
    :param seat_id:当前操作者座位id
    :param allow_op:允许的所有操作
    :param local_v5:是否启用本地的v5
    :return:
    '''
    params = state_transfer(state, seat_id, "operate")
    if local_v5:
        recommend_op = operate_shangrao_v5(*params)
    else:
        recommend_op = operate_shangrao(*params)

    if recommend_op in allow_op:
        return recommend_op
    else:
        print("推荐动作{}不在允许操作中，请检查。".format(recommend_op, allow_op))
        return 0
# date = {"discards": [[52, 54, 5], [54, 55, 50], [24, 24], [51]], "king_card": 0, "remain_num": 62, "catch_card": 5, "discards_op": [[[1, 2, 3]], [], [], [[55,55,55]]], "user_cards": {"hand_cards": [1, 1, 2, 3, 4, 5, 8, 9, 34, 35, 37, 39, 40, 41], "operate_cards": [[55,55,55]]}, "seat_id": 3, "round": 20, "fei_king": 0}


# result = outcard_shangrao(*data)
#
# result=outcard_shangrao(5,{0: [], 1: [], 2: [], 3: []},{0: [], 1: [], 2: [], 3: []},0,0,20,3,[3, 3, 15, 15, 17, 19, 25, 25, 26, 27, 31, 31, 31],[],108)
# print(result)
# {'catch_card': 34, 'discards': [[40], [55], [], []], 'discards_op': [[], [], [], []], 'fei_king': 0, 'king_card': 21, 'round': 0, 'seat_id': 1, 'remain_num': 107, 'user_cards': {'hand_cards': [4, 5, 23, 23, 24, 34, 34, 34, 38, 40, 49, 51, 53], 'operate_cards': []}}

# json_data = {}
# json_data["test"]= [[1,2,3],[4,5,6],[7,8,9]]
# json_data["zengw"] = [[[4,4,4], [5,5,5]],[[]],[[]]]
# json_data["discards"] = [[40], [55], [], []]
# json_data["hand_Cards"] = [4, 5, 23, 23, 24, 34, 34, 34, 38, 40, 49, 51, 53]
# print(json_data)
# print(json.dumps(json_data))
