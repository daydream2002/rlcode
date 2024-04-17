# -*- coding: utf-8 -*-
# import interface_v7.feature_extract_v7 as feature_extract
# import interface_v7.model as model
# import interface_v9.model_res_tf as model2
# import interface_v9.model_lstm_keras as model_lstm

# import interface_v9.num_waiting as nw
# import interface_v9.feature_extract_v9 as feature_extract_v9
# import interface_v7.model_new_res as model3
# import interface_v7.feature_extract_sr_pic as fesp
# import ren.pinghu as ren


# import twmj_KF.twmj_KF_v1 as KF_v1
# import twmj_KF.twmj_KF_v2 as KF_v2
import random
import shangraoMJ.shangraoMJ_v1 as shangraoMJ_v1
import shangraoMJ.shangraoMJ_v3 as shangraoMJ_v2
import shangraoMJ.shangraoMJ_v5 as shangraoMJ_v5
import requests

# import sichuanMJ.sichuanMJ_v1 as sichuanMJ_v1

# ph =pinghu.pinghu()
# import interface_v9.feature_extract_v9 as feature_extract9
# import interface_v9.model_res_tf as model9
"""
20180823 modify the example, add the discarded information
20180906 replace model_res_tf to interface_v7/model_res_tf which was trained with V1 data
20181006 fix waiting chow/pong/gong problem
"""

import time

MOP_NONE = -1
MOP_PASS = 0
MOP_LCHI = 1
MOP_MCHI = 2
MOP_RCHI = 3
MOP_PENG = 4
MOP_MGANG = 5
MOP_AGANG = 6
MOP_BGANG = 7

input_pre = []  # 记录本手之前的输入特征
output_pre = []  # 记录本手之前的输出结果


def trans10to16(i):
    """0-33 to
    0x01-0x09
    0x11-0x19
    0x21-0x29
    0x31-0x37
    10 to 16 in mahjong
    :param i:
    :return:
    """
    if i >= 0 and i <= 8:
        i += 1
    elif i >= 9 and i <= 17:
        i = i + 8
    elif i >= 18 and i <= 26:
        i = i + 15
    elif i >= 27 and i <= 33:
        i = i + 22
    return i


def translate1_37to0_33(i):
    if i >= 1 and i <= 9:
        i = i - 1
    elif i >= 11 and i <= 19:
        i = i - 2
    elif i >= 21 and i <= 29:
        i = i - 2
    elif i >= 31 and i <= 37:
        i = i - 3
    else:
        print ("translate1_37to0_33 is error,i=%d" % i)
        i = 34
    return i


def RecommendCard_shangraoMJ_v1(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)
    discards = input.get('discards', [])
    discards_op = input.get('discards_op', [])

    remain_num = input.get('remain_num', 0)
    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])

    operate_cards = user_cards.get('operate_cards', [])
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = shangraoMJ_v1.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_op=discards_op)
    return recommendCard


def RecommendOprate_shangraoMJ_v1(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    # discards = input.get('discards',[])
    discards_op = input.get('discards_op', [])
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)
    discards = input.get('discards', [])
    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op,isHu = shangraoMJ_v1.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),isHu=isHu)
    return recommend_op,isHu


def RecommendCard_shangraoMJ_v2(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)
    discards = input.get('discards', [])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 136)

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = shangraoMJ_v2.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_op=discards_op, fei_king=fei_king,remain_num=remain_num,round=round)
    return recommendCard




def RecommendOprate_shangraoMJ_v2(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    # discards = input.get('discards',[])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)
    discards = input.get('discards', [])
    round = input.get('round',0)
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = shangraoMJ_v2.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),
                                              fei_king=fei_king,isHu=isHu,round=round)
    return recommend_op,isHu

def RecommendCard_shangraoMJ_v2_thread(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)
    discards = input.get('discards', [])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 136)

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = shangraoMJ_v2_thread.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_op=discards_op, fei_king=fei_king,remain_num=remain_num,round=round,seat_id=seat_id)
    return recommendCard

def RecommendOprate_shangraoMJ_v2_thread(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    # discards = input.get('discards',[])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)
    discards = input.get('discards', [])
    round = input.get('round',0)
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = shangraoMJ_v2_thread.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),
                                              fei_king=fei_king,isHu=isHu,round=round)
    return recommend_op,isHu



def RecommendCard_shangraoMJ_v4(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)
    discards = input.get('discards', [])
    discards_real = input.get('discards_real',[])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 136)

    hands = input.get('hands',[])
    wall = input.get('wall',[])

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = shangraoMJ_v4.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_real=discards_real, discards_op=discards_op, fei_king=fei_king,
                                                 remain_num=remain_num, round=round, seat_id=seat_id, hands=hands, wall=wall)
    return recommendCard

def trandfer_discards(discards, handcards):
    discards_map = {
        0x01: 0,
        0x02: 1,
        0x03: 2,
        0x04: 3,
        0x05: 4,
        0x06: 5,
        0x07: 6,
        0x08: 7,
        0x09: 8,
        0x11: 9,
        0x12: 10,
        0x13: 11,
        0x14: 12,
        0x15: 13,
        0x16: 14,
        0x17: 15,
        0x18: 16,
        0x19: 17,
        0x21: 18,
        0x22: 19,
        0x23: 20,
        0x24: 21,
        0x25: 22,
        0x26: 23,
        0x27: 24,
        0x28: 25,
        0x29: 26,
        0x31: 27,
        0x32: 28,
        0x33: 29,
        0x34: 30,
        0x35: 31,
        0x36: 32,
        0x37: 33,
    }

    discards_list = [0] * 34

    for item in discards:
        discards_list[discards_map[item]] += 1
    for item in handcards:
        discards_list[discards_map[item]] += 1

    return discards_list


def canchi(seat_id, out_seat_id):
    if seat_id == 0:
        if out_seat_id == 3:
            return True
        else:
            return False
    else:
        if seat_id == out_seat_id + 1:
            return True
        else:
            return False


def canpeng(out_card, hand_cards):
    return hand_cards.count(out_card) >= 2


def translate_index_to_hex( i):  # 1-34转换到16进制的card
    """
    将１－３４转化为牌值
    :param i:
    :return:
    """
    if i>=0 and i<=8:
        i=i+1
    if i >= 9 and i <= 17:
        i = i + 8
    elif i >= 18 and i <= 26:
        i = i + 15
    elif i >= 27 and i <= 33:
        i = i + 22
    return i


def test_time():
    # 耗时测试
    handCardsSet = []
    # 随机生成１００手手牌信息
    for i in range(100):
        handCards = []
        for j in random.sample(range(1, 137), 14):
            handCards.append(translate_index_to_hex(j / 4))
        handCardsSet.append(handCards)
    f = open('log.txt', 'w')

    i = 0  # 几率手数
    start = time.time()
    f.write('总开始时间:' + str(start) + '\n')
    for handCards in handCardsSet:
        request['user_cards']['hand_cards'] = handCards
        s = time.time()
        out_card = RecommendCard_shangraoMJ_v1(request)
        e = time.time()
        u = e - s
        i = i + 1
        f.write('第' + str(i) + '手:\n')
        f.write('开始时间:' + str(s) + '\n')
        f.write('手牌:' + str(handCards) + '\n')
        f.write('出牌:' + str(out_card) + '\n')
        f.write('结束时间：' + str(e) + '\n')
        f.write('耗时：' + str(u) + '\n\n')

    end = time.time()
    use_time = end - start
    avg_time = float(use_time) / 100
    f.write('总结束时间：' + str(end) + '\n')
    f.write('总耗时：' + str(use_time) + '\n')
    f.write('平均每手决策时间：' + str(avg_time) + '\n')
    f.write('平均每分钟决策次数：' + str(60.0 / avg_time) + '\n')


def request_interface():
    #
    import json

    # s = requests.session()
    # s.keep_alive = False
    url = "http://http://172.81.238.92:8085/shangraoMJ/v2/outcard"
    headers = {'Connection': 'close','Content-Type': 'application/json;charset=UTF-8'}
    requests.adapters.DEFAULT_RETRIES = 5
    # headers = {'Connection': 'close' }
    request_param = {"discards":[[52,54,50,41,7,39,2,17,9,8,3,34,23,7,6,37,54,6,18],[41,25,35,53,8,21,19,4,53,41,6,1,51,39,20,36,33,22],[50,54,51,9,33,25,20,36,18,52,40,21,24,2,39,4,55,55,33,54,7],[50,52,51,17,55,5,21,52,41,39,3,40,17,9,36,22,23,50,49,55]],"discards_op":[[],[[33,34,35]],[[49,49,49],[25,24,23]],[]],"fei_king":0,"isHu":False,"king_card":18,"out_card":7,"out_seat_id":2,"remain_num":5,"round":20,"seat_id":3,"user_cards":{"hand_cards":[1,2,5,7,8,9,23,24,38,19,5,25,36],"operate_cards":[]}}
    ti=time.time()
    response = requests.post(url, data=json.dumps(request_param), headers=headers)
    tj=time.time()
    print('time=',tj-ti)


if __name__ == '__main__':
    request = {"discards":[[54,34,4,17,39,34,1,18,20,35,41,55,41,6,4,3,25,55],[49,24,35,8,35,3,33,40,9,54,40,54,39,18,17,8,8,5,5,20],
                           [18,34,40,7,36,4,38,36,7,5,21,17,9,24,36,35,34],[50,49,6,25,41,41,24,18,7,24,33,22,49,33,38,36,55]],
               "discards_op":[[[53,53,53,53],[52,52,52]],[[23,23,23,23],[22,22,22],[4,5,6]],[[51,51,51,51]],[[39,38,37]]],
               "discards_real":[[54,23,34,4,17,39,34,1,18,20,35,41,55,4,41,6,4,3,25,55],[49,51,24,35,8,35,3,33,40,9,54,40,54,39,18,17,8,8,5,5,20],
                                [18,22,34,40,39,7,36,4,38,36,7,5,21,52,17,9,24,36,35,34],[50,49,6,25,41,41,53,24,18,7,24,33,22,49,33,38,36,55]],
               "fei_king":0,"hands":[[2,19,19,19,37,37,37],[38,39,40,1],[9,25,33,50,50,52,54,55,50,1,17],[1,2,2,6,7,8,20,3,21,19]],
               "king_card":35,"out_card":17,"out_seat_id":2,"round":20,"seat_id":2,
               "user_cards":{"hand_cards":[1,2,3,4,5,6,8,8,8,9,1], "operate_cards":[[51,51,51,51]]},"wall":[2,20,21,21,3,49,25,9]}
    start = time.time()
    print ("out_card=", RecommendCard_shangraoMJ_v5(request))
    end = time.time()
    print()
    print("time=", end - start)

    request_op = {"discards":[[7,51,54,2,7,25,25,19,18,40,38,22,5],[55,54,9,51,21,17,49,23,7,6,54,17,4],[55,33,17,8,25,2,23,8,37,18,5,2,37,41],[55,50,24,49,40,25,22,36,49,9,2,33,50,21]],"discards_op":[[[23,22,21],[3,3,3]],[[35,36,37],[19,19,19],[40,39,38]],[[53,53,53],[24,23,22]],[[20,20,20],[39,39,39]]],"fei_king":0,"isHu":True,"king_card":52,"out_card":3,"out_seat_id":0,"remain_num":28,"round":15,"seat_id":0,"user_cards":{"hand_cards":[4,5,6,24,24,7,5,3],"operate_cards":[[23,22,21],[3,3,3]]}}


    startop = time.time()
    # print ("op=",RecommendOprate_shangraoMJ_v2(request_op))
    endop = time.time()
    print()
    print("time=", endop - startop)
