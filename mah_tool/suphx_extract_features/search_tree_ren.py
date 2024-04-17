# -*- coding: utf-8 -*-
from ctypes import *
import ctypes
import json
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # 返回当前文件所在的目录

# from mah_tool.so_lib import shangraoMJ_v5
from mah_tool.so_lib.srmj_v5 import  shangraoMJ_v5
from mah_tool.so_lib.fan_cal import FanList

"""
        pinghu
        类变量初始化
        :param cards: 手牌　
        :param suits:副露
        :param leftNum:剩余牌数量列表
        :param discards:弃牌
        :param discards_real:实际弃牌
        :param discardsOp:场面副露
        :param round:轮数
        :param remainNum:牌墙剩余牌数量
        :param seat_id:座位号
        :param kingCard:宝牌
        :param fei_king:飞宝数
        :param op_card:动作操作牌
"""


class SearchInfo(object):

    @staticmethod
    def getFanList(paixing, cards, suits, king_card, fei_king, isHuJudge=False):
        # [清一色、门清、碰碰胡、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
        fanList = FanList(paixing, cards, suits, king_card, fei_king, isHuJudge).getFanList()
        return fanList

    @staticmethod
    def getSearchInfo(cards=[], suits=[], king_card=None, discards=[], discards_op=[], fei_king=0, remain_num=136,
                      round=0, seat_id=0):
        result = shangraoMJ_v5.recommend_card_rf(cards=cards, suits=suits, king_card=king_card, discards=discards,
                                                 discards_op=discards_op, fei_king=fei_king, remain_num=remain_num,
                                                 round=round, seat_id=seat_id)
        # [平胡 九幺　七对 十三烂]
        paixing = result[0]

        recommend_card = result[1]

        # if paixing == 0: # pinghu
        #     if SearchInfo.isPengPengHu(cards, suits, king_card):
        #         paixing = 1
        # else:
        #     paixing += 1

        fanList = SearchInfo.getFanList(paixing, cards, suits, king_card, fei_king)

        return recommend_card, paixing, fanList
# # #
# # if __name__ == '__main__':
#     # result = SearchInfo.getSearchInfo([3, 3, 3, 6, 7],[[36, 36, 36], [29, 29, 29], [9, 9, 9]],2)
#     result = SearchInfo.getSearchInfo([1, 1, 2, 51,52,53,54,55],[[49, 49, 49], [50, 50, 50]],2,fei_king=3,remain_num=0)
#     print(result)
