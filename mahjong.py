#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : mahjong.py
# @Description: 四人麻将对战环境
import sys
import os
import datetime
import random

import os

# from mahjongEnv.mah_tool import hu
# from mahjongEnv.mah_state import RL_state
# from  mahjongEnv.mah_player import Player_RL
import torch

from mah_tool import tool2
from mah_tool import shangrao_hu as hu
from mah_state import RL_state
from mah_player import Player_RL
from mah_tool.suphx_extract_features.search_tree_ren import SearchInfo
import logging
import math

from load_model import generator

# 全局变量：开局序号
from mah_tool.tool2 import card_preprocess_sr_suphx_1330
from sbln_learning.torch_v.reward.GRU_model import CNNModel

i = 0
import numpy as np
from mah_tool import feature_extract_v10 as fev10
import json


class Game2(object):
    '''
        两人上饶麻将，上饶麻将带宝游戏逻辑
        className:Game2
        fileName:mahjong.py
    '''

    # hu_cands_table={}#各个类共享参数
    def __init__(self, select_jing_model="random"):
        """
        构造器
        @param select_jing_model: 宝牌选择方法
        """
        # if Game2.hu_cands_table == {}: #加载胡牌表
        #     with open(file, 'r') as json_file:  # 打开文件
        #         Game2.hu_cands_table = json.load(json_file)
        self.select_jing_model = select_jing_model
        self.round = 0  # 游戏轮数
        self.discards = []  # 弃牌表
        self.player_discards = {0: [], 1: [], 2: [], 3: []}  # 特定某玩家的弃牌 #{0:[0,0,0],1:[1,0,2]}
        self.card_library = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37] * 4  # 所有牌 以十进制表示
        # 打乱牌库
        random.shuffle(self.card_library)

        self.select_jing()  # 宝牌
        self.remain_card_num = len(self.card_library)
        # 两个参数绑定
        self.outcard = -1
        self.out_seat_id = -1
        self.win_result = {0: {"win": 0, "score": 0, "fan": []}, 1: {"win": 0, "score": 0, "fan": []},
                           2: {"win": 0, "score": 0, "fan": []}, 3: {"win": 0, "score": 0, "fan": []}}  # 赢家信息
        self.data = {}  # 对局数据
        self.competition_op = [-1, -1, -1, -1]  # 竞争性op

    def reset(self):
        """
        参数重置
        """
        self.round = 0  # 游戏轮数
        self.discards = []  # 弃牌表
        self.player_discards = {0: [], 1: [], 2: [], 3: []}  # 特定某玩家的弃牌 #{0:[0,0,0],1:[1,0,2]}
        self.card_library = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
                             28, 29, 31, 32, 33, 34, 35, 36, 37] * 4  # 所有牌
        # 打乱牌库
        random.shuffle(self.card_library)

        self.select_jing()  # 宝牌
        self.remain_card_num = len(self.card_library)
        # 两个参数绑定
        self.outcard = -1
        self.out_seat_id = -1

        self.win_result = {0: {"win": 0, "score": 0, "fan": []}, 1: {"win": 0, "score": 0, "fan": []},
                           2: {"win": 0, "score": 0, "fan": []}, 3: {"win": 0, "score": 0, "fan": []}}  # 赢家信息
        self.data = {}  # 对局数据
        self.competition_op = [-1, -1, -1, -1]  # 竞争性op

    # 转换牌为key，用在胡牌算法里
    def cardsToKey(self, cardlist):
        """
        转换牌为key，用在胡牌算法里
        @param cardlist: 手牌
        @return: 胡牌算法中的key
        """
        key1 = ''
        for k in cardlist:
            if 1 <= k <= 9:
                key1 = key1 + '0' + str(k)
            else:
                key1 = key1 + str(k)
        return key1

    # 判断某位玩家是否胡牌
    # def is_hu(self, handcards, fulu, catch_card):
    #     return hu.is_hu(handcards, fulu, self.jing_card, catch_card)[0]
    #     #return  False

    # 给某位玩家发指定数量的牌
    def deal_cards(self, num):
        """
        给某位玩家发指定数量的牌
        @param num:发牌数目
        @return:发的牌
        """
        cards_list = []
        for _ in range(num):
            cards_list.append(self.card_library.pop(0))  # 发牌
        cards_list.sort()
        self.remain_card_num -= num  # 计算牌库剩余牌数
        return cards_list

    def select_jing(self):
        """
        选择宝牌
        """
        if self.select_jing_model == "random":
            self.jing_card = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                            21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37])
        else:
            jing_card = input("请输入精牌：")
            self.jing_card = int(jing_card)


class Mcts_game(Game2):
    '''
        二人上饶麻将mcts类
        className:Mcts_game
        fileName:mahjong.py
    '''

    # 获取玩家当前可见的状态信息
    def __init__(self, select_jing_model="random"):
        """
        构造器，获取玩家当前可见的状态信息
        """
        super(Mcts_game, self).__init__(select_jing_model)
        self.terminal = False
        self.hu_type = ""  # 胡牌类型
        self.players = []
        self.player_fulu = {0: [], 1: [], 2: [], 3: []}
        self.player_discards_display = {0: [], 1: [], 2: [], 3: []}

    # 重置
    def reset(self):
        """
        参数重置
        """
        super(Mcts_game, self).reset()
        self.terminal = False
        self.hu_type = ""  # 胡牌类型
        self.players = []
        self.player_fulu = {0: [], 1: [], 2: [], 3: []}
        self.player_discards_display = {0: [], 1: [], 2: [], 3: []}

    # 输出信息
    def perform(self, player, player0, player1, player2, player3):
        """
        输出信息
        @param player: 四个玩家
        @param player0: 玩家0
        @param player1: 玩家1
        @param player2: 玩家2
        @param player3: 玩家3
        """
        print("player" + str(player.seat_id) + ":")
        print("玩家", self.out_seat_id,
              "   outcard: " + str(self.outcard) + "       余牌：" + str(self.remain_card_num) + "    精牌：" + str(
                  self.jing_card))
        print("幅露：" + str(player.fulu) + "   " + "手牌：" + str(player.handcards) + "    " + "抓牌" + str(
            player.catch_card))
        print("player0:  " + "幅露：" + str(player0.fulu) + "   " + "手牌：" + str(
            player0.handcards) + "   手牌长度：" + str(
            len(player0.handcards)))
        print("player1:  " + "幅露：" + str(player1.fulu) + "   " + "手牌：" + str(
            player1.handcards) + "   手牌长度：" + str(
            len(player1.handcards)))
        print("player2:  " + "幅露：" + str(player2.fulu) + "   " + "手牌：" + str(player2.handcards) + "   " + str(
            len(player2.handcards)))
        print("player3:  " + "幅露：" + str(player3.fulu) + "   " + "手牌：" + str(player3.handcards) + "   " + str(
            len(player3.handcards)))
        print("\n")

    # 判断某位玩家是否胡牌
    def is_hu(self, handcards, fulu, catch_card):
        """
        判断某位玩家是否胡牌
        @param handcards: 手牌
        @param fulu: 副露
        @param catch_card: 摸的牌
        @return: 是否胡牌，bool值
        """
        flag, hu_type = hu.is_hu(handcards, fulu, self.jing_card, catch_card)
        if flag:
            self.hu_type = hu_type
        return flag
        # return  False

    # 返回最终奖励
    def final_reward(self, player, state):
        """
        返回最终奖励
        @param player: 玩家
        @param state: 状态
        """
        pass
        # for pp in player:
        #     #调用相关函数，让函数自己获得最后奖励
        #     card=pp.recommend_card(state)
        #     op = pp.recommend_op(state)
        # return


class Game_RL(Mcts_game):
    '''
        上饶麻将类-强化学习
        className:Game_RL
        fileName:mahjong.py
    '''

    def __init__(self, select_jing_model="random", is_render=False):
        """
        构造器，用在mcts模拟牌局的时候   根据state设置牌局信息，包括游戏信息game与玩家信息player
        Args:
            select_jing_model: 宝牌选择方式
            is_render: 是否打印相关信息
        """
        super(Game_RL, self).__init__(select_jing_model)
        self.to_do = "catch_card"
        self.is_render = is_render
        self.other_seat = [[1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2]]  # 当seat_id = 0 时，其他玩家的id为[1,2,3]
        self.dealer_seat_id = random.choice([0, 1, 2, 3])  # 随机选择庄家座位ID

    def get_state(self, player, game):
        """
        获取当前状态下指定玩家可见的状态信息
        @param player: 玩家
        @param game: 游戏类别
        @param addition_info: 额外信息
        @return: state
        """
        state = RL_state(player, game)
        return state

    def reset(self):
        """
        参数重置
        """
        super(Game_RL, self).reset()
        self.to_do = "catch_card"
        self.dealer_seat_id = random.choice([0, 1, 2, 3])  # 随机选择庄家座位ID

    # 与gym相对应的函数
    def mah_step(self, player0, player1, player2, player3, action_id=0):  # 此处修改，应该只走单步，用单步更新，不直接进行一场游戏
        """
        与gym相对应的函数，此处修改，应该只走单步，用单步更新，不直接进行一场游戏
        @param player0: 玩家0
        @param player1: 玩家1
        @param player2: 玩家2
        @param player3: 玩家3
        @param action_id: 当前玩家座位号
        @return:
        """
        self.round += 1  # 游戏轮数+1
        current_p_index = action_id  # 当前玩家ID
        player = [player0, player1, player2, player3]
        self.players = player

        def get_outcard():
            """
            丢牌
            """
            # card2 = input("player"+str(i % 2)+": "+"请输出牌：")
            state = self.get_state(player[current_p_index % 4], self)  ###############################
            t44 = datetime.datetime.now()
            card2 = player[current_p_index % 4].recommend_card2(state)  ######################
            t444 = datetime.datetime.now() - t44

            try:
                player[current_p_index % 4].handcards.remove(card2)
            except:
                # 不在手里出最后一张牌
                print("[INFO]出牌错误，默认出最后一张牌。", player[current_p_index % 4].handcards, "/t/t",
                      current_p_index % 4, card2)
                card2 = player[current_p_index % 4].handcards[-1]
                player[current_p_index % 4].handcards.remove(card2)

            self.outcard = card2
            self.out_seat_id = player[current_p_index % 4].seat_id
            self.player_discards[current_p_index % 4].append(card2)
            # self.player_discards_display[current_p_index % 4].append(card2)
            self.discards.append(card2)

        def score_op(add_score_idx, add_score):
            """
            对分数进行操作
            @param add_score_idx: 加分玩家的下标
            @param add_score: 加的分数
            """

            # 计算杠牌的加分
            self.win_result[add_score_idx]["score"] += add_score
            for idx in self.other_seat[add_score_idx]:
                self.win_result[idx]["score"] -= (add_score // 3)

        while self.card_library:  # 打四手牌

            if self.to_do == "catch_card":  # 0:00:00.000016
                t21 = datetime.datetime.now()
                card1 = self.deal_cards(1)  # 发牌
                player[current_p_index % 4].catch_card = card1[0]  # 抓牌  #########################
                player[current_p_index % 4].handcards.append(player[current_p_index % 4].catch_card)  # 加入手牌
                player[current_p_index % 4].handcards.sort()

                if self.is_render:
                    print("玩家：{}抓牌中..... 手牌：{}，  副露：{}，抓牌：{}".format(current_p_index % 4,
                                                                                player[current_p_index % 4].handcards,
                                                                                player[current_p_index % 4].fulu,
                                                                                player[current_p_index % 4].catch_card))
                # self.perform(player[i % 2], player0, player1)  # 打印
                self.to_do = "check_allow_op"  #
                t31 = datetime.datetime.now() - t21
                pass

            if self.to_do == "check_allow_op":  # 0:00:00.000363
                t22 = datetime.datetime.now()
                player[
                    current_p_index % 4].allow_op = []  # 重置     #允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
                if player[current_p_index % 4].canAnGang():  # 暗杠
                    player[current_p_index % 4].allow_op.append(6)
                if player[current_p_index % 4].canBuGang():  # 补杠
                    player[current_p_index % 4].allow_op.append(7)
                if self.is_hu(player[current_p_index % 4].handcards, player[current_p_index % 4].fulu,
                              player[current_p_index % 4].catch_card):  # 胡    ##jjjjj
                    player[current_p_index % 4].allow_op.append(8)
                player[current_p_index % 4].allow_op.append(0)
                player[current_p_index % 4].allow_op.sort()

                '''(2)执行操作'''
                if len(player[current_p_index % 4].allow_op) >= 2:  # 除“过”以外，还有其它操作
                    # print("player" + str(player[i % 2].seat_id) + "   outcard:" + str(self.outcard) + "    允许操作" + str(
                    # player[i % 2].allow_op))
                    # op=input("请输入操作：")
                    state = self.get_state(player[current_p_index % 4], self)
                    op = player[current_p_index % 4].recommend_op(state)  ##########################
                    # 执行
                    if op == 0:
                        player[current_p_index % 4].guo()
                        # get_outcard()
                        self.to_do = "output_card"
                    if op == 6:
                        player[current_p_index % 4].anGang()
                        self.player_fulu[player[current_p_index % 4].seat_id] = player[current_p_index % 4].fulu
                        # 计算杠牌的加分
                        # score_op(current_p_index % 4, 3)
                        # i = i - 1  # 轮当前玩家摸牌
                        self.to_do = "catch_card"
                    if op == 7:
                        player[current_p_index % 4].buGang()
                        self.player_fulu[player[current_p_index % 4].seat_id] = player[current_p_index % 4].fulu
                        # 摸牌
                        # 计算杠牌的加分
                        # score_op(current_p_index % 4, 3)
                        # i = i - 1  # 轮当前玩家摸牌
                        self.to_do = "catch_card"
                    if op == 8:  # 胡了
                        assert player[current_p_index % 4].seat_id == current_p_index % 4

                        # 胡牌标志分配
                        self.win_result[player[current_p_index % 4].seat_id]["win"] = 1
                        for idx in self.other_seat[current_p_index % 4]:
                            self.win_result[idx]["win"] = -1

                        # cal_fan.fan(player[i % 2], self, player[i % 2].handcards, player[i % 2].fulu, self.win_result)

                        # [平胡 九幺　七对 十三烂]
                        idx2px_dict = {0: "平胡", 1: "九幺", 2: "七对", 3: "十三烂"}
                        px2idx_dict = {v: k for k, v in idx2px_dict.items()}

                        # [清一色、门清、碰碰胡、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
                        idx2fan_dict = {0: "清一色", 1: "门清", 2: "碰碰胡", 3: "宝吊", 4: "宝还原", 5: "单吊",
                                        6: "七星", 7: "飞宝1", 8: "飞宝2", 9: "飞宝3", 10: "飞宝4"}

                        # if self.hu_type == "平胡":
                        # 胡牌基本分
                        score_op(current_p_index % 4, 3)

                        # if self.hu_type == "碰碰胡":  # 碰碰胡+2×3分
                        #     score_op(current_p_index % 4, 6)
                        # else:  # 普通胡 + 1×3分
                        #     score_op(current_p_index % 4, 3)

                        fanList = SearchInfo.getFanList(
                            paixing=px2idx_dict.get(self.hu_type, 3), cards=player[current_p_index % 4].handcards,
                            suits=player[current_p_index % 4].fulu,
                            king_card=self.jing_card,
                            fei_king=self.player_discards[current_p_index % 4].count(self.jing_card),
                            isHuJudge=True)

                        self.win_result[current_p_index % 4]["fan"].append("自摸:(" + self.hu_type + ") ")

                        # 番型分
                        for fan_index in range(len(fanList)):
                            if fanList[fan_index] == 1:
                                self.win_result[current_p_index % 4]["fan"].append(idx2fan_dict[fan_index])
                                if fan_index != 3:  # 宝吊不计分
                                    score_op(current_p_index % 4, 3)

                        self.terminal = True
                        # state = self.get_state(player[current_p_index % 4], self)
                        # self.final_reward(player, state)  # 游戏结束时，把奖励返回给玩家

                        # TODO 打印最终牌局信息
                        # print(self.win_result, "胡牌玩家：{}".format(current_p_index % 4),
                        #       "庄家ID：{}".format(self.dealer_seat_id), "jing_card", self.jing_card,
                        #       player[current_p_index % 4].handcards, player[current_p_index % 4].fulu)
                        # print("\n")
                        # return self.win_result  # 自摸胡牌结束
                        return
                    if current_p_index % 4 == 0:
                        # self.perform(player[i % 2], player0, player1)  ###
                        pass
                else:
                    # get_outcard()
                    self.to_do = "output_card"
                    # self.perform(player[i % 2], player0, player1)

                    # print("######################################################################################")
                t32 = datetime.datetime.now() - t22
                pass

            if self.to_do == "output_card":  # 0:00:00.088973
                # t23 = datetime.datetime.now()
                if (current_p_index % 4) == 0:  # 如果轮到玩家1在此出牌，则进行中断
                    return
                else:
                    get_outcard()
                    if self.is_render:
                        print("玩家：{}出牌中..... 手牌：{}，  副露：{}，出牌：{}， 庄家ID：{}\n".format(current_p_index % 4,
                                                                                                 player[
                                                                                                     current_p_index % 4].handcards,
                                                                                                 player[
                                                                                                     current_p_index % 4].fulu,
                                                                                                 self.outcard,
                                                                                                 self.dealer_seat_id))
                    self.to_do = "check_others_allow_op"

                # t33 = datetime.datetime.now() - t23
                # self.discards    ######################################

            if self.to_do == "check_others_allow_op":  # 0:00:00.000430

                t24 = datetime.datetime.now()
                other_player = [player0, player1, player2, player3]
                del other_player[current_p_index % 4]
                # other_player2=copy.copy(other_player)
                # random.shuffle(other_player)#打乱
                '''(1)检测所有玩家执行op'''
                while other_player:
                    p = other_player.pop(0)
                    # print(p.seat_id)
                    # 检测
                    # 允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
                    '''(1)检测allow_op'''
                    p.allow_op = []  # 重置
                    # 吃只适合下家
                    # 判断下家
                    if p.seat_id == ((current_p_index + 1) % 4):
                        if p.canZuoChi(self.outcard):
                            p.allow_op.append(1)
                        if p.canZhongChi(self.outcard):
                            p.allow_op.append(2)
                        if p.canYouChi(self.outcard):
                            p.allow_op.append(3)

                    if p.canPeng(self.outcard):
                        p.allow_op.append(4)
                    if p.canMingGang(self.outcard):
                        p.allow_op.append(5)
                    # 上饶麻将中没有放炮胡，取消
                    # handcards_copy = copy.deepcopy(p.handcards)
                    #                     # handcards_copy.append(self.outcard)
                    #                     # if self.is_hu(handcards_copy, p.fulu, p.catch_card) == True:  # 胡
                    #                     #     p.allow_op.append(8)
                    p.allow_op.append(0)
                    p.allow_op.sort()

                    '''(2)汇集所有玩家操作'''
                    if len(p.allow_op) == 1:  # 只有“过”
                        p.compe_op = 0
                    if len(p.allow_op) >= 2:  # 除“过”以外，还有其它操作
                        # print("player"+str(p.seat_id)+"  outcard:" + str(self.outcard) + "    允许操作" + str(p.allow_op))
                        # op = input("请输入操作：")
                        state = self.get_state(p, self)
                        op = p.recommend_op(state)  ######################################################
                        p.compe_op = op
                        # if op!=0:
                        # other_player = [player0,player1,player2,player3]  ######################
                        # a=i%2
                        # del other_player[i % 2]
                        # other_player2 = copy.copy(other_player)
                        # random.shuffle(other_player)  # 打乱e

                '''(2)执行优先级高的op'''
                self.competition_op = [player0.compe_op, player1.compe_op, player2.compe_op, player3.compe_op]

                self.competition_op[current_p_index % 4] = -1  # 不纳入考虑，非本手玩家
                run_op = max(self.competition_op)

                index1 = []
                for j in range(4):
                    if self.competition_op[j] == run_op:
                        index1.append(j)
                player_index = random.sample(index1, 1)[0]  # 随机选取
                # player_index = self.competition_op.index(run_op)
                if run_op > 0:

                    if run_op == 1:  # 左吃
                        player[player_index].zuochi(self.outcard)
                        self.player_fulu[player[player_index].seat_id] = player[player_index].fulu
                        self.discards.pop()  # 打出的牌给别人吃碰杠了
                        current_p_index = player_index
                        self.to_do = "output_card"
                    if run_op == 2:  # 中吃
                        player[player_index].zhongchi(self.outcard)
                        self.player_fulu[player[player_index].seat_id] = player[player_index].fulu
                        self.discards.pop()  # 打出的牌给别人吃碰杠了
                        current_p_index = player_index
                        self.to_do = "output_card"
                    if run_op == 3:  # 右吃
                        player[player_index].youchi(self.outcard)
                        self.player_fulu[player[player_index].seat_id] = player[player_index].fulu
                        self.discards.pop()  # 打出的牌给别人吃碰杠了
                        current_p_index = player_index
                        self.to_do = "output_card"
                    if run_op == 4:  # 碰
                        player[player_index].peng(self.outcard)
                        self.player_fulu[player[player_index].seat_id] = player[player_index].fulu
                        self.discards.pop()  # 打出的牌给别人吃碰杠了
                        current_p_index = player_index
                        self.to_do = "output_card"
                    if run_op == 5:  # 明杠
                        player[player_index].mingGang(self.outcard)
                        self.player_fulu[player[player_index].seat_id] = player[player_index].fulu

                        # score_op(player_index, 3)

                        self.discards.pop()  # 打出的牌给别人吃碰杠了
                        current_p_index = player_index
                        self.to_do = "catch_card"

                if run_op == 0:  # 所有其他玩家都选择“过”
                    player[player_index].guo()
                    current_p_index = current_p_index + 1
                    self.to_do = "catch_card"
                t34 = datetime.datetime.now() - t24
                pass

        if not self.card_library:
            self.win_result[0]["fan"].append("流局")
            self.win_result[0]["score"] -= 1  # 流局智能体扣分
            self.win_result[1]["fan"].append("流局")
            self.win_result[2]["fan"].append("流局")
            self.win_result[3]["fan"].append("流局")
            self.terminal = True
            print(self.win_result)
            print("流局")
            # return self.win_result
            return


# from dqn.BrainDQN_Nature import BrainDQN


class MahjongEnv(object):
    '''
        上饶麻将环境，与gym对应
        className:MahjongEnv
        fileName:mahjong.py
    '''

    def __init__(self):
        """
        构造器，初始化玩家实例与游戏实例
        @param four_player_model_name: 四个玩家的模型
        """
        self.action_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
                           28, 29, 31, 32, 33, 34, 35, 36, 37]
        # self.player0 = Player_RL("a", 0, "DLModel")  # 现有的PPO模型
        self.player0 = Player_RL("a", 0, "zhiyi_last")  # 现有的PPO模型
        self.player1 = Player_RL("b", 1, "zhiyi_last")
        self.player2 = Player_RL("c", 2, "zhiyi_last")
        self.player3 = Player_RL("d", 3, "zhiyi_last")
        self.game = Game_RL()
        self.features = np.zeros((1, 26, 45220))
        self.counter = 0
        self.score = 0
        self.start_flag = True
        self.reward_model = CNNModel(1330, 1, torch.device('cuda:0'))
        # self.reward_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/Globel" \
        # "/lr=0.0001_AdamW_ep230_test_loss_mean=415.8985.pth"
        # self.reward_model.load_state_dict(torch.load(self.reward_path))

    def reset(self):
        """
        参数重置
        @return: state
        """
        while True:
            global i
            i = random.choice([0, 1])
            # 重置玩家实例与游戏实例
            self.player0.reset()
            self.player1.reset()
            self.player2.reset()
            self.player3.reset()
            self.game.reset()
            self.features = np.zeros((1, 26, 45220))
            # self.features = np.zeros((1, 26, 48688))
            self.counter = 0
            self.score = 0
            self.start_flag = True

            # 给玩家发牌
            self.player0.handcards = self.game.deal_cards(13)  # 训练模型
            self.player1.handcards = self.game.deal_cards(13)  # 陪打模型
            self.player2.handcards = self.game.deal_cards(13)  # 陪打模型
            self.player3.handcards = self.game.deal_cards(13)  # 陪打模型

            # 进行游戏
            players = [self.player0, self.player1, self.player2, self.player3]

            self.game.players = players  # 初始化对手
            self.game.players[self.game.dealer_seat_id].catch_card = self.game.deal_cards(1)[0]  # 庄家先发牌

            self.game.players[self.game.dealer_seat_id].handcards.append(
                self.game.players[self.game.dealer_seat_id].catch_card)
            self.player0.handcards.sort()

            self.game.to_do = "check_allow_op"  # 检查是否有杠牌和胡牌的op
            self.game.mah_step(self.player0, self.player1, self.player2, self.player3, self.game.dealer_seat_id)

            if not self.game.terminal:  # 避免开局是天胡等情况
                break

        # 获取状态信息
        state = self.game.get_state(self.player0, self.game)
        self.state = state
        self.start_flag = True
        _, mask = tool2.get_suphx_1330_and_mask(self.state)
        return state

    def step(self, action):
        """
        gym中的step函数，出一张牌
        @param action:执行的动作
        @return:[state,reward,done,info]
        """
        # action指示要打出张牌
        if action not in self.player0.handcards:  # 对无效动作设置负面奖励，并提前终止对局
            logging.error(
                "player0.handcards:{}, 出牌：{}动作并不在手牌中，请检查！".format(self.player0.handcards, action))
            state = self.game.get_state(self.player0, self.game)
            # return state, -3, True, {}
            return state, -50, True, {}  # 模型预测的奖励值较大，故无效动作惩罚应增大
        else:
            # 获取状态信息
            if self.start_flag:  # 第一手
                state0 = self.game.get_state(self.player0, self.game)
                game0 = self.game
            global i
            # TODO 进行对全局奖励的预测
            state0 = self.game.get_state(self.player0, self.game)
            features1 = card_preprocess_sr_suphx_1330(state0, global_state=True)
            features1 = torch.tensor(features1, dtype=torch.float).view(1, 1330, 34, 1)
            r1 = self.reward_model(features1)[0].item()

            agent_out_card = action  # 仅限Agent出牌，另一个陪打模型由智一出牌
            player = [self.player0, self.player1, self.player2, self.player3]
            self.game.outcard = agent_out_card
            self.game.out_seat_id = self.player0.seat_id
            self.game.player_discards[0].append(agent_out_card)
            # self.game.player_discards_display[0].append(agent_out_card)
            self.game.discards.append(agent_out_card)
            # print("player0.catch_card:{},  player0.handcards:{},jing_card:{}, player0.fulu:{}, player0.outcard:{})".
            #       format(self.player0.catch_card, self.player0.handcards, self.game.jing_card, self.player0.fulu,
            #              agent_out_card))
            # print("\n")
            self.player0.handcards.remove(agent_out_card)
            self.game.to_do = "check_others_allow_op"

            # 进行一步游戏
            self.game.mah_step(self.player0, self.player1, self.player2, self.player3)  # player0为训练的模型

            # 获取状态信息
            state1 = self.game.get_state(self.player0, self.game)
            game1 = self.game

            features2 = card_preprocess_sr_suphx_1330(state1, global_state=True)
            features2 = torch.tensor(features2, dtype=torch.float).view(1, 1330, 34, 1)
            r2 = self.reward_model(features2)[0].item()
            # path = "/home/tonnn/.nas/.xiu/works/remote-shangrao_mj_rl_v4_suphx/my_model/"
            # path = "/home/tonnn/.nas/.xiu/works/node4-shangrao_mj_rl_v4_suphx/multi_gru_saved_xts/"
            # with tf.compat.v1.Session() as sess:
            #     sess.run(tf.compat.v1.global_variables_initializer())
            #     saver = tf.compat.v1.train.import_meta_graph(path + "model_ckpt.meta")
            #     # saver.restore(sess, path + "savedmodel_ckpt")
            #     saver.restore(sess, tf.compat.v1.train.latest_checkpoint(path))
            #     graph = tf.compat.v1.get_default_graph()
            #     # 加载模型中的操作节点
            #     self_scores = graph.get_tensor_by_name('scores:0')
            #     self_x = graph.get_tensor_by_name('inputs/x:0')
            #     self_y = graph.get_tensor_by_name('inputs/y:0')
            #
            #     y = np.array([[0]])
            #     if self.start_flag:  # 第一手
            #         self.start_flag = False
            #         self.features[0, 0] = generator(state0, game0)
            #         self.score = sess.run(self_scores, feed_dict={self_x: self.features, self_y: y})
            #     self.counter += 1
            #     self.features[0, self.counter] = generator(state1, game1)
            #     score = sess.run(self_scores, feed_dict={self_x: self.features, self_y: y})

            # 奖励函数设计 '''考虑向听数'''
            done = False
            # reward = score[0][0] - self.score[0][0]
            # self.score = score
            # print("reward:", reward)
            if self.game.terminal:  # 游戏结束
                done = True
                self.start_flag = True
            # if not self.game.terminal:  # 游戏未结束  奖励函数用7-向听数 表示
            #     cur_xt_min = hu.min_xt_add_weight(self.player0.handcards, self.player0.fulu, self.game.jing_card)
            #     if self.start_flag:  # 第一手
            #         self.start_flag = False
            #     else:
            #         # reward = (self.temp_xt - cur_xt_min) / 10  # 除以10，reward求出sqrt，防止agent过于贪婪
            #         reward = (self.temp_xt - cur_xt_min) / 100  # 除以100，防止agent过于贪婪追求中间步奖励
            #     self.temp_xt = cur_xt_min
            #
            # else:  # 游戏结束
            #     # win_flag = self.game.win_result[0]["win"]
            #     # if win_flag == -1 or win_flag == 0:
            #     #     reward = -1
            #     # else:
            #     #     reward = 1
            #     reward = self.game.win_result[0]["score"]
            #     reward = math.sqrt(reward) if reward > 0 else -math.sqrt(-reward)  # reward求出sqrt，防止agent过于贪婪
            #     done = True
            #     self.start_flag = True
            # # mah_state = tool2.card_preprocess_sr_suphx(state, search=True, global_state=True)
        self.state = state1
        return state1, r2 - r1, done, {}

    def get_zhiyi_recommend_action(self, state):
        """
        智一版推荐出牌
        @param state: 状态
        @return: 推荐打出的牌
        """
        return self.player0.recommend_card2(state)  # 因为player0为智一模型，直接使用play0的推荐出牌即可


def run():
    """
    测试函数
    """
    env = MahjongEnv()
    env.player1.model = "zhiyi_last"
    env.player2.model = "zhiyi_last"
    env.player3.model = "zhiyi_last"
    win_count = [0] * 4  # 赢的次数
    win_reward = [0] * 4  # 赢时所有分累加
    all_reward = [0] * 4  # 所有分的累加
    dealer_count = [0] * 4  # 当庄次数
    illegalN = 0  # 出非法牌的次数
    episodes = 10000  # 总的对打局数
    obv_episode = 500  # 观测结果的间隔

    # model_name = "reward0.9359512007854218_3pv5_vs_ppo_sameOp_noPerfectInfo60000"
    model_name = "RL_lr=0.0001_AdamW_ep9599_reward=1920.0000_update=25200_mer=0.8_win_acc:0.3100.pth"
    with open("/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/pretrain/log/"
              + model_name + ".txt", 'w', encoding='utf-8') as f:
        # f.write("model path:"+model_path)
        f.write(
            "当前EPOCh \t 当前赢次 \t\t\t 赢场均分 \t\t\t 累计分 \t\t 当前胜率  \t\t 流局率 \t 非法牌次数 \t 当庄次数")
        f.close()
    print("---test start---")
    for i_episode in range(episodes):

        # 获取回合 i_episode 第一个 observation
        observation = env.reset()
        # print("i_episode:", i_episode, "   observation:(手牌：", observation.handcards,"  副露：",observation.fulu,
        #       "精牌：", observation.jing_card, "   抓牌：", observation.catch_card)
        if i_episode % 100 == 0:
            print(f"------test_{i_episode}_start-------")
        while True:
            # action = observation.handcards[0]  # 选行为

            features = tool2.card_preprocess_sr_suphx_to_tensor1330(observation, True, global_state=False)
            # print("features")
            # print(features.shape)
            # print(observation.card_library)
            # print(observation.handcards)
            # exit(1)
            # action, _ = model.predict(features)
            # action = env.action_set[action]

            action = env.player0.recommend_card2(observation)
            # action = random.sample(observation.handcards, 1)[0]

            # print("action:", action)
            observation, reward, done, info = env.step(action)  # 获取下一个 state
            # print("reward:", reward,  "   observation:(手牌：", observation.handcards,"  副露：",observation.fulu,
            #   "精牌：", observation.jing_card,"   抓牌：", observation.catch_card)

            if done:

                dealer_count[env.game.dealer_seat_id] += 1
                allow_flag = False  # 判断此局是否正常结束
                for i in range(4):
                    if env.game.win_result[i]["win"] == 1:
                        win_count[i] += 1
                        allow_flag = True  # 正常结束， 达到胡牌条件
                        win_reward[i] += env.game.win_result[i]["score"]
                    all_reward[i] += env.game.win_result[i]["score"]

                if reward == -3 and not allow_flag:  # 符合非法动作的reward并且不因达到胡牌而结束
                    illegalN += 1
                # print(env.game.win_result)
                break
        if (i_episode + 1) % obv_episode == 0:
            avgWinScore = np.asarray(win_reward) / np.asarray(win_count)
            winRate = np.asarray(win_count) / obv_episode
            abortRate = 1 - sum(win_count) / obv_episode
            # "当前EPOCh \t 当前赢次 \t 赢均分 \t 累计分 \t 当前胜率  \t 流局率"
            with open("/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/pretrain/log/"
                      + model_name + ".txt", 'a', encoding='utf-8') as f:
                f.write("\n{} \t {} \t {} \t {} \t {} \t {} \t{} \t{}".format(i_episode + 1, win_count, avgWinScore,
                                                                              all_reward, winRate, abortRate, illegalN,
                                                                              dealer_count))
                f.close()
            print(
                "赢次数：{}， 赢场均分：{}， 累计得分：{}， 胜率：{}， 流局率：{}， 非法牌次数：{}".format(win_count, avgWinScore,
                                                                                                all_reward, winRate,
                                                                                                abortRate, illegalN))
            win_count = [0] * 4  # 赢的次数
            win_reward = [0] * 4  # 赢时所有分累加
            all_reward = [0] * 4  # 所有分的累加
            illegalN = 0


if __name__ == '__main__':
    '''
    player0 = Player_RL("a", 0, "random")
    player1 = Player_RL("b", 1, "random")


    game = Game_RL("/media/lei/0EAA18590EAA1859/XinJiang_LuoSiHu/interface_v8/XinJiang_majhong/mahjang_game_RL/hu_cards_table2.json")
    t1 = datetime.datetime.now()
    count = {"0": 0, "1": 0, "2": 0, "3": 0, "流局": 0}
    '''
    '''
    for i in range(100000000):  #

        random.shuffle(game.card_library)
        # player0.handcards=game.deal_cards(13)#[1,9,11,19,21,29,31,32,33,34,35,36,4]#self.deal_cards(13)#
        # player1.handcards = game.deal_cards(13)

        # handcards3 = game.random_deal_handcards(1, game.card_library)
        # game.remain_card_num-=13
        player1.handcards = game.deal_cards(13)  # [1,1,1,5,5,5,31,31,31,34,34,34,36]#self.deal_cards(13)
        player0.handcards = game.deal_cards(13)

        # game.to_do="catch_card"

        s = game.RL_start(player0, player1, i % 2)
        player0.reset()
        player1.reset()
        game.reset()
    '''
    '''
    env=MahjongEnv()
    for i_episode in range(10):

        # 获取回合 i_episode 第一个 observation
        observation = env.reset()
        print("i_episode:",i_episode ,"   observation:", observation.handcards  )

        while True:


            action = observation.handcards[0]  # 选行为
            print("action:",action)
            observation, reward, done, info = env.step(action) # 获取下一个 state
            print("reward:",reward,"  observation:",observation.handcards)

            if done:
                print(env.game.win_result)
                break
    '''
    run()
