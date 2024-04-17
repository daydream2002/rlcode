# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : mah_state.py
# @Description: 状态信息，即麻将对战的某一时刻某个玩家的视野
import copy
from mah_tool import tool


# 游戏状态信息类
# 为了加速计算，对list类型采用自定义deepcopy操作，其他类型用copy.deepcopy
class RL_state(object):  # 记录游戏之前的所有状态
    '''
        记录游戏之前的所有状态
        # 游戏状态信息类
        # 为了加速计算，对list类型采用自定义deepcopy操作，其他类型用copy.deepcopy
        className:RL_state
        fileName:mah_state.py
    '''

    def __init__(self, player, game):
        """
        构造器
        @param player:玩家座位号
        @param game: 游戏类型
        """
        # game信息
        self.game = game  # 游戏
        self.round = game.round  # 游戏轮数
        self.discards = tool.deepcopy(game.discards)  # 弃牌表
        self.player_discards = game.player_discards  # 特定某玩家的弃牌 #{0:[0,0,0],1:[1,0,2]}
        self.player_discards_display = game.player_discards_display  # 弃牌（不包括吃碰杠的牌）
        self.player_fulu = game.player_fulu  # 弃牌（包括吃碰杠的牌）
        self.players = game.players  # 所有玩家
        self.jing_card = game.jing_card  # 精牌

        self.player_handcard_num = [0, 0, 0, 0]  # 每位玩家手牌长度
        for p in self.players:
            self.player_handcard_num[p.seat_id] = len(p.handcards)

        self.card_library = tool.deepcopy(game.card_library)  # 所有牌

        # self.remain_cards = self.card_library + self.players[(player.seat_id + 1)%2] # 可能的剩余牌

        self.remain_card_num = game.remain_card_num

        # 两个参数绑定
        self.outcard = game.outcard  # 打出的牌
        self.out_seat_id = game.out_seat_id  # 刚刚出牌的玩家座位号
        self.dealer_seat_id = game.dealer_seat_id  # 庄家座位号
        self.win_result = copy.deepcopy(game.win_result)  # 赢家信息
        # self.data = []  # 对局数据
        self.competition_op = tool.deepcopy(game.competition_op)  # 竞争性op
        self.terminal = game.terminal  # 游戏是否中止

        # 私人信息
        self.model = player.model  # 玩家模型
        self.catch_card = player.catch_card  # 摸的牌
        self.seat_id = player.seat_id  # 玩家座位号
        self.name = player.name  # 玩家名称
        self.handcards = tool.deepcopy(player.handcards)  # 玩家手牌
        self.fulu = tool.deepcopy(player.fulu)  # 玩家副露
        self.allow_op = tool.deepcopy(player.allow_op)  # 允许操作
        self.allow_handcards = tool.deepcopy(player.allow_handcards)  # 允许操作的手牌
        self.compe_op = player.compe_op  # 竞争性操作(左吃，中吃，右吃，碰，明杠，胡)

        self.to_do = game.to_do  # 下一步动作
