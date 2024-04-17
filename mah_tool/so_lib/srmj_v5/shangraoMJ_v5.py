#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 17:02
# @Author  : Ren
# @Email   : 1543042088@qq.com
# @File    : shangraoMJ_v5.py
# @Software: PyCharm

# -*- coding:utf-8 -*-
# cython: language_level=2
# python 2.0　两整数相处会自动取整，需要人为给被除数添加float型

# -------write by lxx------------
"""
point1: 潜力值（改良牌引入）
        example:
        对于12搭子  57搭子 在表面看起来似乎概率相同
        但是应当考虑的是 实际过程中 如果来了一张4 构成124这种局面
        在舍牌的时候一般是会将1舍去，这意味着57类型是比12更优的，即嵌搭优于边搭
        另外57搭子有转换为45 78的空间 如何考虑转换的潜力
        以及需要考虑如何在搜索树当中体现出这一点
        初步思路：
            模拟摸牌时，加入边搭、嵌搭判断
        todo 验证在搜索中模拟摸牌策略中，加入改良搭子的摸取，这会显著增加搜索的宽度，运算上需要再做简化
point2  潜力值
        孤张如何评估，比如89搭子与37孤张
        37孤张具有较大潜力 89搭子是已经成型的顺子，民间打法倾向拆边搭
        如何找到理论依据？
        潜力值和需求张数有关系，如37 123456789都可以靠上成搭  成搭对的概率有1/4
        todo 验证加入潜力值是否会提高胜率
point3：剩余概率
        example:
        97:39
        4：x
          当前有四张1万为出现，牌墙中还有68张 则通过分布的概率 4:136=x:68 x=2
          大概还有2张在牌墙中 这个分布可以通过统计的方式前去验证
          todo 统计验证牌的分布是否符合规律
point4：对手建模
        采用RNN 或者LSTM 或者CNN 做对手建模
        方法：同花色等同的概念，增大数据集
             比如 打出123万 猜测万字 和打出123筒 猜测筒字
point5: 加入轮次信息。统计平均胡牌轮次，在当前轮次已经接近胡牌轮次时，不应该再追求大番，而是速胡
        需引入一个公式来衡量  参数为：平均胡牌轮数 当前轮数（若能判断其他玩家听牌与否更佳，考虑做一个听牌模型）
"""
# -------write by lxx------------

import copy
import time
import lib_MJ as MJ
import logging
# import opp_srmj as DFM
import datetime
import itertools

# 日志输出
# logger = logging.getLogger("shangraoMJ_log_v5")
# logger.setLevel(level=logging.DEBUG)
# time_now = datetime.datetime.now()
# handler = logging.FileHandler("./log_v5_%i%i%i.txt" % (time_now.year, time_now.month, time_now.day))   # 将日志写入对应的文件中
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.info("shangrao_v5 compile finished...")

# global variable
TIME_START = time.time()
w_type = 0  # lib_MJ的权重选择
ROUND = 0  # 轮数
t3Set = MJ.get_t3info()
t2Set, t2Efc, efc_t2index = MJ.get_t2info()
REMAIN_NUM = 136  # 剩余牌数
KING = None  # 宝牌
fei_king = 0  # 飞宝数

T_SELFMO = [0] * 34  # 自摸概率表，牌存在于牌墙中的概率表
LEFT_NUM = [0] * 34  # 未出现的牌的数量表
RT1 = [[0] * 34, [0] * 34]  # 危险度表
RT2 = [[0] * 34, [0] * 34]
RT3 = [[0] * 34, [0] * 34]

t1tot2_dict = {}
t1tot3_dict = {}
t2tot3_dict = {}


class Node_PH:
    def __init__(self, take=None, AAA=[], ABC=[], jiang=[], T2=[], T1=[], raw=[], taking_set=[], taking_set_w=[],
                 king_num=0,
                 fei_king=0, baohuanyuan=False):
        self.take = take  # take:宝牌 0和-1的区别
        self.AAA = AAA
        self.ABC = ABC
        self.jiang = jiang
        self.T2 = T2
        self.T1 = T1
        self.raw = raw  # 有效牌集合，待扩展
        self.king_num = king_num  # 宝牌数量
        self.fei_king = fei_king  # 飞宝数量
        self.children = []
        self.taking_set = taking_set  # 已摸牌集合
        self.baohuanyuan = baohuanyuan  # 宝还原
        self.taking_set_w = taking_set_w

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print(self.AAA, self.ABC, self.jiang, "T1=", self.T1, "T2=", self.T2, self.raw, self.taking_set, self.king_num,
              self.fei_king, self.baohuanyuan)


class Node_SSL:
    def __init__(self, take=None, taking_set=[], wan=[], tiao=[], tong=[], zi=[], T1=[], raw=[]):
        self.wan = wan
        self.tiao = tiao
        self.tong = tong
        self.zi = zi
        self.T1 = T1
        self.take = take
        self.raw = raw
        self.taking_set = taking_set
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def node_info(self):
        print(self.wan, self.tiao, self.tong, self.zi, "take ", self.take, "taking_set ", self.taking_set)


class SearchTree_PH():
    def __init__(self, hand, suits, combination_sets, king_card=None, fei_king=0):
        self.hand = hand  # 手牌
        self.suits = suits
        self.combination_sets = combination_sets  # 组合集合
        self.xts = combination_sets[0][-2]
        self.tree_dict = []
        self.king_card = king_card  # 宝牌
        self.fei_king = fei_king  # 飞宝数
        if king_card != None:
            self.king_num = hand.count(king_card)
        else:
            self.king_num = 0
        self.discard_score = {}
        self.discard_state = {}
        self.node_num = 0
        self.chang_num = 0
        self.path = []

    def expand_node(self, node):
        # node.node_info()
        # 胡牌判断
        # 先定将
        if node.jiang == []:
            has_jiang = False
            if node.king_num >= 2:  # 宝还原（宝牌数量大于等于2时，宝牌作为将牌）
                has_jiang = True  # 补上，有宝时不再搜索无将情况
                child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=[self.king_card, self.king_card],
                                T2=node.T2,
                                T1=node.T1,
                                taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                king_num=node.king_num - 2,
                                fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                node.add_child(child=child)
                self.expand_node(child)

            if node.king_num > 0:  # 宝吊,
                has_jiang = True
                taking_set = copy.copy(node.taking_set)
                taking_set.append(0)  # 宝牌当成任意牌定将牌，用0表示任意牌
                taking_set_w = copy.copy(node.taking_set_w)
                taking_set_w.append(1)  # 任意牌的获取概率为1
                child = Node_PH(take=0, AAA=node.AAA, ABC=node.ABC, jiang=[0, 0], T2=node.T2,
                                T1=node.T1,
                                taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num - 1,
                                fei_king=node.fei_king, baohuanyuan=False)
                node.add_child(child=child)
                self.expand_node(child)

            if node.king_num <= 1:
                for t2 in node.T2:  # 有将
                    T2 = MJ.deepcopy(node.T2)
                    # 从t2中找到对子作为将牌
                    if t2[0] == t2[1]:
                        has_jiang = True
                        T2.remove(t2)
                        child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=t2, T2=T2,
                                        T1=node.T1,
                                        taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                        king_num=node.king_num,
                                        fei_king=node.fei_king, baohuanyuan=False)  # 非宝吊宝还原
                        node.add_child(child=child)
                        self.expand_node(node=child)
                        # break #移除，好像也不影响，后面评估去重是按摸牌来确定的，这里也不会摸牌了

            if not has_jiang:  # todo 这里可以考虑有将时也扩展
                jiangs = copy.copy(node.T1)
                # for t2 in node.T2:  # 将T2中的牌也加入到将牌中
                #     jiangs.extend(t2)
                if jiangs == []:  # 没有t1
                    for t2 in node.T2:
                        jiangs = t2
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        for t1 in jiangs:
                            taking_set = copy.copy(node.taking_set)
                            taking_set.append(t1)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.append(1)
                            T1 = copy.copy(jiangs)
                            T1.remove(t1)
                            child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T2=T2,
                                            T1=T1,
                                            taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num,
                                            fei_king=node.fei_king, baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                else:
                    for t1 in jiangs:
                        if t1 == -1:  # 对-1不作扩展
                            continue
                        taking_set = copy.copy(node.taking_set)
                        taking_set.append(t1)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.append(1)
                        T1 = copy.copy(jiangs)
                        T1.remove(t1)
                        child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T2=node.T2,
                                        T1=T1,
                                        taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num,
                                        fei_king=node.fei_king, baohuanyuan=False)
                        node.add_child(child=child)
                        self.expand_node(node=child)

        # 胡牌判断
        elif len(node.AAA) + len(node.ABC) == 4:  # 有将牌且N=4则胡牌
            if node.king_num > 0:
                node.fei_king += node.king_num  # 多余的宝牌都没飞掉
                node.king_num = 0
                '''
                node.fei_king:是当前节点的飞宝数，即从最初的手牌经过一系列的动作达到胡牌时的飞宝数
                self.king_num:最初手牌的宝牌数
                self.fei_king:最初手牌的飞宝数
                node.fei_king == self.king_num+self.fei_king : 当前节点的飞宝数等于最初手牌的宝牌数加飞宝数，
                                                               也就是说经过一系列的动作之后，所有宝牌都飞出去了。
                '''
                if node.baohuanyuan and node.fei_king == self.king_num + self.fei_king:  # 宝牌全部飞完了，所以就不是宝还原了
                    node.baohuanyuan = False
            return

        # T3扩展
        else:
            # 当待扩展集合不为空时，使用该集合进行扩展
            if node.raw != []:  # raw可能是T1的集合，也有可能是T2的集合
                tn = node.raw[-1]  # 待扩展的牌
                raw = copy.copy(node.raw)  # 深度搜索后面的节点会改变raw，回退可能导致前面的节点raw不正确，这里需要copy
                raw.pop()
                if type(tn) == list:  # 使用t2扩展t3
                    t2 = tn
                    for item in t2tot3_dict[str(t2)]:
                        if item[1][0] == item[1][1]:
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[1])
                            ABC = node.ABC
                        else:
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[1])
                        if node.king_num > 0 and item[-2] == self.king_card:  # 宝还原
                            child = Node_PH(take=-1, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=node.taking_set,
                                            taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 1,
                                            fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num > 0 and (0 in node.jiang):  # 宝牌补一张
                            child = Node_PH(take=0, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=node.taking_set,
                                            taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 1,
                                            fei_king=node.fei_king, baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                            # normal
                        else:
                            taking_set = copy.copy(node.taking_set)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set.append(item[-2])
                            taking_set_w.append(item[-1])
                            child = Node_PH(take=item[-2], AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=taking_set, taking_set_w=taking_set_w,
                                            king_num=node.king_num,
                                            fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            # child.node_info()
                            self.expand_node(node=child)
                elif type(tn) == int:  # T1扩展成T3
                    t1 = tn
                    for item in t1tot3_dict[str(t1)]:
                        # T1 = copy.copy(node.T1)
                        flag2 = False
                        if node.king_num > 0:  # 用于处理宝还原
                            for card in item[1]:
                                if card == self.king_card:
                                    # T2 = MJ.deepcopy(node.T2)
                                    # T2.append(sorted([card, t1]))
                                    flag2 = True
                                    raw_copy = copy.copy(raw)
                                    raw_copy.append(sorted([card, t1]))
                                    child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=node.jiang, T2=node.T2,
                                                    T1=node.T1, raw=raw_copy,
                                                    taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                                    king_num=node.king_num - 1, fei_king=node.fei_king,
                                                    baohuanyuan=node.baohuanyuan)
                                    node.add_child(child=child)
                                    self.expand_node(node=child)
                        if flag2:  # 上述宝还原后不再继续扩展
                            continue

                        if item[0][0] == item[0][1]:
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[0])
                            ABC = node.ABC
                        else:
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[0])

                        if node.king_num >= 2:  # 宝牌有3张以上，直接补2张，即使其中有一张被作为宝还原也不影响
                            child = Node_PH(take=[0, 0], AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                            raw=raw,
                                            taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 2, fei_king=node.fei_king,
                                            baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num == 0:  # 宝为1或0 的处理
                            take = item[1]
                            take_w = item[-1]

                            taking_set = copy.copy(node.taking_set)
                            taking_set.extend(take)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.extend(take_w)
                            child = Node_PH(take=take, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                            raw=raw,
                                            taking_set=taking_set, taking_set_w=taking_set_w,
                                            king_num=node.king_num, fei_king=node.fei_king,
                                            baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num == 1:  # king_num=1 ,补一张牌,或者不补摸2张
                            # 用1张宝牌
                            for i in range(len(item[1])):
                                card = item[1][i]
                                take = [0, card]

                                taking_set = copy.copy(node.taking_set)
                                taking_set.append(card)
                                taking_set_w = copy.copy(node.taking_set_w)
                                taking_set_w.append(1)

                                child = Node_PH(take=take, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                                raw=raw,
                                                taking_set=taking_set, taking_set_w=taking_set_w,
                                                king_num=node.king_num - 1, fei_king=node.fei_king,
                                                baohuanyuan=False)
                                node.add_child(child=child)
                                self.expand_node(node=child)
                        else:
                            print("node.king_num==%s", (node.king_num))
                else:
                    print("tn Error")
            else:
                if node.T2 != []:  # 1、先扩展T2为T3
                    t2_sets = node.T2
                    T2 = copy.copy(node.T2)
                    for t2_set in itertools.combinations(t2_sets, min(4 - len(node.AAA) - len(node.ABC), len(t2_sets))):
                        node.T2 = copy.copy(T2)
                        node.raw = list(t2_set)
                        for t2 in node.raw:
                            node.T2.remove(t2)
                        self.expand_node(node=node)

                elif node.T1 != []:
                    t1_sets = copy.copy(node.T1)
                    # 这里移除了填充的-1，不作扩展
                    if -1 in t1_sets:
                        t1_sets.remove(-1)
                    T1 = copy.copy(node.T1)
                    for t1_set in itertools.combinations(t1_sets, min(4 - len(node.AAA) - len(node.ABC), len(t1_sets))):
                        node.T1 = copy.copy(T1)
                        node.raw = list(t1_set)
                        for t1 in node.raw:
                            node.T1.remove(t1)
                        self.expand_node(node=node)

    def generate_tree(self):
        # print("begin generate tree:-------------------------------------------------------------------------")
        kz = []
        sz = []
        for t3 in self.suits:
            if t3[0] == t3[1]:
                kz.append(t3)
            else:
                sz.append(t3)
        for cs in self.combination_sets:  # combination_sets: 组合信息集合  [AAA, ABC, aa, ab, xts, leftCards]
            root = Node_PH(take=None, AAA=cs[0] + kz, ABC=cs[1] + sz, jiang=[], T2=cs[2] + cs[3], T1=cs[-1],
                           taking_set=[], taking_set_w=[], king_num=self.king_num,
                           fei_king=self.fei_king, baohuanyuan=self.king_num > 0)
            self.tree_dict.append(root)
            self.expand_node(node=root)
            # print(root.AAA, root.ABC)
            self.DFS(node=root, path=[])

    def DFS(self, node, path):
        if (node):
            path_temp = copy.copy(path)
            path_temp.append(
                [node.AAA, node.ABC, node.raw, node.take, node.taking_set, node.taking_set_w, node.baohuanyuan])
            for child in node.children:
                self.DFS(node=child, path=path_temp)
        else:
            print(path)

    def cal_score(self, node):
        value = 1
        if node.taking_set_w != []:
            node.taking_set_w[node.taking_set_w.index(min(node.taking_set_w))] = 1
            for i in range(len(node.taking_set)):
                card = node.taking_set[i]
                if card == 0:  # 宝吊的任意牌
                    taking_rate = 1.0
                else:
                    taking_rate = T_SELFMO[MJ.convert_hex2index(card)]
                value *= taking_rate * node.taking_set_w[i]

        # 摸牌概率修正，当一张牌被重复获取时，T_selfmo修改为当前数量占未出现牌数量的比例
        taking_set = list(set(node.taking_set))
        taking_set_num = [node.taking_set.count(i) for i in taking_set]
        for i in range(len(taking_set_num)):
            n = taking_set_num[i]
            j = 0
            while n > 1:
                j += 1
                index = MJ.convert_hex2index(taking_set[i])
                if LEFT_NUM[index] >= n:
                    value *= float(LEFT_NUM[index] - j) / max(LEFT_NUM[index], 1)
                else:  # 摸牌数超过了剩余数，直接舍弃
                    value = 0
                    return value
                n -= 1

        fan = Fan_PH(kz=node.AAA, sz=node.ABC, jiang=node.jiang, fei_king=node.fei_king,
                     using_king=self.king_num + self.fei_king - node.fei_king,
                     baohuanyuan=node.baohuanyuan).fanDetect()
        score = fan * value
        return score

    def calculate_path_expectation(self, node):
        # 深度搜索
        if len(node.AAA) + len(node.ABC) == 4 and node.jiang != []:
            self.node_num += 1
            discard_set = []
            for i in range(node.fei_king - self.fei_king):
                discard_set.append(self.king_card)
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)
            taking_set_sorted = sorted(node.taking_set)
            taking_set_lable = str(taking_set_sorted)  # 转化为str可以加快查找
            if discard_set != []:
                score = self.cal_score(node=node)  # 放到外面减耗时
                # if score == 0:  # 胡牌概率为0
                #     return
            else:
                return
            # todo 这种按摸牌的评估方式是否唯一准确
            for card in list(set(discard_set)):
                if card not in self.discard_state.keys():
                    self.discard_state[card] = [[], []]
                if taking_set_lable not in self.discard_state[card][0]:
                    self.discard_state[card][0].append(taking_set_lable)
                    self.discard_state[card][-1].append(score)
                else:
                    index = self.discard_state[card][0].index(taking_set_lable)
                    if score > self.discard_state[card][-1][index]:
                        self.chang_num += 1
                        self.discard_state[card][-1][index] = score

        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        # t1 = time.time()
        self.generate_tree()
        # t2 = time.time()
        for root in self.tree_dict:
            self.calculate_path_expectation(root)
        # t3 = time.time()
        # print ("tree time:",t2 - t1, "value time:",t3 - t2)
        state_num = 0
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = [0, 0]  # [得分，牌型]
            self.discard_score[discard][0] = sum(self.discard_state[discard][-1])
            state_num += len(self.discard_state[discard][-1])

        # print ("discard_state", self.discard_state)
        # print ("discard_score", self.discard_score)
        # print ("leaf node ", self.node_num)
        # print ("state_num", state_num)
        # print ("chang_num", self.chang_num)
        # print("------", self.discard_score)
        # if not self.discard_score:
        #     print("------", self.discard_score)
        return self.discard_score


class ShiSanLan:
    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        self.cards = cards
        self.suits = suits
        self.king_card = king_card
        self.discard_state = {}
        self.discard_score = {}
        self.ssl_cards = {}
        self.tree_list = []
        self.fei_king = fei_king
        self.padding = padding
        self.ssl_three_table = [[1, 6, 9],
                                [1, 4, 9],
                                [1, 4, 7],
                                [1, 4, 8],
                                [1, 5, 9],
                                [1, 5, 8],
                                [2, 5, 9],
                                [2, 5, 8],
                                [3, 6, 9],
                                [2, 6, 9]]

        self.ssl_two_table = [[1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
                              [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
                              [3, 6], [3, 7], [3, 8], [3, 9],
                              [4, 7], [4, 8], [4, 9],
                              [5, 8], [5, 9],
                              [6, 9]]
        self.ssl_two_efc = [[[7], [8], [9]], [[8], [9]], [[9]], [[4]], [[4], [5]], [[4], [5], [6]],
                            [[8], [9]], [[9]], [[]], [[5]], [[5], [6]],
                            [[9]], [[]], [[]], [[6]],
                            [[1]], [[1]], [[1]],
                            [[1]], [[1]],
                            [[1], [2], [3]]
                            ]

        self.ssl_one_table = [[1], [9], [2], [8], [3], [7], [4], [6], [5]]
        self.ssl_one_efc = [[[4, 7], [4, 8], [4, 9], [5, 8], [5, 9], [6, 9]],  # 1
                            [[1, 4], [1, 5], [1, 6], [2, 5], [2, 6], [3, 6]],  # 9
                            [[5, 8], [5, 9], [6, 9], [7], [8]],  # 2
                            [[1, 4], [1, 5], [2, 5], [3]],  # 8
                            [[6, 9], [7], [8]],  # 3
                            [[1, 4], [2], [3]],  # 7
                            [[1, 7], [8], [9]],  # 4
                            [[1], [2], [3, 9]],  # 6
                            [[1, 8], [1, 9], [2, 8], [2, 9]]  # 5
                            ]

        self.ssl_zero_efc = [[1, 6, 9],
                             [1, 4, 9],
                             [1, 4, 7],
                             [1, 4, 8],
                             [1, 5, 9],
                             [1, 5, 8],
                             [2, 5, 9],
                             [2, 5, 8],
                             [3, 6, 9],
                             [2, 6, 9],
                             [2, 7], [3, 7], [3, 8]]
        # self.zi = [0x]

    # list:手牌，color:花色
    # 作用：返回该花色的牌值
    def add_color(self, list, color):
        return [i + color for i in list]

    # color：牌的花色，0x00,0x10,0x20代表萬、条、筒
    # return CSs=[ [,], [,], [,] ]
    def color_info(self, cards, color):
        CSs = []
        tiles = list(set(cards))  # 将重复的牌去除

        waitnumMax1 = max((tiles.count(1) + tiles.count(4) + tiles.count(7)),
                          (tiles.count(1) + tiles.count(4) + tiles.count(8)),
                          (tiles.count(1) + tiles.count(4) + tiles.count(9)),
                          (tiles.count(1) + tiles.count(5) + tiles.count(8)),
                          (tiles.count(1) + tiles.count(5) + tiles.count(9)),
                          (tiles.count(1) + tiles.count(6) + tiles.count(9)),
                          (tiles.count(2) + tiles.count(5) + tiles.count(8)),
                          (tiles.count(2) + tiles.count(5) + tiles.count(9)),
                          (tiles.count(2) + tiles.count(6) + tiles.count(9)),
                          (tiles.count(3) + tiles.count(6) + tiles.count(9)))
        waitnumMax2 = max((tiles.count(2) + tiles.count(7)),
                          (tiles.count(3) + tiles.count(7)),
                          (tiles.count(3) + tiles.count(8)), )

        # CSs = [ [同花色中的十三烂牌组合，同花色中非十三烂牌剩余牌],[ , ],....]
        if max(waitnumMax1, waitnumMax2) == 3:  # 当向听数为3 的时候 直接返回3，无有效牌
            for tb in self.ssl_three_table:
                if tb[0] in cards and tb[1] in cards and tb[2] in cards:
                    tmp = copy.copy(cards)
                    tmp.remove(tb[0])
                    tmp.remove(tb[1])
                    tmp.remove(tb[2])
                    CSs.append([self.add_color(tb, color), self.add_color(tmp, color)])
        elif max(waitnumMax1, waitnumMax2) == 2:  # 当向听数为2 的时候 返回向听数
            for i in range(len(self.ssl_two_table)):
                tb = self.ssl_two_table[i]
                if tb[0] in cards and tb[1] in cards:
                    tmp = copy.copy(cards)
                    tmp.remove(tb[0])
                    tmp.remove(tb[1])
                    CSs.append([self.add_color(tb, color), self.add_color(tmp, color)])
        elif max(waitnumMax1, waitnumMax2) == 1:  # 当向听数只有1 的时候
            for card in range(1, 10):
                if card in tiles:
                    tmp = copy.copy(cards)
                    tmp.remove(card)

                    CSs.append([[card + color], self.add_color(tmp, color)])

        else:
            CSs.append([[], []])
        # print(CSs)
        return CSs

    def ssl_CS(self):
        CSs = []
        if self.suits != []:  # 有副露，则不存在十三烂的牌
            return [[[], [], 14]]
        wan, tiao, tong, zi = MJ.split_type_s(self.cards)
        wan_CS = self.color_info(wan, 0)
        tiao_CS = self.color_info([i & 0x0f for i in tiao], 0x10)
        tong_CS = self.color_info([i & 0x0f for i in tong], 0x20)
        # zi
        zi_ssl = list(set(zi))
        zi_T1 = copy.copy(zi)
        for card in zi_ssl:
            zi_T1.remove(card)
            # zi_efc.remove(card)
        zi_CS = [[zi_ssl, zi_T1]]
        for cs_wan in wan_CS:
            for cs_tiao in tiao_CS:
                for cs_tong in tong_CS:
                    for cs_zi in zi_CS:
                        xts = 14 - len(cs_wan[0]) - len(cs_tiao[0]) - len(cs_tong[0]) - len(cs_zi[0])
                        CSs.append([cs_wan[0], cs_tiao[0], cs_tong[0], cs_zi[0],
                                    cs_wan[-1] + cs_tiao[-1] + cs_tong[-1] + cs_zi[-1], xts])

        CSs.sort(key=lambda k: k[-1], reverse=True)
        return CSs

    def efc_ssl(self, cards, type):
        cards_cp = [i & 0x0f for i in cards]
        if type <= 0x20:
            if len(cards_cp) == 0:
                efc = self.ssl_zero_efc
            elif len(cards_cp) == 1:
                efc = self.ssl_one_efc[self.ssl_one_table.index(cards_cp)]
            elif len(cards_cp) == 2:
                efc = self.ssl_two_efc[self.ssl_two_table.index(cards_cp)]
            elif len(cards_cp) == 3:
                efc = [[]]
            efc_c = []
            for s in efc:  # 2020.12.28 bug解决，这里没有花色还原
                efc_c.append([i + type for i in s])
            efc = efc_c
        else:  # 字牌的有效牌
            efc = [[]]
            for card in range(0x31, 0x38):
                if card not in cards:
                    efc[0].append(card)  # 手牌中不存在的字牌为有效牌
        return efc

    def expand_node(self, node):
        if len(node.wan) + len(node.tiao) + len(node.tong) + len(node.zi) == 14:
            # node.node_info()
            return  # 达到胡牌条件
        else:
            # 分别扩展萬、条、筒
            if node.raw != []:
                raw = copy.copy(node.raw)
                card = raw[-1]
                raw.pop()
                type = card & 0xf0  # 需要扩展的节点类型
                taking_set = copy.copy(node.taking_set)
                taking_set.append(card)
                if type == 0x00:  # 萬
                    wan = copy.copy(node.wan)
                    wan.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=wan, tiao=node.tiao, tong=node.tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x10:
                    tiao = copy.copy(node.tiao)
                    tiao.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=tiao, tong=node.tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x20:
                    tong = copy.copy(node.tong)
                    tong.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=node.tiao, tong=tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x30:
                    zi = copy.copy(node.zi)
                    zi.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=node.tiao, tong=node.tong,
                                     zi=zi, T1=node.T1, raw=raw)
                node.add_child(child)
                self.expand_node(node=child)
            else:  # 生成raw
                # raws = []
                for wan_efc in self.efc_ssl(node.wan, 0):
                    for tiao_efc in self.efc_ssl(node.tiao, 0x10):
                        for tong_efc in self.efc_ssl(node.tong, 0x20):
                            for zi_efc in self.efc_ssl(node.zi, 0x30):
                                efcs = wan_efc + tiao_efc + tong_efc + zi_efc
                                xts = 14 - len(node.wan) - len(node.tiao) - len(node.tong) - len(node.zi)
                                for efc in itertools.combinations(efcs, xts):
                                    node.raw = list(efc)
                                    # raws.append(raw)
                                    self.expand_node(node)

    def generate_tree(self):
        CSs = self.ssl_CS()
        CSs.sort(key=lambda k: k[-1], reverse=True)
        # 取xts最小的一组
        min_xts = CSs[0][-1]
        CSs_min_xts = []
        for cs in CSs:
            if cs[-1] == min_xts:
                CSs_min_xts.append(cs)

        for cs in CSs_min_xts:
            node = Node_SSL(take=None, taking_set=[], wan=cs[0], tiao=cs[1], tong=cs[2], zi=cs[3], T1=cs[4], raw=[])
            self.tree_list.append(node)
            self.expand_node(node=node)

    def cal_score(self, node):
        """
            计算节点的评估值
        :param node: 节点
        :return: 评估值
        """

        value = 1
        # print node.taking_set
        for card in node.taking_set:
            value *= T_SELFMO[MJ.convert_hex2index(card)]
        # fan检测
        fan = 8
        # 飞宝
        fei_king = self.fei_king + node.T1.count(self.king_card)
        fan *= 2 ** fei_king
        # 七星
        if len(node.zi) == 7:
            fan *= 2
        score = value * fan
        return score

    def evaluate(self, node):
        if node.children == []:  # 叶子节点
            if len(node.wan) + len(node.tiao) + len(node.tong) + len(node.zi) == 14:
                score = self.cal_score(node)
                taking_set_sorted = sorted(node.taking_set)
                discards = node.T1 + self.padding
                for discard in discards:
                    if discard not in self.discard_state.keys():
                        self.discard_state[discard] = [[], []]
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
                    elif taking_set_sorted not in self.discard_state[discard][0]:
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
        else:
            for child in node.children:
                self.evaluate(node=child)

    def get_discard_score(self):
        self.generate_tree()
        for tree in self.tree_list:
            self.evaluate(node=tree)
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = [0, 3]
                self.discard_score[discard][0] = sum(self.discard_state[discard][-1])
        return self.discard_score


class JiuYao:
    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        self.cards = cards
        self.suits = suits  # 副露
        self.king_card = king_card  # 宝牌
        self.fei_king = fei_king  # 飞宝数量
        self.padding = padding
        self.discard_score = {}
        self.discard_state = {}
        self.yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]  # 九幺牌型所能包含的牌

    def yaojiu_CS(self):
        # 判断是否是91
        CS = [[], [], 14]
        flag = True
        for suit in self.suits:
            for card in suit:
                if card not in self.yaojiu:
                    flag = False
        if not flag:
            return CS  # 当不是九幺牌型时，返回值为[[],[],14]
        CS[-1] -= len(self.suits) * 3
        for card in self.cards:
            if card in self.yaojiu:
                CS[0].append(card)
            else:
                CS[1].append(card)
        CS[-1] -= len(CS[0])
        return CS  # 最终返回的值为 [[九幺牌],[非九幺牌],[凑成九幺还差几张牌]]

    # def generate_tree(self):
    # for
    '''
    
    '''

    def get_discard_score(self):
        CS = self.yaojiu_CS()

        if CS[-1] != 14:
            value = 1  # 凑成九幺的概率
            yaojiu_take = 0
            n = 0  # 保存当前还未出现的九幺牌的总合
            for card in self.yaojiu:
                if CS[0].count(card) > 2 and CS[-1] != 1:  # todo 待完善  九幺存在刻子
                    w = 6
                else:
                    w = 1
                n += LEFT_NUM[MJ.convert_hex2index(card)]  # 将所有的九幺牌的剩余数加一起
                yaojiu_take += T_SELFMO[MJ.convert_hex2index(card)] * w  # 下一张是九幺牌的获取概率
            # for i in range(CS[-1]):
            value *= yaojiu_take ** CS[-1]  # 凑成九幺的概率

            xt = CS[-1]
            j = 0

            # 每次摸完一张九幺牌之后，对概率进行重新调整
            while xt > 1:
                j += 1
                value *= float(n - j) / max(n, 1)
                xt -= 1
                # n -=1
            # fan
            fan = 4
            fei_king = self.fei_king + CS[1].count(self.king_card)
            fan *= 2 ** fei_king  # 番型得分
            # todo 有七星91吗？
            score = value * fan  # 得分概率
            discards = CS[1] + self.padding
            for discard in discards:
                if discard not in self.discard_score:
                    self.discard_score[discard] = [0, 1]
                    self.discard_score[discard][0] = score
            # return self.discard_score
        return self.discard_score


class Node_Qidui:
    def __init__(self, take=None, AA=[], T1=[], raw=[], taking_set=[], king_num=0):
        self.take = take
        self.AA = AA
        self.T1 = T1
        self.raw = raw
        self.taking_set = taking_set
        self.king_num = king_num
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print(self.AA, self.T1, self.raw, self.taking_set, self.king_num)


class Qidui:
    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        self.cards = cards
        self.suits = suits
        self.king_card = king_card
        self.fei_king = fei_king
        self.discard_score = {}
        self.king_num = cards.count(king_card)
        self.padding = padding
        self.tree_list = []
        self.discard_state = {}

    def qidui_CS(self):
        CS = [[], [], 14]
        if self.suits != []:  # 如果有副露，则不能成为七对牌型
            return CS
        cards_rm_king = copy.copy(self.cards)  # 将宝牌单独移除
        for i in range(self.king_num):
            cards_rm_king.remove(self.king_card)
        for card in list(set(cards_rm_king)):
            n = cards_rm_king.count(card)
            if n == 1:
                CS[1].append(card)
            elif n == 2:
                CS[0].append([card, card])
            elif n == 3:
                CS[0].append([card, card])
                CS[1].append(card)
            elif n == 4:
                CS[0].append([card, card])
                CS[0].append([card, card])
        king_num = self.king_num
        # 这里把宝用掉
        while king_num > 0:
            if len(CS[0]) + king_num > 7:  # 宝还原
                CS[0].append([self.king_card, self.king_card])
                king_num -= 2
            else:
                CS[0].append([0, 0])  # 宝牌当任意牌，用0表示
                king_num -= 1
        CS[-1] -= len(CS[0]) * 2 + (7 - len(CS[0])) - 1  # 这里给七对的xt+1，减少后面选择打七对的概率
        if CS[-1] < 0:
            CS[-1] = 0
        return CS

    def expand_node(self, node):
        if len(node.AA) == 7:
            return
        else:
            if node.raw != []:
                # for card in node.raw:
                card = node.raw[-1]
                node.raw.pop()
                AA = copy.copy(node.AA)
                AA.append([card, card])

                # if node.king_num==1:
                #
                #     child = Node_Qidui(take=0, AA=AA, T1=node.T1, taking_set=node.taking_set, king_num=node.king_num-1)
                # else:
                # T1 = copy.copy(node.T1)
                # T1.remove(card)

                taking_set = copy.copy(node.taking_set)
                taking_set.append(card)
                child = Node_Qidui(take=card, AA=AA, T1=node.T1, raw=node.raw, taking_set=taking_set,
                                   king_num=node.king_num)
                node.add_child(child=child)
                self.expand_node(node=child)
            else:
                if node.T1 != []:
                    t1_sets = copy.copy(node.T1)
                    # if -1 in t1_sets:
                    #     t1_sets.remove(-1)
                    T1 = copy.copy(node.T1)
                    for t1_set in itertools.combinations(t1_sets, min(7 - len(node.AA), len(t1_sets))):
                        node.T1 = copy.copy(T1)
                        node.raw = list(t1_set)
                        for t1 in node.raw:
                            node.T1.remove(t1)
                        self.expand_node(node=node)

    def generate_tree(self):
        CS = self.qidui_CS()
        # print "qidui CS",CS
        node = Node_Qidui(take=None, AA=CS[0], T1=CS[1], taking_set=[], king_num=self.king_num)
        self.tree_list.append(node)
        self.expand_node(node=node)

    def fan(self, node):
        """
        七对番型
        :param node:
        """
        fei_king = self.fei_king + node.T1.count(self.king_card)
        if self.king_num == 0 or fei_king == self.fei_king + self.king_num:  # 无宝自摸 || 摸到了宝，但全部宝都被丢弃
            fan = 16
        else:
            fan = 12  # 七对宝吊
        fan *= 2 ** fei_king
        # 91
        jiuyao = [1, 9, 0x11, 0x19, 0x21, 0x29, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        ziyise = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        flag_jiuyao = True
        for t2 in node.AA:
            if t2[0] not in jiuyao:
                flag_jiuyao = False
                break
        flag_ziyise = True
        for t2 in node.AA:
            if t2[0] not in ziyise:
                flag_ziyise = False
                break
        if flag_jiuyao:  # 九幺，番数翻倍
            fan *= 2
        if flag_ziyise:  # 字一色，番数翻倍
            fan *= 2
        return fan

    def evaluate(self, node):
        if node.children == []:
            if len(node.AA) == 7:
                # node.node_info()
                taking_set_sorted = sorted(node.taking_set)
                value = 1
                for card in taking_set_sorted:
                    # print "card",card
                    if card == -1:  # padding扩展的牌，获取概率默认为1.0/34
                        value = 1.0 / 34
                    else:
                        value *= T_SELFMO[MJ.convert_hex2index(card)]
                fan = self.fan(node=node)
                score = value * fan  # 得分
                discards = node.T1 + self.padding
                for discard in discards:
                    if discard not in self.discard_state.keys():
                        self.discard_state[discard] = [[], []]
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
                    elif taking_set_sorted not in self.discard_state[discard][0]:
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
        else:
            for child in node.children:
                self.evaluate(child)

    def get_discard_score(self):
        # t1 = time.time()
        self.generate_tree()
        # t2 = time.time()
        for tree in self.tree_list:
            self.evaluate(tree)
        # t3=time.time()
        # print ("qidui time",t2-t1,t3-t2)
        for discard in self.discard_state.keys():
            if discard not in self.discard_score:
                self.discard_score[discard] = [0, 2]
            self.discard_score[discard][0] = sum(self.discard_state[discard][-1])
        return self.discard_score


'''
番数计算类
'''


class Fan_PH():
    def __init__(self, kz, sz, jiang, fei_king=0, using_king=0, baohuanyuan=False):
        """
        初始化类变量
        :param kz: 刻子
        :param sz: 顺子
        :param jiang: 将
        :param node: 待检测的结点
        :param fei_king: 飞宝数
        """
        self.kz = kz
        self.sz = sz
        self.jiang = jiang
        self.fei_king = fei_king
        self.using_king = using_king
        self.baohuanyuan = baohuanyuan
        self.mul = 2  # 翻倍

    # 碰碰胡
    def pengPengHu(self):
        """
        碰碰胡检测
        是否刻子树数达到４个
        :return: bool
        """
        if len(self.kz) == 4:
            # if self.usingKing==0:
            return True
        else:
            return False

    # 宝还原 x2
    # def baoHuanYuan(self):
    #
    #     if self.baohuanyuan:
    #         return True
    #     else:
    #         return False

    # 清一色 x2
    def qingYiSe(self):
        """
        清一色检测
        手牌为同一花色
        :return: bool
        """
        # todo 宝吊无法检测清一色，因为将牌无法确定
        w = 0
        ti = 0
        to = 0
        z = 0
        # print self.kz + self.sz+ self.jiang
        for t in self.kz + self.sz + [self.jiang]:
            card = t[0]
            if card != 0:
                if card & 0xf0 == 0x00:
                    w = 1
                elif card & 0xf0 == 0x10:
                    ti = 1
                elif card & 0xf0 == 0x20:
                    to = 1
                else:
                    return False

        if w + ti + to <= 1:
            return True
        else:
            return False

    def fanDetect(self):
        """
        番数计算
        基础分４分，通过调用上述的番种检测来增加基础分
        :return: int 番数
        """
        # 基础分判定
        score = 4
        if self.pengPengHu():
            # print "0"
            score *= self.mul
            if self.using_king == 0 or self.baohuanyuan:
                score *= self.mul
            score *= 2  # 碰碰胡再给2倍分

        # 翻倍机制
        # 飞宝 当可以宝吊时，将飞宝倍数得到提高
        # if 0 in self.jiang:
        #     for i in range(self.fei_king):
        #         score *= 2.5
        # else:
        for i in range(self.fei_king):
            # print "1"
            score *= self.mul

        # # 宝还原　x2
        if self.baohuanyuan:
            # print score, self.baohuanyuan,self.jiang,
            # print "2"
            score *= self.mul

        # 清一色
        if self.qingYiSe():
            score *= self.mul
            # print "3"
        # 单吊　x2
        # 这里无法处理，宝吊需要吃碰杠吃碰杠处理
        # if score>16: #得分大于16时，分数评估提高
        #     score*=1.5
        # print
        return score


'''
平胡类，相关处理方法
分为手牌拆分模块sys_info，评估cost,出牌决策，吃碰杠决策等部分
'''


class PingHu:
    '''
    '''

    def __init__(self, cards, suits, kingCard=None, fei_king=0, padding=[]):
        """
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
        cards.sort()
        self.cards = cards
        self.suits = suits
        self.kingCard = kingCard
        self.fei_king = fei_king
        self.padding = padding
        self.kingNum = cards.count(kingCard)

    @staticmethod
    def split_type_s(cards=[]):
        """
        功能：手牌花色分离，将手牌分离成万条筒字各色后输出
        :param cards: 手牌　[]
        :return: 万,条,筒,字　[],[],[],[]
        """
        cards_wan = []
        cards_tiao = []
        cards_tong = []
        cards_zi = []
        for card in cards:
            if card & 0xF0 == 0x00:
                cards_wan.append(card)
            elif card & 0xF0 == 0x10:
                cards_tiao.append(card)
            elif card & 0xF0 == 0x20:
                cards_tong.append(card)
            elif card & 0xF0 == 0x30:
                cards_zi.append(card)
        return cards_wan, cards_tiao, cards_tong, cards_zi

    @staticmethod
    def get_effective_cards(dz_set=[]):
        """
        获取有效牌
        :param dz_set: 搭子集合 list [[]]
        :return: 有效牌 list []
        """
        effective_cards = []
        for dz in dz_set:
            if len(dz) == 1:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0]:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
        effective_cards = set(effective_cards)  # set 和list的区别？
        return list(effective_cards)

    # 判断３２Ｎ是否存在于ｃａｒｄｓ中
    @staticmethod
    def in_cards(t32=[], cards=[]):
        """
        判断３２Ｎ是否存在于ｃａｒｄｓ中
        :param t32: ３Ｎ或2N组合牌
        :param cards: 本次判断的手牌
        :return: bool
        """
        for card in t32:
            if card not in cards:
                return False
        return True

    @staticmethod
    def get_32N(cards=[]):
        """
        功能：计算所有存在的手牌的３Ｎ与２Ｎ的集合，例如[3,4,5]　，将得到[[3,4],[3,5],[4,5],[3,4,5]]
        思路：为减少计算量，对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
        :param cards: 手牌　[]
        :return: 3N与2N的集合　[[]]
        """
        cards.sort()
        kz = []
        sz = []
        aa = []
        ab = []
        ac = []
        lastCard = 0
        # 对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
        if len(cards) >= 12:
            for card in cards:
                if card == lastCard:  #
                    continue
                else:
                    lastCard = card
                if cards.count(card) >= 3:
                    kz.append([card, card, card])
                elif cards.count(card) >= 2:
                    aa.append([card, card])
                if card + 1 in cards and card + 2 in cards:
                    sz.append([card, card + 1, card + 2])
                else:
                    if card + 1 in cards:
                        ab.append([card, card + 1])
                    if card + 2 in cards:
                        ac.append([card, card + 2])
        else:
            for card in cards:
                if card == lastCard:
                    continue
                else:
                    lastCard = card
                if cards.count(card) >= 3:
                    kz.append([card, card, card])
                if cards.count(card) >= 2:
                    aa.append([card, card])
                if card + 1 in cards and card + 2 in cards:
                    sz.append([card, card + 1, card + 2])
                if card + 1 in cards:
                    ab.append([card, card + 1])
                if card + 2 in cards:
                    ac.append([card, card + 2])
        return kz + sz + aa + ab + ac

    def extract_32N(self, cards=[], t32_branch=[], t32_set=[]):
        """
        功能：递归计算手牌的所有组合信息，并存储在t32_set，
        思路: 每次递归前检测是否仍然存在３２N的集合,如果没有则返回出本此计算的结果，否则在手牌中抽取该３２N，再次进行递归
        :param cards: 手牌
        :param t32_branch: 本次递归的暂存结果
        :param t32_set: 所有组合信息
        :return: 结果存在t32_set中
        """
        t32N = self.get_32N(cards=cards)

        if len(t32N) == 0:
            t32_set.extend(t32_branch)
            # t32_set.extend([cards])
            t32_set.append(0)
            t32_set.extend([cards])
        else:
            for t32 in t32N:
                if self.in_cards(t32=t32, cards=cards):
                    cards_r = copy.copy(cards)
                    for card in t32:
                        cards_r.remove(card)
                    t32_branch.append(t32)
                    self.extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                    if len(t32_branch) >= 1:
                        t32_branch.pop(-1)

    def tree_expand(self, cards):
        """
        功能：对extract_32N计算的结果进行处理同一格式，计算万条筒花色的组合信息
        思路：对t32_set的组合信息进行格式统一，分为[kz,sz,aa,ab,xts,leftCards]保存，并对划分不合理的地方进行过滤，例如将３４５划分为35,4为废牌的情况
        :param cards: cards [] 万条筒其中一种花色手牌
        :return: allDeWeight　[kz,sz,aa,ab,xts,leftCards] 去除不合理划分情况的组合后的组合信息
        """
        all = []
        t32_set = []
        self.extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)
        kz = []
        sz = []
        t2N = []
        aa = []
        length_t32_set = len(t32_set)
        i = 0
        # for i in range(len(t32_set)):
        while i < length_t32_set:
            t = t32_set[i]
            flag = True  # 本次划分是否合理
            if t != 0:
                if len(t) == 3:

                    if t[0] == t[1]:
                        kz.append(t)
                    else:
                        sz.append(t)  # print (sub)
                elif len(t) == 2:
                    if t[1] == t[0]:
                        aa.append(t)
                    else:
                        t2N.append(t)

            else:
                '修改，使计算时间缩短'
                leftCards = t32_set[i + 1]
                efc_cards = self.get_effective_cards(dz_set=t2N)  # t2N中不包含ａａ
                # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
                for card in leftCards:
                    if card in efc_cards:
                        flag = False
                        break

                if flag:
                    all.append([kz, sz, aa, t2N, 0, leftCards])
                kz = []
                sz = []
                aa = []
                t2N = []
                i += 1
            i += 1

        allSort = []  # 给每一个元素排序
        allDeWeight = []  # 排序去重后

        for e in all:
            for f in e:
                if f == 0:  # 0是xts位，int不能排序
                    continue
                else:
                    f.sort()
            allSort.append(e)

        for a in allSort:
            if a not in allDeWeight:
                allDeWeight.append(a)

        allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)  # 居然可以这样排序！！
        return allDeWeight

    @staticmethod
    def zi_expand(cards=[]):
        """
        功能：计算字牌组合信息
        思路：字牌组合信息需要单独计算，因为没有字顺子，迭代计算出各张字牌的２Ｎ和３Ｎ的情况，由于某些情况下，可能只会需要ａａ作为将牌的情况，同时需要刻子和ａａ的划分结果
        :param cards: 字牌手牌
        :return: ziBranch　字牌的划分情况　[kz,sz,aa,ab,xts,leftCards]
        """
        cardList = []
        for i in range(7):
            cardList.append([])
        ziCards = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        for card in ziCards:
            index = (card & 0x0f) - 1
            # print(index)

            if cards.count(card) == 4:
                # 此结构为[3N,2N,leftCards]
                cardList[index].append([[[card, card, card]], [], [], [], 0, [card]])
            elif cards.count(card) == 3:
                cardList[index].append([[[card, card, card]], [], [], [], 0, []])
                cardList[index].append([[], [], [[card, card]], [], 0, [card]])
            elif cards.count(card) == 2:

                cardList[index].append([[], [], [[card, card]], [], 0, []])
            elif cards.count(card) == 1:
                cardList[index].append([[], [], [], [], 0, [card]])
            else:
                cardList[index].append([[], [], [], [], 0, []])

        ziBranch = []
        for c1 in cardList[0]:
            for c2 in cardList[1]:
                for c3 in cardList[2]:
                    for c4 in cardList[3]:
                        for c5 in cardList[4]:
                            for c6 in cardList[5]:
                                for c7 in cardList[6]:
                                    branch = []
                                    for n in range(6):
                                        branch.append(c1[n] + c2[n] + c3[n] + c4[n] + c5[n] + c6[n] + c7[n])
                                    ziBranch.append(branch)
        return ziBranch

    def pinghu_CS(self, cards=[], suits=[], t1=[]):
        if cards == []:
            cards = self.cards
            suits = self.suits
        cards.sort()
        # 加入一些特殊的处理 例如碰碰胡的CS生成
        CS = self.pinghu_CS2(cards, suits, t1)
        kingNum = 0
        RM_King = copy.copy(cards)
        if self.kingCard != None:
            kingNum = cards.count(self.kingCard)
            for i in range(kingNum):
                RM_King.remove(self.kingCard)
        pph = MJ.pengpengHu(outKingCards=RM_King, suits=suits, kingNum=kingNum)
        if pph[0] not in CS and pph[0][-2] <= CS[0][-2] + 1:
            CS += pph
        return CS

    def pinghu_CS2(self, cards=[], suits=[], t1=[]):
        """
        功能：综合计算手牌的组合信息
        思路：对手牌进行花色分离后，单独计算出每种花色的组合信息　，再将其综合起来，计算每个组合向听数，最后输出最小向听数及其加一的组合
        :param cards: 手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :param kingCard: 宝牌
        :return: 组合信息
        """
        # 去除宝牌计算信息，后面出牌和动作决策再单独考虑宝牌信息
        if cards == []:
            cards = self.cards
            suits = self.suits
        RM_King = copy.copy(cards)
        kingNum = 0
        if self.kingCard != None:
            kingNum = cards.count(self.kingCard)
            for i in range(kingNum):
                RM_King.remove(self.kingCard)

        # 花色分离
        wan, tiao, tong, zi = self.split_type_s(RM_King)
        wan_expd = self.tree_expand(cards=wan)
        tiao_expd = self.tree_expand(cards=tiao)
        tong_expd = self.tree_expand(cards=tong)
        zi_expd = self.zi_expand(cards=zi)

        all = []
        for i in wan_expd:
            for j in tiao_expd:
                for k in tong_expd:
                    for m in zi_expd:
                        branch = []
                        # 将每种花色的4个字段合并成一个字段
                        for n in range(6):
                            branch.append(i[n] + j[n] + k[n] + m[n])
                        branch[-1] += self.padding + t1
                        all.append(branch)

        # 将获取概率为０的组合直接丢弃到废牌中 todo 由于有宝，这里也可能会被宝代替
        # 移到了出牌决策部分处理
        if self.kingNum <= 1:  # 这里只考虑出牌、宝做宝吊的情况
            for a in all:
                for i in range(len(a[3]) - 1, -1, -1):
                    ab = a[3][i]
                    efc = self.get_effective_cards([ab])
                    if sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) == 0:
                        a[3].remove(ab)
                        a[-1].extend(ab)
                        # logger.info("remove rate 0 ab,%s,%s,%s,a=%s",self.cards,self.suits,self.kingCard,a)

        # 计算向听数
        # 计算拆分组合的向听数
        all = MJ.cal_xts(all, suits, kingNum)

        # 获取向听数最小的all分支
        min_index = 0
        for i in range(len(all)):
            if all[i][4] > all[0][4]:  # xts+１以下的组合
                min_index = i
                break

        if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予ａｌｌ长度
            min_index = len(all)

        all = all[:min_index]

        # 处理向听数为0时的情况，需要从中依次选择一张牌作为t1
        if all[0][-2] == 0 and all[0][-1] == []:
            all = []
            for card in list(set(cards)):
                cards_ = copy.copy(cards)
                cards_.remove(card)
                all += self.pinghu_CS2(cards=cards_, suits=suits, t1=[card])
        return all

    def left_card_weight(self, card, left_num, need_jiang=False):
        """
        功能：对废牌组合中的每张废牌进行评估，计算其成为３Ｎ的概率
        思路：每张牌能成为３N的情况可以分为先成为搭子，在成为３Ｎ２步，成为搭子的牌必须自己摸到，而成为kz,sz可以通过吃碰。刻子为获取２张相同的牌，顺子为其邻近的２张牌
        :param card: 孤张牌
        :param left_num: 剩余牌
        :return: 评估值
        """

        # if self.remainNum==0:
        #     remainNum=1
        # else:
        #     remainNum = self.remainNum
        # remainNum = 1
        i = convert_hex2index(card)

        if need_jiang:
            return left_num[i]
        # d_w = 0

        # if left_num[i] == self.remainNum:
        #     sf = float(self.leftNum[i])
        # else:
        #     sf = float(left_num[i]) / remainNum * float((left_num[i] - 1)) / remainNum * 6

        if left_num[i] > 1:
            aa = left_num[i] * (left_num[i] - 1) * 4
        else:
            aa = left_num[i]
        if card >= 0x31:  # kz概率
            # todo if card == fengwei:
            # if card >= 0x35 and left_num[i] >= 2:
            #     d_w = left_num[i] * left_num[i] * 2  # bug 7.22 修正dw-d_w
            # else:
            d_w = aa  # 7.22 １６:３５ 去除字牌
        elif card % 16 == 1:  # 11＋２3
            d_w = aa + left_num[i + 1] * left_num[i + 2] * 2
        elif card % 16 == 2:  # 22+13+3(14)+43   222 123 234
            d_w = aa + left_num[i - 1] * left_num[i + 1] * 2 + left_num[i + 1] * left_num[i + 2] * 2
        elif card % 16 == 8:  # 888 678 789
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2 + left_num[i - 1] * left_num[i + 1] * 2
        elif card % 16 == 9:  # 999 789
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2
        # 删除多添加的×２
        else:  # 555 345 456 567
            # print (left_num)
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2 + left_num[i - 1] * left_num[i + 1] * 2 + left_num[i + 1] * \
                  left_num[
                      i + 2] * 2
        # if card<=0x31:
        #     if (card%0x0f==3 or card %0x0f==7): #给金3银7倍数
        #         d_w*=1.5
        #     elif card%0x0f==5:
        #         d_w*=1.2
        # print("i=", i, d_w)
        return d_w


def translate16_33(i):
    """
    将牌值１６进制转化为０－３３的下标索引
    :param i: 牌值
    :return: 数组下标
    """
    i = int(i)
    if i >= 0x01 and i <= 0x09:
        i = i - 1
    elif i >= 0x11 and i <= 0x19:
        i = i - 8
    elif i >= 0x21 and i <= 0x29:
        i = i - 15
    elif i >= 0x31 and i <= 0x37:
        i = i - 22
    else:
        # i=1/0
        # print ("translate16_33 is error,i=%d" % i)
        i = -1
    return i


def convert_hex2index(a):
    """
    将牌值１６进制转化为０－３３的下标索引
    :param a: 牌
    :return: 数组下标
    """
    if a > 0 and a < 0x10:
        return a - 1
    if a > 0x10 and a < 0x20:
        return a - 8
    if a > 0x20 and a < 0x30:
        return a - 15
    if a > 0x30 and a < 0x40:
        return a - 22


def trandfer_discards(discards, discards_op, handcards):
    """
    获取场面剩余牌数量
    计算手牌和场面牌的数量，再计算未知牌的数量
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param handcards: 手牌
    :return: left_num, discards_list　剩余牌列表，已出现的牌数量列表
    """
    discards_map = {0x01: 0, 0x02: 1, 0x03: 2, 0x04: 3, 0x05: 4, 0x06: 5, 0x07: 6, 0x08: 7, 0x09: 8, 0x11: 9, 0x12: 10,
                    0x13: 11, 0x14: 12, 0x15: 13, 0x16: 14, 0x17: 15, 0x18: 16, 0x19: 17, 0x21: 18, 0x22: 19, 0x23: 20,
                    0x24: 21, 0x25: 22, 0x26: 23, 0x27: 24, 0x28: 25, 0x29: 26, 0x31: 27, 0x32: 28, 0x33: 29, 0x34: 30,
                    0x35: 31, 0x36: 32, 0x37: 33, }
    left_num = [4] * 34
    discards_list = [0] * 34
    d = []
    if isinstance(discards, dict):
        d.append(discards[0])
        d.append(discards[1])
        d.append(discards[2])
        d.append(discards[3])
        discards = d
    for per in discards:
        for item in per:
            discards_list[discards_map[item]] += 1
            left_num[discards_map[item]] -= 1
    for seat_op in discards_op:
        for op in seat_op:
            for item in op:
                discards_list[discards_map[item]] += 1
                left_num[discards_map[item]] -= 1
    for item in handcards:
        left_num[discards_map[item]] -= 1

    # print ("trandfer_discards,left_num=",left_num)
    return left_num, discards_list


# 获取ｌｉｓｔ中的最小值和下标
def get_min(list=[]):
    """
    获取最小ｘｔｓ的下标
    :param list: 向听数列表
    :return: 返回最小向听数及其下标
    """
    min = 14
    index = 0
    for i in range(len(list)):
        if list[i] < min:
            min = list[i]
            index = i
    return min, index


def pre_king(king_card=None):
    """
    计算宝牌的前一张
    :param king_card: 宝牌
    :return:宝牌的前一张牌
    """
    if king_card == None:
        return None
    if king_card == 0x01:
        return 0x09
    elif king_card == 0x11:
        return 0x19
    elif king_card == 0x21:
        return 0x29
    elif king_card == 0x31:
        return 0x37
    else:
        return king_card - 1


def value_t1(card):
    value = 0
    if card != -1:
        for e in t1tot3_dict[str(card)]:
            v = 1
            for i in range(len(e[1])):
                v *= T_SELFMO[MJ.convert_hex2index(e[1][i])] * e[-1][i]
            value += v
    return value


def get_score_dict(cards, suits, king_card, fei_king, padding=[], max_xts=14):
    # 寻找向听数在阈值内的牌型
    PH = PingHu(cards=cards, suits=suits, kingCard=king_card, fei_king=fei_king, padding=padding)
    SSL = ShiSanLan(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    JY = JiuYao(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    QD = Qidui(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    # 组合信息
    CS_PH = PH.pinghu_CS2()  # CS=[kz, sz, aa, ab, xts, leftcard]
    # print("CS_PH:-------------------------------------------------------------------------")
    # for cs in CS_PH:
    #     print(cs)
    CS_SSL = SSL.ssl_CS()  # CS=[十三烂萬组合，十三烂条组合，十三烂筒组合，十三烂字组合，非十三烂牌集合，xts]
    # print("CS_SSL:-------------------------------------------------------------------------")
    # for cs in CS_SSL:
    #     print(cs)
    CS_JY = JY.yaojiu_CS()  # CS=[[九幺牌集合],[非九幺牌集合],[向听数]]
    # print("CS_JY:-------------------------------------------------------------------------")
    # print(CS_JY)
    CS_QD = QD.qidui_CS()  # CS=[[对子aa集合],[弃牌集合],向听数]
    # print("CS_QD:-------------------------------------------------------------------------")
    # print(CS_QD)
    # 向听数

    xts_list = [CS_PH[0][-2], CS_SSL[0][-1], CS_JY[-1], CS_QD[-1]]
    # print("xts_list PH,SSL,JY,QD",xts_list)
    # logger.info("xts PH,SSL,JY,QD:%s",xts_list)
    min_xts = min(xts_list)
    if min_xts > max_xts + 1:  # op中吃碰后向听数增加的情况，特别是打非平胡的牌型
        return {cards[-1]: 0}, min_xts
    type_list = []  # 需搜索的牌型
    for i in range(4):
        if xts_list[i] - 1 <= min_xts:
            type_list.append(i)
    # 多进程处理
    # jobs = [] #需执行的进程
    score_list = []
    time_start = time.time()
    time_list = []
    for i in type_list:
        if i == 0:
            search_PH = SearchTree_PH(hand=cards, suits=suits, combination_sets=CS_PH, king_card=king_card,
                                      fei_king=fei_king)
            score_list.append(search_PH.get_discard_score())
            # print("PH_INFO:state------------------------------------------------------------------------------------")
            # print(search_PH.discard_state)
            # print("PH_INFO:score--------------------------------------------------------------------------------")
            # print(search_PH.discard_score)
            # p=multiprocessing.Process(target=search_PH.get_discard_score())
        elif i == 1:
            score_list.append(SSL.get_discard_score())

            # p=multiprocessing.Process(target=SSL.get_discard_score())
        elif i == 2:
            score_list.append(JY.get_discard_score())

            # p=multiprocessing.Process(target=JY.get_discard_score())
        elif i == 3:
            score_list.append(QD.get_discard_score())

            # p = multiprocessing.Process(target=QD.get_discard_score())
        time_list.append(time.time() - time_start - sum(time_list))
        # p.start()
        # jobs.append(p)
    # for job in jobs:
    #     job.join()
    # print(time_list)
    # logger.info("time use%s",time_list)
    score_dict = {}
    for score in score_list:
        for key in score.keys():
            if key not in score_dict.keys():
                score_dict[key] = [0, 0]
                score_dict[key][0] = score[key][0] - float(value_t1(key)) / 10  # 用来区分相同权重的出牌
                score_dict[key][1] = score[key][1]
            else:
                score_dict[key][0] += score[key][0]
    if king_card in score_dict.keys():  # 飞宝的权重增加
        score_dict[king_card][0] *= 1.2

    # score = max(score_dict.values())[0]
    # print("get_score_dict:", score_dict)
    return score_dict, min_xts  # score_dict = { discard: [得分，牌型] }


def recommend_card_rf(cards=[], suits=[], king_card=None, discards=[], discards_op=[], fei_king=0, remain_num=136,
                      round=0, seat_id=0):
    """
    功能：推荐出牌接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类出牌决策
    :param cards: 手牌
    :param suits: 副露
    :param king_card: 宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param fei_king: 飞宝数
    :param remain_num: 剩余牌
    :return: outCard 推荐出牌
    """
    # logger.info("recommond card start...")
    # 更新全局变量
    global T_SELFMO, LEFT_NUM, t2tot3_dict, t1tot3_dict, TIME_START, RT1, RT2, RT3, ROUND
    ROUND = round
    MJ.KING = king_card
    TIME_START = time.time()
    LEFT_NUM, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
    LEFT_NUM[translate16_33(pre_king(king_card))] -= 1
    REMAIN_NUM = max(1, min(sum(LEFT_NUM), remain_num))

    if True:
        # if round <8:
        T_SELFMO = [float(i) / max(REMAIN_NUM, 1) for i in LEFT_NUM]
        # print T_SELFMO
        RT1 = []
        RT2 = []
        RT3 = []
    # else:
    #     # 当round>=8时，使用对手建模
    #     # cards, suits, king_card, fei_king, discards, discardsOp, discardsReal, round, seat_id, xts_round, M
    #     _, T_SELFMO, RT1, RT2, RT3 = DFM.DefendModel(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king,
    #                                                  discards=discards, discardsOp=discards_op, discardsReal=discards,
    #                                                  round=round, seat_id=seat_id, xts_round=DFM.xts_round,
    #                                                  M=100).getWTandRT()
    #     RT1 = []
    #     RT2 = []
    #     RT3 = []
    # t1tot2_dict = MJ.t1tot2_info(T_selfmo=T_SELFMO)

    t1tot3_dict = MJ.t1tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])
    t2tot3_dict = MJ.t2tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])

    score_dict, _ = get_score_dict(cards, suits, king_card, fei_king)
    recommend_card = {}
    paixing = 0
    if score_dict != {}:
        key = max(score_dict, key=lambda x: score_dict[x])
        recommend_card[key] = score_dict[key]
        paixing = recommend_card[key][-1]  # 牌型
        card = key
        # paixing = recommend_card[key]
    else:
        recommend_card = cards[-1]
        card = recommend_card

    #     logger.error("no card be recommonded,cards=%s,suits=%s,king_card=%s",cards, suits, king_card)
    # end = time.time()
    # if end - TIME_START > 3:
    #     logger.error("overtime %s,%s,%s,%s", end - TIME_START, cards, suits, king_card)
    # logger.info("recommend_card %s",recommend_card)
    # print("recommed_card",recommend_card)

    return paixing, card


def recommend_op(op_card, cards=[], suits=[], king_card=None, discards=[], discards_op=[], canchi=False,
                 self_turn=False, fei_king=0, isHu=False, round=0):
    """
    功能：动作决策接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类动作决策
    :param op_card: 操作牌
    :param cards: 手牌
    :param suits: 副露
    :param king_card: 宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param canchi: 吃牌权限
    :param self_turn: 是否是自己回合
    :param fei_king: 飞宝数
    :param isHu: 是否胡牌
    :return: [],isHu 动作组合牌，是否胡牌
    """
    if isHu:
        return [], True

    # 更新全局变量
    global T_SELFMO, LEFT_NUM, t2tot3_dict, t1tot3_dict, TIME_START
    TIME_START = time.time()
    LEFT_NUM, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
    LEFT_NUM[translate16_33(pre_king(king_card))] -= 1
    # if remain_num == 0:
    #     remain_num = 1
    REMAIN_NUM = sum(LEFT_NUM)
    if round > 100:
        T_SELFMO = []
        RT1 = []
        RT2 = []
        RT3 = []
    else:
        T_SELFMO = [float(i) / max(REMAIN_NUM, 1) for i in LEFT_NUM]
        RT1 = []
        RT2 = []
        RT3 = []

    t1tot3_dict = MJ.t1tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])
    t2tot3_dict = MJ.t2tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])

    # 计算操作前评估值
    cards_pre = copy.copy(cards)
    # cards_pre.append(-1) #加入一张0作为下次摸到的牌，并提升一定的概率a
    score_dict_pre, min_xts_pre = get_score_dict(cards_pre, suits, king_card, fei_king, padding=[-1])
    # xts_pre = min
    if score_dict_pre != {}:
        score_pre = max(score_dict_pre.values())[0]
    else:
        score_pre = 0

    # 计算操作后的评估值
    # 确定可选动作
    set_cards = list(set(cards))
    if self_turn:  # 自己回合，暗杠或补杠
        for card in set_cards:
            if cards.count(card) == 4:
                return [card, card, card, card], False  # 暗杠必杠
        for suit in suits:
            if suit.count(suit[0]) == 3 and suit[0] in cards:
                return suit + [suit[0]], False

    else:  # 其他玩家回合 #明杠，吃碰
        if cards.count(op_card) == 3:
            return [op_card, op_card, op_card, op_card], False

        op_sets = []
        if canchi:
            # 可操作的集合
            if op_card < 0x30:  # 字牌不能吃
                rm_sets = [[op_card - 2, op_card - 1], [op_card - 1, op_card + 1], [op_card + 1, op_card + 2]]
            else:
                rm_sets = []
            for op_set in rm_sets:
                if op_set[0] in cards and op_set[1] in cards:
                    op_sets.append(op_set)
            # 碰
            if cards.count(op_card) >= 2:
                op_sets.append([op_card, op_card])
        else:
            if cards.count(op_card) >= 2:
                op_sets.append([op_card, op_card])

        score_set = []
        for op_set in op_sets:
            cards_ = copy.copy(cards)
            cards_.remove(op_set[0])
            cards_.remove(op_set[1])

            suits_ = MJ.deepcopy(suits)
            suits_.append(sorted(op_set + [op_card]))
            score_dict, _ = get_score_dict(cards=cards_, suits=suits_, king_card=king_card, fei_king=fei_king,
                                           max_xts=min_xts_pre)
            # max_discard = max(score_dict, key=lambda x: score_dict[x])
            # print "score_dict",score_dict
            # print("score_dict", score_dict)
            if score_dict != {}:
                # print("now score", score_dict)
                max_v = max(score_dict.values())
                score = max_v if type(max_v) == int else max_v[0]
                # score = max(score_dict.values())[0]
                score_set.append(score)
        # if time.time() - TIME_START > 3:
        #     logger.warning("op time out %s", time.time() - TIME_START)
        if score_set == []:
            return [], False
        else:
            max_score = max(score_set)
            # print max_score, score_pre
            if max_score > score_pre:
                return sorted(op_sets[score_set.index(max_score)] + [op_card]), False

    return [], False


# 计算某一牌型的向听数(平胡，九幺，七对，十三烂) add by xwc
def get_xts(cards, suits, king_card, fei_king, padding=[]):
    # 寻找向听数在阈值内的牌型
    PH = PingHu(cards=cards, suits=suits, kingCard=king_card, fei_king=fei_king, padding=padding)
    JY = JiuYao(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    QD = Qidui(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    SSL = ShiSanLan(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    # 组合信息
    CS_PH = PH.pinghu_CS2()  # CS=[kz, sz, aa, ab, xts, leftcard]
    CS_JY = JY.yaojiu_CS()  # CS=[[九幺牌集合],[非九幺牌集合],[向听数]]
    CS_QD = QD.qidui_CS()  # CS=[[对子aa集合],[弃牌集合],向听数]
    CS_SSL = SSL.ssl_CS()  # CS=[十三烂萬组合，十三烂条组合，十三烂筒组合，十三烂字组合，非十三烂牌集合，xts]
    # 向听数
    xts_list = [CS_PH[0][-2], CS_JY[-1], CS_QD[-1], CS_SSL[0][-1]]
    # min_xts = min(xts_list)
    # return min_xts
    return xts_list


# 向听数 [平胡，九幺，七对，十三烂] add by xwc 2022.2.16
def get_xts_list(cards, suits, king_card):
    xts = get_xts(cards, suits, 0, 0)
    xts2 = get_xts(cards, suits, king_card, 0)
    l = [min(xts[i], xts2[i]) for i in range(4)]
    return l


# 平胡向听数 add by xwc 2022.2.22
def get_xts_ph(cards, suits, king_card):
    PH = PingHu(cards=cards, suits=suits, kingCard=king_card, fei_king=0, padding=[])
    CS_PH = PH.pinghu_CS2()  # CS=[kz, sz, aa, ab, xts, leftcard]
    xts = CS_PH[0][-2]
    return xts


# 九幺向听数 add by xwc 2022.2.22
def get_xts_91(cards, suits, king_card):
    JY = JiuYao(cards=cards, suits=suits, king_card=king_card, fei_king=0, padding=[])
    CS_JY = JY.yaojiu_CS()  # CS=[[九幺牌集合],[非九幺牌集合],[向听数]]
    xts = CS_JY[-1]
    return xts


# 七对向听数 add by xwc 2022.2.22
def get_xts_7(cards, suits, king_card):
    QD = Qidui(cards=cards, suits=suits, king_card=king_card, fei_king=0, padding=[])
    CS_QD = QD.qidui_CS()  # CS=[[对子aa集合],[弃牌集合],向听数]
    xts = CS_QD[-1]
    return xts


# 十三烂向听数 add by xwc 2022.2.22
def get_xts_13(cards, suits, king_card):
    SSL = ShiSanLan(cards=cards, suits=suits, king_card=king_card, fei_king=0, padding=[])
    CS_SSL = SSL.ssl_CS()  # CS=[十三烂萬组合，十三烂条组合，十三烂筒组合，十三烂字组合，非十三烂牌集合，xts]
    xts = CS_SSL[0][-1]
    return xts


# 返回十三烂的组合列表 add by xwc 2022.2.22
def get_CS_13(cards, suits, king_card):
    SSL = ShiSanLan(cards=cards, suits=suits, king_card=king_card, fei_king=0, padding=[])
    CS_SSL = SSL.ssl_CS()  # CS=[十三烂萬组合，十三烂条组合，十三烂筒组合，十三烂字组合，非十三烂牌集合，xts]
    return CS_SSL[0]


# 各个番型的差牌数 [清一色、门清、碰碰胡、宝吊、宝还原、单吊、七星、飞宝1234] add by xwc 2022.2.23
def get_xts_fan_list(handCards0, fulu0, king_card, fei_king):
    # 1.清一色
    QingYiSe = get_xts_QingYiSe(handCards0, fulu0)
    # 2.门清
    MenQing = get_xts_MenQing(handCards0, fulu0, king_card)
    # 3.碰碰胡
    PengPengHu = get_xts_PengPengHu(handCards0, fulu0, king_card)
    # 4.宝吊
    BaoDiao = get_xts_BaoDiao(handCards0, fulu0, king_card)
    # 5.宝还原
    BaoHuanYuan = get_xts_BaoHuanYuan(handCards0, fulu0, king_card)
    # 6.单吊
    DanDiao = get_xts_DanDiao(handCards0)
    # 7.七星
    QiXing = get_xts_QiXing(handCards0, fulu0, king_card)
    # 8.9.10.11飞宝1234
    feiking_num = fei_king
    return QingYiSe, MenQing, PengPengHu, BaoDiao, BaoHuanYuan, DanDiao, QiXing, feiking_num


# 差牌数（清一色）
def get_xts_QingYiSe(handCards0, fulu0):
    # 将所有的副露融合成手牌的形式
    all_fulu0_cards = []
    for fulu in fulu0:
        all_fulu0_cards.extend(fulu)
    w, ti, to, z = 0, 0, 0, 0
    for card in handCards0 + all_fulu0_cards:
        if card & 0xf0 == 0x00:
            w += 1
        if card & 0xf0 == 0x10:
            ti += 1
        if card & 0xf0 == 0x20:
            to += 1
        if card & 0xf0 == 0x30:
            z += 1
    l = []
    l.append(w)
    l.append(ti)
    l.append(to)
    l.append(z)
    return len(handCards0) + len(all_fulu0_cards) - max(l)


# 差牌数（门清）有副露，差牌数14；否则，差牌数为向听数
def get_xts_MenQing(handCards0, fulu0, king_card):
    if len(fulu0) > 0:
        return 14
    return min(get_xts_list(handCards0, fulu0, king_card))


# 差牌数（碰碰胡）副露中有顺子，差牌数14；否则碰碰胡向听数
def get_xts_PengPengHu(handCards0, fulu0, king_card):
    for fulu in fulu0:
        if fulu[0] != fulu[1]:
            return 14
    xts = 9 - len(fulu0) * 2
    S = set(handCards0)
    for i in S:
        if i == king_card:
            continue
        if handCards0.count(i) > 2:
            xts -= 2
        elif handCards0.count(i) == 2:
            xts -= 1
    king_num = handCards0.count(king_card)
    xts -= king_num
    if xts < 0:
        xts = 0
    return xts


# 差牌数（宝吊） 无宝牌，14
def get_xts_BaoDiao(handCards0, fulu0, king_card):
    # 没有宝牌不能宝吊
    king_num = handCards0.count(king_card)
    if king_num == 0:
        return 14
    xts_ph = 14
    # 平胡
    xts_king = get_xts_ph(handCards0, fulu0, king_card)
    xts_no_king = get_xts_ph(handCards0, fulu0, 0)
    if xts_king < xts_no_king:
        xts_ph = xts_king

    # 碰碰胡九幺
    xts_19 = 14
    sz = False
    l = [1, 9, 17, 25, 33, 41, 49, 50, 51, 52, 53, 54, 55]
    for fulu in fulu0:
        if fulu[0] != fulu[1] or fulu[0] not in l:
            sz = True
            break
    if not sz and (king_card not in l):
        num0, num1, num2 = 0, 0, 0
        num3 = len(fulu0)
        S = set(handCards0)
        for i in S:
            n = handCards0.count(i)
            if i not in l:
                num0 += n
            elif n == 1:
                num1 += n
            elif n == 2:
                num2 += n
            else:
                num3 += n
        xts_19 = 9 - num3 * 2 - num2 + num0

    # 七对
    xts_7 = 14
    qdXt = getQiDuiXt(handCards0, king_card)
    if (king_num % 2 == 0) and (king_num // 2 == qdXt):  # 宝全部还原
        xts_7 = qdXt - king_num
        if xts_7 < 0:
            xts_7 = 0
    xts = min(xts_ph, xts_19, xts_7)
    return xts


# 去除宝牌的七对向听数
def getQiDuiXt(handCards0, king_card):
    qdXt = 7
    S = set(handCards0)
    for i in S:
        if i != king_card and handCards0.count(i) >= 2:
            qdXt -= 1
    return qdXt


# 差牌数（宝还原）
def get_xts_BaoHuanYuan(handCards0, fulu0, king_card):
    king_num = handCards0.count(king_card)
    if king_num == 0:
        return 14

    # 平胡
    xts_ph = 14
    xts_king = get_xts_ph(handCards0, fulu0, king_card)
    xts_no_king = get_xts_ph(handCards0, fulu0, 0)
    if xts_king == xts_no_king:
        xts_ph = xts_king

    # 九幺
    xts_19 = 14
    if king_card in [1, 9, 17, 25, 33, 41, 49, 50, 51, 52, 53, 54, 55]:
        xts_19 = get_xts_91(handCards0, fulu0, king_card)

    # 七对
    xts_7 = 14
    if king_num > getQiDuiXt(handCards0, king_card):
        xts_7 = get_xts_7(handCards0, fulu0, king_card)

    # 十三烂必定宝还原
    xts13 = get_xts_13(handCards0, fulu0, king_card)
    # print(xts_ph, " ", xts_19, " ", xts_7, " ", xts13)
    return min(xts_ph, xts_19, xts_7, xts13)


# 差牌数（单吊）
def get_xts_DanDiao(handCards0):
    return len(handCards0)


# 差牌数（七星）
def get_xts_QiXing(handCards0, fulu0, king_card):
    # 将所有的副露融合成手牌的形式
    all_fulu0_cards = []
    for fulu in fulu0:
        all_fulu0_cards.extend(fulu)
    # 七星九幺
    tag = False
    for i in all_fulu0_cards:
        if i not in [1, 9, 17, 25, 33, 41, 49, 50, 51, 52, 53, 54, 55]:
            xts_91 = 14
            tag = True
            break
    if not tag:
        d, n, x, be, z, f, b = 0, 0, 0, 0, 0, 0, 0
        n_remain = 0
        for card in handCards0 + all_fulu0_cards:
            if card & 0xf0 == 0x30:
                if card & 0x0f == 1:
                    d = 1
                elif card & 0x0f == 2:
                    n = 1
                elif card & 0x0f == 3:
                    x = 1
                elif card & 0x0f == 4:
                    be = 1
                elif card & 0x0f == 5:
                    z = 1
                elif card & 0x0f == 6:
                    f = 1
                else:
                    b = 1
            elif card not in [1, 9, 17, 25, 33, 41]:
                n_remain += 1
        num = 7 - (d + n + x + be + z + f + b)
        xts_91 = max(num, n_remain)

    # 七星十三烂 CS=[十三烂万组合，十三烂条组合，十三烂筒组合，十三烂字组合，非十三烂牌集合，xts]
    CS = get_CS_13(handCards0, fulu0, king_card)
    if len(CS) == 3:
        xts_13 = 14
    else:
        xts_13 = max(7 - len(CS[3]), CS[-1])
    # print(xts_91, " ", xts_13)
    return min(xts_91, xts_13)


if __name__ == '__main__':
    cards = [1, 1, 9, 9, 17, 17, 25, 33, 41, 49, 50, 40, 41, 49]
    suits = []
    # cards = [1, 2, 3, 7, 7, 19, 20, 21, 22, 50, 50]
    # suits = [[2, 2, 2]]
    king_card = 49
    QingYiSe, MenQing, PengPengHu, BaoDiao, BaoHuanYuan, DanDiao, QiXing, feiking_num = get_xts_fan_list(cards, suits,
                                                                                                         king_card, 3)
    print(QiXing)
    # xts = get_xts_PengPengHu(cards, suits, king_card)
    # print(xts)
    # xts = get_xts(cards, suits, 0, 0)
    # xts2 = get_xts(cards, suits, king_card, 0)
    # l = [min(xts[i], xts2[i]) for i in range(4)]
    # print(l)
    # l = get_xts_QingYiSe(cards, suits)
    # print(l)

    # discards = [[50, 53, 34, 19, 37, 51, 34, 52, 18, 33, 49, 34, 35, 54, 9, 41, 50, 8, 17, 38, 9],
    #                                 [53, 50, 40, 24, 22, 53, 5, 35, 8, 36, 7, 33, 4, 7, 51, 25, 40, 2, 37, 55, 41, 1],
    #                                 [25, 4, 21, 52, 33, 23, 7, 39, 25, 34, 39, 6, 53, 55, 38, 55, 9, 1, 39, 38, 1, 21],
    #                                 [41, 54, 5, 6, 41, 36, 19, 37, 40, 55, 17, 54, 8, 50, 52, 20, 4, 19, 51, 23, 49]]
    # discards_op = [[[2, 3, 4]], [[35, 36, 37], [1, 1, 1]], [[35, 36, 37]], [[2, 3, 4]]]
    # fei_king = 0
    # remain_num = 20
    # round = 2
    # seat_id = 0
    #
    # result = recommend_op(23, cards, suits, king_card, discards, discards_op, fei_king=fei_king, canchi=True,round=round,isHu=True)
    # print("result", result)
