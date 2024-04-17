# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/10 14:54
@Auth ： Joisen
@File ：dynamic_handcard.py
"""
import copy
import random
import time


class DynamicHandCard:
    def __init__(self):
        self.card_library = {}
        self.hz_card_library = {}
        self.repeat_nums = 10
        self.reset()
        # sorted(self.card_library, key=lambda x: x[0])

    def reset(self):
        self.card_library = {}
        self.hz_card_library = {}
        for i in range(1, 10):
            self.card_library[i] = 4
            self.card_library[i + 16] = 4
            self.card_library[i + 32] = 4
        self.hz_card_library[49] = 6

    def randomKZ(self, color=None, hasCardList=None):
        rightCards = []
        for key, cardNums in self.card_library.items():  # card / 16 == 0, 1, 2, 3(万，条，筒，中)
            if cardNums < 3:
                continue
            if color is not None:  # 有颜色
                if color != key // 16:  # 当前牌不是该颜色
                    continue
                else:  # 当前牌是该颜色
                    if hasCardList is not None:  # 所需范围不为空
                        if key % 16 in hasCardList:  # 当前牌属于所需范围
                            rightCards.append(key)
                    else:  # 所需范围为空
                        rightCards.append(key)
            else:  # 没有颜色
                if hasCardList is not None:  # 所需范围不为空
                    if key % 16 in hasCardList:  # 当前牌属于所需范围
                        rightCards.append(key)
                else:  # 所需范围为空
                    rightCards.append(key)

        random.shuffle(rightCards)
        if rightCards.__len__() == 0:  # 如果没有任何一张牌
            return None
        self.card_library[rightCards[0]] -= 3
        return [rightCards[0], rightCards[0], rightCards[0]]

    def randomSZ(self, color=None, hasCardList=None):
        sz_list = []
        for key, cardNum in self.card_library.items():
            if cardNum >= 1 and key + 1 in self.card_library.keys() and key + 2 in self.card_library.keys():  # 如果key为第一张牌的顺子的所有牌都存在
                if self.card_library[key + 1] >= 1 and self.card_library[key + 2] >= 1:  # 且都有余牌
                    sz_list.append([key, key + 1, key + 2])

            if cardNum >= 1 and key - 1 in self.card_library.keys() and key + 1 in self.card_library.keys():  # 如果key为第一张牌的顺子的所有牌都存在
                if self.card_library[key - 1] >= 1 and self.card_library[key + 1] >= 1:  # 且都有余牌
                    sz_list.append([key - 1, key, key + 1])

            if cardNum >= 1 and key - 1 in self.card_library.keys() and key - 2 in self.card_library.keys():  # 如果key为第一张牌的顺子的所有牌都存在
                if self.card_library[key - 1] >= 1 and self.card_library[key - 2] >= 1:  # 且都有余牌
                    sz_list.append([key - 2, key - 1, key])
        # print(sz_list)
        sz_list_copy = copy.copy(sz_list)
        for sz in sz_list_copy:
            if color is not None:  # 定义了花色
                if sz[0] // 16 == color:  # 属于所定花色的牌
                    if hasCardList is not None:  # 定义了牌的范围
                        if sz[0] % 16 not in hasCardList or sz[1] % 16 not in hasCardList or sz[
                            2] % 16 not in hasCardList:  # 该顺子的牌都属于所定范围
                            # 如果不符合要求则移除
                            sz_list.remove(sz)
                else:
                    # 如果不符合要求则移除
                    sz_list.remove(sz)
            else:
                if hasCardList is not None:  # 定义了牌的范围
                    if sz[0] % 16 not in hasCardList or sz[1] % 16 not in hasCardList or sz[
                        2] % 16 not in hasCardList:  # 该顺子的牌都属于所定范围
                        # 如果不符合要求则移除
                        sz_list.remove(sz)

        random.shuffle(sz_list)
        if sz_list.__len__() == 0:
            return None
        sz = sz_list[0]
        self.card_library[sz[0]] -= 1
        self.card_library[sz[2]] -= 1
        self.card_library[sz[1]] -= 1

        return sz

    def randomGZ(self, color=None, hasCardList=None):
        rightCards = []
        for key, cardNums in self.card_library.items():  # card / 16 == 0, 1, 2, 3(万，条，筒，中)
            if cardNums < 4:
                continue
            if color is not None:  # 有颜色
                if color != key // 16:  # 当前牌不是该颜色
                    continue
                else:  # 当前牌是该颜色
                    if hasCardList is not None:  # 所需范围不为空
                        if key % 16 in hasCardList:  # 当前牌属于所需范围
                            rightCards.append(key)
                    else:  # 所需范围为空
                        rightCards.append(key)
            else:  # 没有颜色
                if hasCardList is not None:  # 所需范围不为空
                    if key % 16 in hasCardList:  # 当前牌属于所需范围
                        rightCards.append(key)
                else:  # 所需范围为空
                    rightCards.append(key)

        random.shuffle(rightCards)
        if rightCards.__len__() == 0:  # 如果没有任何一张牌
            return None
        self.card_library[rightCards[0]] -= 4
        return [rightCards[0], rightCards[0], rightCards[0], rightCards[0]]

    def randomDZ(self, color=None, hasCardList=None):
        rightCards = []
        for key, cardNums in self.card_library.items():  # card / 16 == 0, 1, 2, 3(万，条，筒，中)
            if cardNums < 2:
                continue
            if color is not None:  # 有颜色
                if color != key // 16:  # 当前牌不是该颜色
                    continue
                else:  # 当前牌是该颜色
                    if hasCardList is not None:  # 所需范围不为空
                        if key % 16 in hasCardList:  # 当前牌属于所需范围
                            rightCards.append(key)
                    else:  # 所需范围为空
                        rightCards.append(key)
            else:  # 没有颜色
                if hasCardList is not None:  # 所需范围不为空
                    if key % 16 in hasCardList:  # 当前牌属于所需范围
                        rightCards.append(key)
                else:  # 所需范围为空
                    rightCards.append(key)

        random.shuffle(rightCards)
        if rightCards.__len__() == 0:  # 如果没有任何一张牌
            return None
        self.card_library[rightCards[0]] -= 2
        return [rightCards[0], rightCards[0]]

    def replace_card(self, handCard, num=3, hz_num=0):
        random.shuffle(handCard)
        rm_cards = handCard[:num]
        handCard = handCard[num:]
        self.back_card(rm_cards)  # 归还牌

        min_hz = min(hz_num, self.hz_card_library[49])  # 获取应该替换的最小红中数
        handCard += [49] * min_hz  # 将红中给手牌
        self.hz_card_library[49] -= min_hz  # 更新牌库中的红中数

        cardLibrary = []
        for key, val in self.card_library.items():  # 将牌库转为列表
            cardLibrary += [key] * val

        random.shuffle(cardLibrary)
        add_cards = cardLibrary[:num - min_hz]  # 从牌库中取出需要替换的牌
        handCard += add_cards

        for i in add_cards:  # 牌库中移除掉替换的牌
            if i == 49:
                self.hz_card_library[49] -= 1
            else:
                self.card_library[i] -= 1

        return sorted(handCard)

    def get_level_one(self, is_leader=False):

        kz_num = random.randint(0, 4)
        # print(kz_num)
        sz_num = 4 - kz_num
        # print(sz_num)
        handCard = []
        for i in range(kz_num):
            kz = self.randomKZ()
            if kz is None:
                break
            else:
                handCard += kz

        for i in range(sz_num):
            sz = self.randomSZ()
            if sz is None:
                sz = self.randomKZ()
            handCard += sz

        dz = self.randomDZ()
        if dz is not None:
            handCard += dz

        cardNum = handCard.__len__()
        if cardNum != 14:  # 生成失败
            print("生成失败！！！ 请重新调用！！！")
            return None
        if not is_leader:
            random.shuffle(handCard)
            self.back_card(handCard[13:])
            handCard = handCard[0:13]

        return self.replace_card(handCard, 5, 0)

    def level_two_qidui(self):
        qiDui_cards = []
        for i in range(7):
            cardList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            rd_list = [3, 4, 5, 6, 7]
            rm_idx = random.randint(0, 4)
            cardList.remove(rd_list[rm_idx])
            qiDui_cards += self.randomDZ(hasCardList=cardList)
        return qiDui_cards

    def level_two_other(self):
        ret_cards = []
        cardList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for i in range(2):
            is_greater_5 = random.randint(0, 1)
            kz_num = random.randint(0, 2)
            for j in range(kz_num):
                if is_greater_5 == 1:
                    kz = self.randomKZ(hasCardList=cardList[5:])
                    ret_cards += [] if kz is None else kz

                else:
                    kz = self.randomKZ(hasCardList=cardList[0:4])
                    ret_cards += [] if kz is None else kz
            for k in range(2 - kz_num):
                if is_greater_5 == 1:
                    sz = self.randomSZ(hasCardList=cardList[5:])
                    ret_cards += [] if sz is None else sz
                else:
                    sz = self.randomSZ(hasCardList=cardList[0:4])
                    ret_cards += [] if sz is None else sz
        # print(ret_cards)
        dz = self.randomDZ()
        ret_cards += [] if dz is None else dz
        if len(ret_cards) != 14:
            for i in ret_cards:
                self.card_library[i] += 1
            return None
        return ret_cards

    def get_level_two(self, is_leader=False):
        rd_type = random.randint(0, 1)
        ret_card = []
        if rd_type == 0:  # 七对
            qd = self.level_two_qidui()
            ret_card = [] if qd is None else qd
        else:
            other = self.level_two_other()
            if other is None:
                qd = self.level_two_qidui()
                ret_card = [] if qd is None else qd
            else:
                ret_card = other

        cardNum = ret_card.__len__()
        if cardNum != 14:  # 生成失败
            print("生成失败！！！ 请重新调用！！！")
            return None
        if not is_leader:
            random.shuffle(ret_card)
            self.back_card(ret_card[13:])
            ret_card = ret_card[0:13]

        return self.replace_card(ret_card, 2, 0)

    def level_three_jgg(self):
        jgg = []
        gz_nums = random.randint(1, 2)

        for i in range(gz_nums):
            gz = self.randomGZ()
            jgg += [] if gz is None else gz

        kz_nums = 4 - jgg.__len__() // 4
        for i in range(kz_nums):
            kz = self.randomKZ()
            jgg += [] if kz is None else kz

        if jgg.__len__() == 13:
            card_list = [item[0] for item in self.card_library.items()]
            random.shuffle(card_list)
            for item in card_list:
                if self.card_library[item] >= 1:
                    self.card_library[item] -= 1
                    jgg.append(item)
                    break

        if jgg.__len__() == 12:
            dz = self.randomDZ()
            jgg += [] if dz is None else dz

        if jgg.__len__() != 14:
            for i in jgg:
                self.card_library[i] += 1
            return None
        return jgg

    def level_three_bai(self):

        bwd = [5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
        bzj = []
        bcl = []

        for item in bwd:
            bzj.append(item + 16)
            bcl.append(item + 32)
        res = [bwd, bzj, bcl]
        for item in bwd:
            if self.card_library[item] < bwd.count(item):
                res.remove(bwd)
                break

        for item in bzj:
            if self.card_library[item] < bzj.count(item):
                res.remove(bzj)
                break

        for item in bcl:
            if self.card_library[item] < bcl.count(item):
                res.remove(bcl)
                break

        random.shuffle(res)
        if res.__len__() == 0:
            return None
        res = res[0]
        for item in res:
            self.card_library[item] -= 1
        return res

    def level_three_dzx(self):
        type = [0, 1, 2]
        random.shuffle(type)
        card = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for it in type:
            res = []
            kz_nums = random.randint(0, 4)
            sz_nums = 4 - kz_nums
            for i in range(kz_nums):
                kz = self.randomKZ(hasCardList=card[it * 3: (it + 1) * 3])
                res += [] if kz is None else kz

            for i in range(sz_nums):
                sz = self.randomSZ(hasCardList=card[it * 3: (it + 1) * 3])
                res += [] if sz is None else sz

            dz = self.randomDZ()
            res += [] if dz is None else dz
            if res.__len__() != 14:
                for i in res:
                    self.card_library[i] += 1
                continue
            return res
        return None

    def back_card(self, cardList):
        for i in cardList:
            if i == 49:
                self.hz_card_library[i] += 1
            else:
                self.card_library[i] += 1

    def get_level_three(self, is_leader=False):
        ret_card = []
        bai = self.level_three_bai()
        jgg = self.level_three_jgg()
        dzx = self.level_three_dzx()
        if jgg is not None:
            ret_card.append(jgg)

        if bai is not None:
            ret_card.append(bai)

        if dzx is not None:
            ret_card.append(dzx)

        random.shuffle(ret_card)
        # print(ret_card)
        if ret_card.__len__() == 0:
            print("生成失败！！！ 请重新调用！！！")
            return None
        cardNum = ret_card[0]
        if not is_leader:
            random.shuffle(cardNum)
            self.back_card(cardNum[13:])
            cardNum = cardNum[0:13]

        for it in ret_card[1:]:
            self.back_card(it)
        return self.replace_card(cardNum, 2, 0)

    def level_four_qys(self):  # 生成  绿一色  天圆地方  万里长城、
        # 三个排序都进行生成，随机取出其中的一个返回
        green_range = [2, 3, 4, 6, 8]
        ton_range = [1, 2, 4, 5, 8, 9]
        wan_range = [1, 3, 5, 7, 9]
        green_cards = []
        green_kz_num = random.randint(0, 4)
        for i in range(green_kz_num):
            tmp = self.randomKZ(color=1, hasCardList=green_range)
            green_cards += [] if tmp is None else tmp
        green_sz_num = 4 - green_cards.__len__() // 3
        for i in range(green_sz_num):
            tmp = self.randomSZ(color=1, hasCardList=green_range)
            green_cards += [] if tmp is None else tmp
        dz = self.randomDZ(color=1, hasCardList=green_range)
        green_cards += [] if dz is None else dz
        if green_cards.__len__() != 14:
            self.back_card(green_cards)  # 返还牌
        else:
            return green_cards

        wan_cards = []
        wan_kz_num = 4
        for i in range(wan_kz_num):
            tmp = self.randomKZ(color=0, hasCardList=wan_range)
            wan_cards += [] if tmp is None else tmp
        dz = self.randomDZ(color=0, hasCardList=wan_range)
        wan_cards += [] if dz is None else dz
        if wan_cards.__len__() != 14:
            self.back_card(wan_cards)  # 返还牌
        else:
            return wan_cards

        ton_cards = []
        ton_kz_num = 4
        for i in range(ton_kz_num):
            tmp = self.randomKZ(color=2, hasCardList=ton_range)
            ton_cards += [] if tmp is None else tmp
        dz = self.randomDZ(color=2, hasCardList=ton_range)
        ton_cards += [] if dz is None else dz
        if ton_cards.__len__() != 14:
            self.back_card(ton_cards)  # 返还牌
        else:
            return ton_cards
        return None

    def level_four_other(self):
        # 生成1个杠子 + 3个刻子 + 1个红中 -> 中吊中
        dzd = []
        gang = self.randomGZ()
        dzd += [] if gang is None else gang
        for i in range(3):
            kz = self.randomKZ()
            dzd += [] if kz is None else kz
        if self.hz_card_library[49] >= 1:
            dzd.append(49)
            self.hz_card_library[49] -= 1
        if dzd.__len__() != 14:
            self.back_card(dzd)
        else:
            return dzd
        sejc = []
        for i in range(3):
            gan = self.randomGZ()
            sejc += [] if gan is None else gan
        dz = self.randomDZ()
        sejc += [] if dz is None else dz
        if sejc.__len__() != 14:
            self.back_card(sejc)
        else:
            return sejc

        return None

    def get_level_four(self, is_leader=False):
        qys = self.level_four_qys()
        if qys is not None:
            if not is_leader:
                random.shuffle(qys)
                self.back_card(qys[13:])
                qys = qys[0:13]
            return self.replace_card(qys, 2, 0)

        other = self.level_four_other()
        if other is not None:
            if not is_leader:
                random.shuffle(other)
                self.back_card(other[13:])
                other = other[0:13]
            return self.replace_card(other, 2, 0)

        print("生成失败！！！ 请重新调用！！！")
        return None

    def level_five_955819(self):
        card_59 = []
        gz = self.randomGZ(hasCardList=[5, 9])
        card_59 += [] if gz is None else gz
        kz_nums = 3
        if card_59.__len__() != 4:
            kz_nums = 4

        for i in range(kz_nums):
            kz = self.randomKZ(hasCardList=[5, 9])
            card_59 += [] if kz is None else kz

        if card_59.__len__() == 13:
            card_59.append(49)
            self.hz_card_library[49] -= 1
        elif card_59.__len__() == 12:
            dz = self.randomDZ(hasCardList=[5, 9])
            card_59 += [] if dz is None else dz

        if card_59.__len__() != 14:
            self.back_card(card_59)
        else:
            return card_59

        card_58 = []
        gz = self.randomGZ(hasCardList=[5, 8])
        card_58 += [] if gz is None else gz
        kz_nums = 3
        if card_58.__len__() != 4:
            kz_nums = 4

        for i in range(kz_nums):
            kz = self.randomKZ(hasCardList=[5, 8])
            card_58 += [] if kz is None else kz

        if card_58.__len__() == 13:
            card_58.append(49)
            self.hz_card_library[49] -= 1
        elif card_58.__len__() == 12:
            dz = self.randomDZ(hasCardList=[5, 8])
            card_58 += [] if dz is None else dz

        if card_58.__len__() != 14:
            self.back_card(card_58)
        else:
            return card_58

        card_19 = []
        gz = self.randomGZ(hasCardList=[1, 9])
        card_19 += [] if gz is None else gz
        kz_nums = 3
        if card_19.__len__() != 4:
            kz_nums = 4

        for i in range(kz_nums):
            kz = self.randomKZ(hasCardList=[1, 9])
            card_19 += [] if kz is None else kz

        if card_19.__len__() == 13:
            card_19.append(49)
            self.hz_card_library[49] -= 1
        elif card_19.__len__() == 12:
            dz = self.randomDZ(hasCardList=[1, 9])
            card_19 += [] if dz is None else dz

        if card_19.__len__() != 14:
            self.back_card(card_19)
            return None
        else:
            return card_19

    def remove_cardList(self, card_list):
        for item in card_list:
            self.card_library[item] -= 1

    def level_five_lqd_item(self, color):
        flag = True
        for i in range(1 + color * 16, 8 + color * 16):
            if self.card_library[i] < 2:
                flag = False
                break

        if flag:
            tmp = [1, 2, 3, 4, 5, 6, 7] * 2
            tmp = [i + color * 16 for i in tmp]
            self.remove_cardList(tmp)
            return tmp

        flag = True
        for i in range(2 + color * 16, 9 + color * 16):
            if self.card_library[i] < 2:
                flag = False
                break

        if flag:
            tmp = [2, 3, 4, 5, 6, 7, 8] * 2
            tmp = [i + color * 16 for i in tmp]
            self.remove_cardList(tmp)
            return tmp

        flag = True
        for i in range(3 + color * 16, 10 + color * 16):
            if self.card_library[i] < 2:
                flag = False
                break

        if flag:
            tmp = [3, 4, 5, 6, 7, 8, 9] * 2
            tmp = [i + color * 16 for i in tmp]
            self.remove_cardList(tmp)
            return tmp

        return None

    def level_five_lqd(self):  # 龙七对
        tmp = self.level_five_lqd_item(color=0)
        if tmp is not None:
            return tmp

        tmp = self.level_five_lqd_item(color=1)
        if tmp is not None:
            return tmp

        tmp = self.level_five_lqd_item(color=2)
        if tmp is not None:
            return tmp

        return None

    def get_level_five(self, is_leader=False):
        sp = self.level_five_955819()
        if sp is not None:
            if not is_leader:
                random.shuffle(sp)
                self.back_card(sp[13:])
                sp = sp[0:13]
            return self.replace_card(sp, 2, 0)

        other = self.level_five_lqd()
        if other is not None:
            if not is_leader:
                random.shuffle(other)
                self.back_card(other[13:])
                other = other[0:13]
            return self.replace_card(other, 2, 0)

        print("生成失败！！！ 请重新调用！！！")
        return None

    def eval(self, handCards=[]):
        ''' 根据已分配
        :param handCards:
        :return:
        '''
        for hand in handCards:
            self.back_card(hand)
        for k, v in self.card_library.items():
            if self.card_library[k] != 4 or (k == 49 and self.hz_card_library[k] != 6):
                print("error  11111")
                return False
        return True

    def my_sort(self, level=[]):
        ''' 自定义排序
        :param level: [(等级，dealer) ... ] 按照等级排序
        :return: 需要返回排好序的level 以及排好序后的level[i] 在原列表中的索引
        '''
        sorted_lst = sorted(enumerate(level), key=lambda x: -x[1][
            0])  # enumerate()函数将level列表中的元素与其在列表中的索引进行关联，并返回一个可迭代对象。enumerate(level)的结果类似于[(0, level[0]), (1, level[1]), ...]。
        return [item[1] for item in sorted_lst], [item[0] for item in sorted_lst]

    def changeSeat(self, handCards=[], index=[]):
        ''' 根据index列表，将handCards进行位置变换， 即将handCards[i] 移动到 handCards[index[i]]
        :param handCards: 四个玩家的手牌，是按照等级从大到小排序的
        :param index: 四个玩家手牌的原始位置
        :return:
        '''
        length = len(handCards)
        ret_cards = [[] for i in range(length)]
        # print(index)
        for i in range(length):
            ret_cards[index[i]] = handCards[i]
        return ret_cards

    def getHandCards(self, level=[]):
        # level = sorted(level, key=lambda x: -x[0])  # 将列表中的元组按照第一个元素（等级）进行排序
        level, level_index = self.my_sort(level)
        # print(level)
        ret_cards = []
        flag = False

        for l, is_leader in level:
            if l == 1:
                level_1 = self.get_level_one(is_leader=is_leader)
                if level_1 is None:
                    flag = True
                    break
                ret_cards.append(level_1)

            if l == 2:
                level_1 = self.get_level_two(is_leader=is_leader)
                if level_1 is None:
                    flag = True
                    break
                ret_cards.append(level_1)

            if l == 3:
                level_1 = self.get_level_three(is_leader=is_leader)
                if level_1 is None:
                    flag = True
                    break
                ret_cards.append(level_1)

            if l == 4:
                level_1 = self.get_level_four(is_leader=is_leader)
                if level_1 is None:
                    flag = True
                    break
                ret_cards.append(level_1)

            if l == 5:
                level_1 = self.get_level_five(is_leader=is_leader)
                if level_1 is None:
                    flag = True
                    break
                ret_cards.append(level_1)
        if flag:
            print("error")
            for item in ret_cards:
                self.back_card(item)
            return None
        # return ret_cards
        return self.changeSeat(handCards=ret_cards, index=level_index)

    def generate_handCards(self, level):
        if level.__len__() != 4:
            print("参数错误！")
            return None

        for i, _ in level:
            if i > 5 or i < 1:
                print("参数错误！")
                return None

        for i in range(self.repeat_nums):
            ret_cards = self.getHandCards(level)
            if ret_cards.__len__() == 4:
                return ret_cards  # 返回的是经过等级排序后的玩家手牌列表，显然不是需要的类型
        print("system error")
        return None

    def gen_lastHands(self, hands=[], level=[]):
        '''
        :param hands: 已经生成的几副手牌 数目不确定 需要验证
        :param level: 列表 表示剩余3个玩家的等级和是否是庄家，每个玩家是一个元组（level, isDealer）
        :return: 返回
        '''
        length = len(level)
        if len(hands) != 4 - length:
            print("参数错误！！！")
            return None
        for i, _ in level:
            if i > 5 or i < 1:
                print("参数错误！！！")
                return None
        for hand in hands:  # 从牌库中移除掉这些牌
            for card in hand:
                if card == 49:
                    self.hz_card_library[49] -= 1
                else:
                    self.card_library[card] -= 1
        # 接下来生成剩余的手牌
        for i in range(self.repeat_nums):
            ret_cards = self.getHandCards(level)
            if ret_cards.__len__() == length:
                return ret_cards  # 返回的是经过等级排序后的玩家手牌列表，显然不是需要的类型
        print("system error")
        return None




if __name__ == '__main__':
    # dh = DynamicHandCard()
    # card = dh.randomKZ(1, [5, 6, 7, 8, 9])
    # print(dh.card_library)
    # print(card)
    #
    # sz = dh.randomSZ()
    # print(dh.card_library)
    # print(sz)
    #
    # gz = dh.randomGZ(1, [5, 6, 7, 8, 9])
    # print(dh.card_library)
    # print(gz)
    #
    # dz = dh.randomDZ(1, [5, 6, 7, 8, 9])
    # print(dh.card_library)
    # print(dz)

    # print(dh.get_level_one(True))
    # print(dh.card_library)
    # print(dh.level_two_qidui())
    # print("==============")

    # ============================================== 测试 ===============================================
    # for i in range(1000):
    #     dh = DynamicHandCard()
    #     alHands = []
    #     hands = dh.get_level_five(is_leader=False)
    #     alHands.append(hands)
    #     print(dh.eval(alHands))

    # ===========================================测试生成4手牌==================================================

    # cnt = 0
    # w_cnt = 0
    # dh = DynamicHandCard()
    # while True:
    #     dh.reset()
    #     player = []
    #     tmp = 0
    #     for i in range(4):
    #         t = random.randint(1, 5)
    #         tmp += t
    #         player.append(t)
    #     if tmp > 15:
    #         continue
    #
    #     leader = random.randint(0, 3)
    #     player_list = []
    #     for i in range(4):
    #         if i == leader:
    #             player_list.append((player[i], 1))
    #         else:
    #             player_list.append((player[i], 0))
    #
    #     print(player_list)
    #     # exit(1)
    #     res = dh.generate_handCards(level=player_list)
    #     if not dh.eval(res):
    #         print("错了")
    #         exit(1)
    #     # print(dh.eval(res))
    #
    #     # print(player_list)
    #     print(res)
    #     cnt += 1
    #     if res is None:
    #         w_cnt += 1
    #
    #     if cnt == 1000:
    #         print(f"出错率{w_cnt / cnt:0.4f}")
    #         break

    # ===========================================测试生成剩余几副手牌(牌库生成失败备选方案)==================================================

    cnt = 0
    w_cnt = 0
    dh = DynamicHandCard()
    # while True:
    #     dh.reset()
    #     player = []  # 定义玩家的等级
    #     tmp = 0
    #     hasGen_nums = random.randint(1, 3)  # 生成1-3个玩家的手牌
    #     need_nums = 4 - hasGen_nums
    #     leader = random.randint(0, 3)  # 庄家位置
    #     for i in range(need_nums):  # 需要生成的玩家的等级
    #         t = random.randint(1, 5)
    #         tmp += t
    #         player.append(t)
    #
    #
    #     player_list = []
    #     for i in range(4):
    #         if i == leader:
    #             player_list.append((player[i], 1))
    #         else:
    #             player_list.append((player[i], 0))
    #
    #     print(player_list)
    #     # exit(1)
    #     res = dh.generate_handCards(level=player_list)
    #     if not dh.eval(res):
    #         print("错了")
    #         exit(1)
    #     # print(dh.eval(res))
    #
    #     # print(player_list)
    #     print(res)
    #     cnt += 1
    #     if res is None:
    #         w_cnt += 1
    #
    #     if cnt == 1000:
    #         print(f"出错率{w_cnt / cnt:0.4f}")
    #         break

    hands = [[2, 7, 8, 8, 9, 9, 17, 17, 23, 23, 33, 33, 34], [9, 18, 18, 18, 18, 19, 19, 22, 22, 22, 24, 24, 24, 36]]
    level = [(3, 0), (4, 0)]
    res = dh.gen_lastHands(hands=hands, level=level)
    print(res)

    # ===========================================测试my_sort()函数==================================================

    # dh = DynamicHandCard()
    # j = 0
    # while True:
    #     player = []
    #     tmp = 0
    #     for i in range(4):
    #         t = random.randint(1, 5)
    #         tmp += t
    #         player.append(t)
    #     if tmp > 15:
    #         continue
    #
    #     leader = random.randint(0, 3)
    #     player_list = []
    #     for i in range(4):
    #         if i == leader:
    #             player_list.append((player[i], 1))
    #         else:
    #             player_list.append((player[i], 0))
    #     print(player_list)
    #     player_list, player_index = dh.my_sort(player_list)
    #     print(player_list)
    #     print(player_index)
    #     print("======================================")
    #     j += 1
    #     print(j)
    #     if j == 1000:
    #         break

# print(dh.card_library)
# print(dh.level_three_jgg())
# print(dh.level_three_dzx())
