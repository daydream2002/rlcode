# -*- coding:utf-8 -*-
from sr_xt_ph import pinghu


# 计算可能的番型
# [清一色、门清、碰碰胡、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]

class FanList(object):
    def __init__(self, choosePaiXing=0, handcards=[], suits=[], jingCard=0, feiking_num=0, isHuJudge=False):
        '''
        番型检测功能
        :param choosePaiXing:  选择的胡牌类型 [0,1,2,3] -> 平胡， 九幺 ， 七对， 十三烂
        :param handcards:  手牌
        :param suits:  副露
        :param jingCard: 宝牌（精牌）
        :param feiking_num: 飞宝数
        :param isHuJudge: 是否胡牌时的番型检测
        '''
        self.choosePaiXing = choosePaiXing
        self.handcards = handcards
        self.suits = suits
        self.jingCard = jingCard
        self.all_suits_cards = self.__merge_suits()
        self.feiking_num = feiking_num
        self.jingCount = self.handcards.count(self.jingCard)
        self.isHuJudge = isHuJudge

    def __merge_suits(self):
        '''
        将所有的副露融合成手牌的形式 内部函数
        :return:
        '''
        all_suits_cards = []
        for suit in self.suits:
            all_suits_cards.extend(suit)
        return all_suits_cards

    def __getQiDuiXt(self):
        '''
        去除宝牌的七对向听数
        :return:
        '''
        qdXt = 7
        L = set(self.handcards)
        for i in L:
            if i != self.jingCard and self.handcards.count(i) >= 2:
                qdXt -= 1
        return qdXt

    def isPengPengHu(self):
        '''
        判断是否为碰碰胡，碰碰胡只存在平胡和九幺中
        :return:
        '''

        def getSzKzInSuits(suits):  # 判断碰碰胡
            kz = 0
            sz = 0
            for suit in suits:
                if suit[0] == suit[1]:
                    kz += 1
                else:
                    sz += 1
            return kz, sz

        # 九幺同样适用该函数
        pinghu_info = pinghu(self.handcards, self.suits, self.jingCard).get_xts_info()
        kz, sz = getSzKzInSuits(self.suits)
        if sz == 0:
            if self.isHuJudge:  # 判胡
                if (len(pinghu_info[0]) + kz) == 4 and len(pinghu_info[1]) == 0:  # 碰碰胡，并且已经胡牌
                    return True
            else:  # 判番型方向 只有手牌刻子数跟副露刻子数大于三时才会引导往碰碰胡番上走
                if (len(pinghu_info[0]) + kz) >= 3 and len(pinghu_info[1]) == 0:  # 可以往碰碰胡方向走
                    return True
        return False

    def isQingYiSe(self):
        '''
        判断是否为清一色
        :return:
        '''
        w = 0
        ti = 0
        to = 0
        z = 0
        for card in self.handcards + self.all_suits_cards:
            if card & 0xf0 == 0x00:
                w = 1
            if card & 0xf0 == 0x10:
                ti = 1
            if card & 0xf0 == 0x20:
                to = 1
            if card & 0xf0 == 0x30:
                z = 1
        if w + ti + to + z <= 1:
            return True
        return False

    def isMenQing(self):
        return len(self.suits) == 0

    def isBaoDiao(self):
        if self.jingCount and self.choosePaiXing != 3:  # 有精牌或者选择牌型不为13烂
            if self.choosePaiXing == 0:
                return pinghu(self.handcards, self.suits, self.jingCard).get_xts() < \
                       pinghu(self.handcards, self.suits, 0).get_xts()
            elif self.choosePaiXing == 1:  # 九幺
                return self.isPengPengHu() and self.jingCard not in \
                       [1, 9, 17, 25, 33, 41, 49, 50, 51, 52, 53, 54, 55]  # 如果走碰碰胡方向，可以是宝吊
            else:  # 七对默认宝吊
                if (self.jingCount % 2 == 0) and (self.jingCount // 2 == self.__getQiDuiXt()):  # 宝全部还原
                    return True
                else:
                    return False
        else:
            return False

    def isBaoHuanYuan(self):
        if self.jingCount:
            if self.choosePaiXing == 0:  # 平胡
                return pinghu(self.handcards, self.suits, self.jingCard).get_xts() == \
                       pinghu(self.handcards, self.suits, 0).get_xts()
            elif self.choosePaiXing == 1:  # 九幺
                if self.jingCard in [1, 9, 17, 25, 33, 41, 49, 50, 51, 52, 53, 54, 55]:
                    return True
                else:
                    return False
            elif self.choosePaiXing == 2:  # 七对
                if self.jingCount > self.__getQiDuiXt():  #
                    return True
                else:
                    return False
            else:  # 十三烂必定宝还原
                return True

        else:
            return False

    def isDanDiao(self):
        return len(self.suits) == 4

    def isQingXing(self):
        if self.choosePaiXing in [1, 3]:  # 九幺 十三烂e
            d, n, x, be, z, f, b = 0, 0, 0, 0, 0, 0, 0  # 东南西北中发白
            for card in self.handcards + self.all_suits_cards:
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
            return d + n + x + be + z + f + b == 7
        else:
            return False

    def getFanList(self):
        '''
        return the may fans base on the choosePaiXing
        :return:
        '''
        # [清一色、门清、碰碰胡、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
        # choosePaiXing[平胡  九幺　七对 十三烂]
        fanList = [0] * 11

        if self.isQingYiSe():  fanList[0] = 1
        if self.isMenQing(): fanList[1] = 1
        if self.isPengPengHu(): fanList[2] = 1

        if self.isBaoDiao(): fanList[3] = 1  # baodiao

        if self.isBaoHuanYuan(): fanList[4] = 1
        if self.isDanDiao(): fanList[5] = 1

        if self.isQingXing(): fanList[6] = 1

        if self.feiking_num > 0:
            for i in range(self.feiking_num):
                fanList[7 + i] = 1

        return fanList

# if __name__ == '__main__':
#     # test
#     # fan_list= FanList(1, [1, 2, 3, 6, 35],[[36, 36, 36], [29, 29, 29], [9, 9, 9]], 35, 2, isHuJudge=False).getFanList()
#     fan_list= FanList(0, [6, 35], [[1, 2, ],[36, 36, 36], [29, 29, 29], [9, 9, 9]], 35, 2, isHuJudge=False).getFanList()
#     print(fan_list)
