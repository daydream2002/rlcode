#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : mah_player.py
# @Description: 玩家信息，包括推荐出牌、推荐动作、换三张推荐、定缺推荐、执行碰杠动作等
import random
import torch

from interface.interface_v1.recommand import recommand_card as recommand_card1  # 智一
from sbln_learning.torch_v.agent.PPO_resnet50 import policyNet
from sbln_learning.torch_v.policy_model.resnet_50 import ResNet50
from mah_tool import url_recommend
from stable_baselines.ppo2 import PPO2
from mah_tool import tool2
from sbln_learning.torch_v.replay_buffer import Transition


class Player_RL(object):
    '''
        玩家信息，包括推荐出牌、推荐动作、换三张推荐、定缺推荐、执行碰杠动作等
        className:Player_RL
        fileName:mah_player.py
    '''
    def __init__(self, name, seat_id, model_name, model_path=None, brain=None):
        """
        构造器
        @param name: 玩家名称
        @param seat_id: 玩家座位号
        @param model_name: 模型名称
        @param brain: DQN模型
        """
        self.model = model_name
        self.catch_card = 0
        self.seat_id = seat_id
        self.name = name  # 玩家名称
        self.handcards = []  # 玩家手牌
        self.fulu = []  # 玩家副露
        self.allow_op = []  # 允许操作
        self.allow_handcards = self.handcards  # 允许操作的手牌
        self.compe_op = -1  # 竞争性操作(左吃，中吃，右吃，碰，明杠，胡)
        self.model_path = model_path
        self.brain1 = brain  # BrainDQN(34) #先不改
        self.isInitstate = True  # 用于为DQN获取初始状态的标志
        self.action = None  # 存储DQN上一次的动作
        self.reward = None  # 存储DQN与上一次的动作对应的奖励
        self.PPOMODEL_FLAG = False  # 是否启用PPOModel进行决策
        self.action_list = []
        # self.feature_extract_model = FeatureExtractModel(1330, 256)
        self.init()
        # if model_name == "DQN":
        #     model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/DQN" \
        #                  "/RL_lr=0.0001_AdamW_ep2999_reward=1497.5000_update=24823_mer=0.8_win_acc:0.3500.pth"
        #     state_dict = torch.load(model_path)
        #     self.brain1 = ResNet50(1330, 1)
        #     self.brain1.load_state_dict(state_dict)
        #     print("deep learn model load success")

    def clear_action_list(self):
        self.action_list = []

    def init(self):
        if self.model == "PPOModel" and self.PPOMODEL_FLAG:
            model_path = "ppo2_policyResnet101_tf_Nfea1274_updateStep2048_lr0.00028860349439999996_gamma0.925_entCoef-0.03278114010654392_vfCoef0.5_reward0.9359512007854218_GangIsFalse_MidReward0.01_illegaA-3_abort_reward-1_nsteps_4222976"
            self.brain1 = PPO2.load(model_path)

        if self.model == "DLModel":
            # model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param" \
            #              "/lr=0.001_AdamW_ep8_loss=505.4013366699219_acc=0.6620.pth"
            if self.model_path is None:
                self.model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/DQN" \
                         "/RL_lr=0.0001_AdamW_ep9599_reward=1920.0000_update=25200_mer=0.8_win_acc:0.3100.pth"
            state_dict = torch.load(self.model_path)
            self.brain1 = ResNet50(1330, 1)
            self.brain1.load_state_dict(state_dict)
            print("DQN learn model load success")

        if self.model == "PPOModel_New":
            if self.model_path is None:
                self.model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/PPO_self/linear" \
                     "/RL_lr_Actor=1e-07_AdamW_ep199999_gamma=0.99_reward=-0.2100_update=100000_win_acc:0.2400.pth"
            #state_dict = torch.load(self.model_path)
            self.brain1 = policyNet(1330, 34)
            #self.brain1.load_state_dict(state_dict)
            print("PPO actor learn model load success " + "path:" + self.model_path)

    def reset(self):
        """
        玩家信息重置
        """
        self.catch_card = 0
        self.handcards = []  # 玩家手牌
        self.fulu = []  # 玩家副露
        self.allow_op = []  # 允许操作
        self.allow_handcards = self.handcards  # 允许操作的手牌
        self.compe_op = -1  # 竞争性操作(左吃，中吃，右吃，碰，明杠，胡)

        self.isInitstate = True
        self.action = None
        self.reward = None

    # 推荐出牌
    def recommend_card2(self, state):
        """
        推荐出牌
        @param state: 状态信息
        @return: 推荐打出的牌
        """
        # import reward
        # 转换格式
        handCards = tool2.list10_to_16(state.handcards)
        actions = []
        for i in state.fulu:
            actions.append(tool2.list10_to_16(i))
        allow_hand_cards = handCards
        self.allow_handcards = self.handcards
        state.allow_handcards = state.handcards

        if self.model == "random":
            return random.sample(self.handcards, 1)[0]
        elif self.model == "ppo":  # ppo算法实现部分,等后续训练好之后再加
            # return PPO.choose_action(state)
            # return random.sample(self.handcards, 1)[0]
            result = self.brain1.predict(state)
            result = tool2.f16_to_10(result)
            return result
        elif self.model == "zhiyi_last":  # 最新版知一
            result = url_recommend.get_url_recommend(state, self.seat_id)
            state_fearture, _ = tool2.get_suphx_1330_and_mask(state)
            # print(state_fearture)
            self.action_list.append(Transition(state_fearture, tool2.card_to_index(result), 0, state_fearture, False))
            return result
        elif self.model == "zhiyi":
            result1 = recommand_card1(handCards, actions, allow_hand_cards, discarded_hands=None, round=None)
            result1 = int(tool2.f16_to_10(result1))
            return result1
        elif self.model == "DLModel":
            mask = [0] * 34
            handCards = [tool2.f10_to_16(i) for i in list(set(state.handcards))]
            handCards = [tool2.translate3(i) for i in handCards]
            for item in handCards:
                mask[item] = 1
            mask = torch.tensor(mask)
            mask = mask.view(1, 34)

            features = tool2.card_preprocess_sr_suphx_to_tensor1330(state, search=True, global_state=True)
            features = features.view(1, 1330, 34, 1)
            # features = self.feature_extract_model(features)
            action = self.brain1.predict_with_mask(features, mask).item()
            return tool2.index_to_card(action)
        elif self.model == "PPOModel_New":
            features, mask = tool2.get_suphx_1330_and_mask(state, search=True, global_state=True)
            features = torch.tensor(features, dtype=torch.float)
            mask = torch.tensor(mask)
            mask = mask.view(1, 34)
            features = features.view(1, 1330, 34, 1)
            action = self.brain1.predict(features, mask).item()
            return tool2.index_to_card(action)
        elif self.model == "PPOModel":
            if self.PPOMODEL_FLAG:
                features = tool2.card_preprocess_sr_suphx(state, True, global_state=False)
                # print("sssssssss")
                action, _ = self.brain1.predict(features)
                result = tool2.index_to_card(action)
            else:
                result = url_recommend.get_url_recommend(state, self.seat_id, False)
            return result
        else:
            result3 = input("player" + str(self.seat_id) + ": " + "请输出牌：")
            return result3

    # 推荐操作  把ppo策略也让智一完成
    def recommend_op(self, state):
        '''
        推荐动作
        :param state: 状态
        :param op_map: 待请求决策的map表， key为动作决策下标 value为动作决策对应的牌集
        :return: 操作动作，操作了哪张牌  比如，有两个暗杠，需要知道是对哪个暗杠进行操作的
        '''
        handCards = tool2.list10_to_16(state.handcards)
        actions = []  # 副露
        for i in state.fulu:
            actions.append(tool2.list10_to_16(i))

        if 8 in state.allow_op:
            op = 8
        else:
            # print("玩家", state.seat_id, "可执行的op为：" + str(state.allow_op))
            if "zhiyi" in self.model:
                # op = recommand_op1(handCards, actions, tool2.f10_to_16(state.outcard), state.allow_op,
                # discarded_hands=None,
                # round=None)
                op = url_recommend.get_url_recommend_op(state, state.seat_id, state.allow_op, local_v5=False)
            elif "random" == self.model:
                # op = recommand_op1(handCards, actions, tool2.f10_to_16(state.outcard), state.allow_op,
                # discarded_hands=None,
                # round=None)
                op = url_recommend.get_url_recommend_op(state, state.seat_id, state.allow_op, local_v5=False)
            elif self.model == "ppo":
                # op = recommand_op1(handCards, actions, tool2.f10_to_16(state.outcard), state.allow_op,
                #                    discarded_hands=None,
                #                    round=None)
                op = url_recommend.get_url_recommend_op(state, state.seat_id, state.allow_op, local_v5=False)
            elif self.model == "PPOModel":
                op = url_recommend.get_url_recommend_op(state, state.seat_id, state.allow_op, local_v5=False)
            elif self.model == "DLModel":
                op = url_recommend.get_url_recommend_op(state, state.seat_id, state.allow_op, local_v5=False)
            elif self.model == "PPOModel_New":
                op = url_recommend.get_url_recommend_op(state, state.seat_id, state.allow_op, local_v5=False)
            else:
                op = int(input("请输入操作："))
        return op

    # 左吃
    def zuochi(self, outcard):
        """
        左吃
        Args:
            outcard: 操作牌

        Returns:

        """
        self.handcards.remove(outcard + 1)
        self.handcards.remove(outcard + 2)
        self.fulu.append([outcard, outcard + 1, outcard + 2])
        return

    # 中吃
    def zhongchi(self, outcard):
        """
        中吃
        Args:
            outcard: 操作牌

        Returns:

        """
        self.handcards.remove(outcard - 1)
        self.handcards.remove(outcard + 1)
        self.fulu.append([outcard - 1, outcard, outcard + 1])
        return

    # 右吃
    def youchi(self, outcard):
        """
        右吃
        Args:
            outcard: 操作牌

        Returns:

        """
        self.handcards.remove(outcard - 1)
        self.handcards.remove(outcard - 2)
        self.fulu.append([outcard - 2, outcard - 1, outcard])
        return

    # 碰
    def peng(self, outcard):
        """
        碰
        Args:
            outcard: 操作牌

        Returns:

        """
        self.handcards.remove(outcard)
        self.handcards.remove(outcard)
        self.fulu.append([outcard, outcard, outcard])
        return

    # 补杠
    def buGang(self):
        """
        补杠
        Returns:

        """
        for i in self.fulu:
            if i.count(i[0]) == 3:
                if i[0] in self.handcards:
                    self.handcards.remove(i[0])
                    i.append(i[0])
                    break
        return

    # 明杠
    def mingGang(self, outcard):
        """
        明杠
        Args:
            outcard: 操作牌

        Returns:

        """
        if self.handcards.count(outcard) == 3:
            self.handcards.remove(outcard)
            self.handcards.remove(outcard)
            self.handcards.remove(outcard)
            self.fulu.append([outcard, outcard, outcard, outcard])

        return

    # 暗杠
    def anGang(self):
        """
        暗杠
        Returns:

        """
        S = set(self.handcards)
        for i in S:
            if self.handcards.count(i) == 4:
                self.handcards.remove(i)
                self.handcards.remove(i)
                self.handcards.remove(i)
                self.handcards.remove(i)
                self.fulu.append([i, i, i, i])
                break
        return

    # 过
    def guo(self):
        """
        过牌
        Returns:

        """
        self.allow_op = []
        return

    def canPeng(self, outcard):
        """
        是否可以碰牌
        Args:
            outcard: 操作牌

        Returns: bool值

        """
        if self.handcards.count(outcard) >= 2:
            return True
        return False

    def canZuoChi(self, outcard):
        """
        是否可以左吃
        Args:
            outcard: 操作牌

        Returns: bool值

        """
        if ((outcard + 1) in self.handcards) and ((outcard + 2) in self.handcards) and 1 <= outcard <= 27:
            return True
        return False

    def canZhongChi(self, outcard):
        """
        是否可以中吃
        Args:
            outcard: 操作牌

        Returns: bool值

        """
        if ((outcard - 1) in self.handcards) and ((outcard + 1) in self.handcards) and 2 <= outcard <= 28:
            return True
        return False

    def canYouChi(self, outcard):
        """
        是否可以右吃
        Args:
            outcard: 操作牌

        Returns: bool值

        """
        if ((outcard - 1) in self.handcards) and ((outcard - 2) in self.handcards) and 3 <= outcard <= 29:
            return True
        return False

    def canMingGang(self, outcard):
        """
        是否可以明杠
        Args:
            outcard: 操作牌

        Returns: bool值

        """
        if self.handcards.count(outcard) == 3:
            return True
        return False

    def canAnGang(self):
        """
        是否可以暗杠
        Returns: bool值

        """
        S = set(self.handcards)
        for i in S:
            if self.handcards.count(i) == 4:
                return True
        return False

    def canBuGang(self):
        """
        是否可以补杠
        Returns: bool值

        """
        for i in self.fulu:
            if i.count(i[0]) == 3:
                if i[0] in self.handcards:
                    return True
        return False
