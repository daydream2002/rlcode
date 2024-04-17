#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : mahEnv.py
# @Description: 强化学习环境配置
# import demo.hello.hello  as h
import gym
from gym import spaces
import numpy as np

from mahjong import MahjongEnv
import mah_tool.tool2 as tool2
from mah_tool import url_recommend


class MahEnv(gym.Env):
    '''
        强化学习环境
        className:MahEnv
        fileName:mahEnv.py
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, is_global_state=False):
        """
        构造器
        """
        self.mahjong = MahjongEnv()  # 创建麻将环境
        self.action_space = spaces.Discrete(34)
        self.observation_space = spaces.Box(low=0, high=1, dtype=np.int, shape=(1274, 34, 1))
        self.is_global_state = is_global_state
        self.dropout_prob = 0
        self.reset()

    def step(self, action):
        """
        执行action动作，获得状态state、奖励reward、是否结束done和其他信息info
        @param action: 执行的动作
        @return: [state，reward，done，info]
        """
        # print(self.action_space)
        if isinstance(action, np.int32) or isinstance(action, np.int64) or isinstance(action, int):
            a = action
        elif len(action) == 1:
            a = action[0]
        else:
            a = np.argmax(action)
            assert self.action_space.contains(a), "%r (%s) invalid" % (a, type(a))
            # 调用MahjongEnv的step函数

        mah_action = self.mahjong.action_set[a]  # 转换成麻将动作类型
        mah_state, reward, done, info = self.mahjong.step(mah_action)
        self.states = mah_state
        # print(self.states.round)
        # mah_state = tool2.card_preprocess_sr_king(mah_state,search=True,global_state=True)
        # mah_state = tool2.get_reshape_obs_(mah_state, 61, 61)
        # info = {"handcards": [tool2.card_to_index(card) for card in self.states.handcards]}

        # test suphx features encoding
        mah_state, mask = tool2.get_suphx_1330_and_mask(mah_state, search=True, global_state=self.is_global_state,
                                                        dropout_prob=self.dropout_prob)
        info["mask"] = mask
        return mah_state, reward, done, info

    # 用智一专家模型，生成轨迹
    def get_zhiyi_recommend_action(self):
        """
        用智一专家模型，生成轨迹
        @return: 推荐出牌
        """
        recommend_card = self.mahjong.get_zhiyi_recommend_action(self.get_url_state())  # 为十进制牌，未转换成0-33表示
        zhiyi_recommend_card = tool2.card_to_index(recommend_card)
        # print("state中的手牌是：", self.get_url_state().handcards)
        return zhiyi_recommend_card

    def get_url_state(self):
        """
        获取玩家0的当前状态state
        @return: state
        """
        state = self.mahjong.game.get_state(self.mahjong.player1, self.mahjong.game)
        return state

    def get_url_recommand(self):
        """
        玩家0的推荐出牌
        @return: 推荐的出牌
        """
        state = self.get_url_state()
        result = url_recommend.get_url_recommend(state, 0)
        # print(result)
        result = tool2.card_to_index(result)
        return result

    def reset(self):
        """
        游戏参数重置
        @return: state
        """
        mah_state = self.mahjong.reset()

        self.states = mah_state  # 特征编码处理

        mah_state, mask = tool2.get_suphx_1330_and_mask(mah_state, search=True, global_state=False,
                                                        dropout_prob=0)

        return mah_state, mask

    def render(self, mode='human'):
        """
        # 打印相关信息
        @param mode: 模型
        @return:
        """
        print("player" + str(self.states.seat_id) + ":")
        print("玩家", self.states.seat_id,
              "   outcard: " + str(self.states.outcard) + "       余牌：" + str(
                  self.states.card_library) + "    精牌：" + str(
                  self.states.jing_card))
        print("幅露：" + str(self.states.player_fulu) + "   " + "手牌：" + str(self.states.handcards) + "    " + "抓牌" + str(
            self.states.catch_card))
        # print("player0:  " + "幅露：" + str(player0.fulu) + "   " + "手牌：" + str(player0.handcards) + "   手牌长度：" + str(
        #     len(player0.handcards)))
        # print("player1:  " + "幅露：" + str(player1.fulu) + "   " + "手牌：" + str(player1.handcards) + "   手牌长度：" + str(
        #     len(player1.handcards)))
        # print("player2:  " + "幅露：" + str(player2.fulu) + "   " + "手牌：" + str(player2.handcards)+"   "+ str(len(player2.handcards)))
        # print("player3:  " + "幅露：" + str(player3.fulu) + "   " + "手牌：" + str(player3.handcards)+"   "+ str(len(player3.handcards)))
        print("\n")
        return None

    def close(self):
        return None


# 环境测试
if __name__ == '__main__':
    env = MahEnv()
    env.reset()
    env.step(env.action_space.sample())
    print("env.state", env.states)
    env.step(env.action_space.sample())
    print("env.state", env.states)
