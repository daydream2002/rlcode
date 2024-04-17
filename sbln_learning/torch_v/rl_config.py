import torch
from mahEnv import MahEnv
from mah_player import Player_RL
from sbln_learning.torch_v.agent.PPO_resnet50 import PPO
from sbln_learning.torch_v.agent.DQN_resnet50 import Agent
from sbln_learning.torch_v.replay_buffer import ReplayBufferWithNormal, ReplayMemoryWithImportance, ReplayMemoryWithWin


class Config:
    env = None  # 构建训练环境
    env_test = MahEnv()  # 构建测试环境
    agent_name = "PPO_resnet50"  # 需要构建的Agent类型
    model_path = None  # 是否装载预训练模型
    agent = None
    replay_name = "ReplayBufferWithNormal"  # 需要的经验buffer
    replay = None
    is_global_state = False  # 是否开启全局信息
    dropout_prob = 1
    is_load_model = False  # 是否装载模型参数
    actor_model_path = None
    critic_model_path = None
    is_open_action_mask = False  # 是否开启行为掩码
    is_open_self_play = False  # 是否开启自博弈
    start_ep = 0
    num_episode = 1200
    state_dim = 1330
    action_dim = 34
    actor_lr = 1e-5
    critic_lr = 1e-4
    PPO_kwargs = {
        'lmbda': 0.95,
        'eps': 0.2,
        'ppo_epochs': 10,
        'replay_review_ep': 5
    }
    DQN_kwargs = {
        "epsilon": 0.01,
        "target_update": 10,
        "train_eps": 10
    }
    gamma = 0.99
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 20480
    minimal_size = 1024
    batch_size = 128
    save_path_root = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/DQN"
    # 回合停止控制
    max_episode_rewards = 260
    max_episode_steps = 260

    # buffer权重
    win_rate = 0.8

    def __init__(self):
        # self.reset()
        self.env_test.mahjong.player1 = Player_RL("b", 1, "zhiyi_last")
        self.env_test.mahjong.player2 = Player_RL("c", 2, "zhiyi_last")
        self.env_test.mahjong.player3 = Player_RL("d", 3, "zhiyi_last")

    def reset(self):
        self.env = MahEnv()
        # 构造Agent
        if self.agent_name == "PPO_resnet50":  # 构建ppo
            self.agent = PPO(self.state_dim, self.action_dim, self.actor_lr, self.critic_lr, self.gamma,
                             self.PPO_kwargs,
                             self.device)
            # if self.model_path is not None:
            #     # print(self.model_path)
            #     state_dict = torch.load(self.model_path)
            #     self.agent.actor.load_state_dict(state_dict)
            #     print("PPO Actor pretrain model load success!!")

            if self.is_load_model:
                if self.actor_model_path is not None:
                    #self.agent.actor.load_state_dict(torch.load(self.actor_model_path))
                    print("PPO Actor model load success!!")
                if self.critic_model_path is not None:
                    #self.agent.critic.load_state_dict(torch.load(self.critic_model_path))
                    print("PPO Critic model load success!!")

            if self.is_open_self_play:
                self.env.mahjong.player1.model_path = self.actor_model_path
                self.env.mahjong.player2.model_path = self.actor_model_path
                self.env.mahjong.player3.model_path = self.actor_model_path
                self.env.mahjong.player1.model = "PPOModel_New"
                self.env.mahjong.player2.model = "PPOModel_New"
                self.env.mahjong.player3.model = "PPOModel_New"
                self.env.mahjong.player1.init()
                self.env.mahjong.player2.init()
                self.env.mahjong.player3.init()
                print(f"play1, 2, 3 load {self.actor_model_path} success")
                # print(self.env.mahjong.player3.model_path)
                # self.env.mahjong.player1.model_path = self.model_path
                # self.env.mahjong.player1.brain1.load_state_dict(torch.load(self.actor_model_path))
                # self.env.mahjong.player2.brain1.load_state_dict(torch.load(self.actor_model_path))
                # self.env.mahjong.player3.brain1.load_state_dict(torch.load(self.actor_model_path))
        elif self.agent_name == "DQN_resnet50":  # 构建DQN
            self.agent = Agent(self.state_dim, self.action_dim, self.gamma, self.actor_lr,
                               epsilon=self.DQN_kwargs["epsilon"],
                               target_update=self.DQN_kwargs["target_update"])
            if self.actor_model_path is not None:
                self.agent.q_net.load_state_dict(torch.load(self.actor_model_path))
                print("DQN model load success")
                print(f"model name:{self.actor_model_path}")

        # 构造经验回放buffer
        if self.replay_name == "ReplayBufferWithNormal":  # 构造正常的replay
            self.replay = ReplayBufferWithNormal(self.buffer_size)
        elif self.replay_name == "ReplayMemoryWithImportance":  # 构造重要性采样的replay
            self.replay = ReplayMemoryWithImportance(self.buffer_size, self.win_rate)
        elif self.replay_name == "ReplayMemoryWithWin":  # 构造重要性采样的replay
            self.replay = ReplayMemoryWithWin(self.buffer_size, self.win_rate)

        # 环境
        if self.is_global_state:
            self.env.is_global_state = True
            self.env.dropout_prob = self.dropout_prob

        if self.is_open_self_play:
            print("is self play")
            pass
        else:
            self.env.mahjong.player1 = Player_RL("b", 1, "zhiyi_last")
            self.env.mahjong.player2 = Player_RL("c", 2, "zhiyi_last")
            self.env.mahjong.player3 = Player_RL("d", 3, "zhiyi_last")
            print("is not self play")
