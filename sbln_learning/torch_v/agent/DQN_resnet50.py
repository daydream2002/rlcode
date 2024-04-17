import numpy as np
from torch import optim, nn

from mah_tool.feature_extract_v10 import calculate_king_sys_suphx2
from sbln_learning.torch_v.policy_model.resnet_50 import ResNet50
import torch

device = torch.device("cuda")


class Agent(object):

    def __init__(self, observation_dim, action_dim, gamma, lr, epsilon, target_update):

        model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param" \
                     "/lr=0.001_AdamW_ep9_loss=490.4792175292969_acc=0.6570.pth"
        state_dict = torch.load(model_path)

        self.action_dim = action_dim
        self.q_net = ResNet50(observation_dim, 1)
        self.q_net.load_state_dict(state_dict)
        self.q_net = self.q_net.to(device)
        self.target_q_net = ResNet50(observation_dim, 1)
        self.target_q_net.load_state_dict(state_dict)
        self.target_q_net = self.target_q_net.to(device)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dropout = 1

        self.optimizer = optim.AdamW(params=self.q_net.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def take_action(self, state, mask=None):
        if np.random.uniform(0, 1) < 1 - self.epsilon:
            state = torch.tensor(state, dtype=torch.float).view(1, 1330, 34, 1)
            state = state.to(device)
            if mask is not None:
                mask = torch.tensor(mask, dtype=torch.bool).view(1, 34).to(device)
                action = self.q_net.predict_with_mask(state, mask).item()
            else:
                action = self.q_net.predict(state).item()
        else:
            action = np.random.choice(self.action_dim)
        return action

    def predict(self, inputs, mask):
        inputs = torch.tensor(inputs, dtype=torch.float).view(1, 1330, 34, 1)
        inputs = inputs.to(device)
        mask = torch.tensor(mask, dtype=torch.bool).view(1, 34).to(device)
        return self.q_net.predict_with_mask(inputs, mask).item()

    def update(self, transition_dict):

        # print(transition_dict)
        # exit(1)
        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1)  # 扩充维度
        rewards = np.expand_dims(transition_dict.reward, axis=-1)  # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1)  # 扩充维度


        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(dones.shape)
        # exit(1)
        # update q_values
        # gather(1, acitons)意思是dim=1按行号索引， index=actions
        # actions=[[1, 2], [0, 1]] 意思是索引出[[第一行第2个元素， 第1行第3个元素],[第2行第1个元素， 第2行第2个元素]]
        # 相反，如果是这样
        # gather(0, acitons)意思是dim=0按列号索引， index=actions
        # actions=[[1, 2], [0, 1]] 意思是索引出[[第一列第2个元素， 第2列第3个元素],[第1列第1个元素， 第2列第2个元素]]
        # states.shape(64, 4) actions.shape(64, 1), 每一行是一个样本，所以这里用dim=1很合适
        predict_q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            # max(1) 即 max(dim=1)在行向找最大值，这样的话shape(64, ), 所以再加一个view(-1, 1)扩增至(64, 1)
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        l = self.loss(predict_q_values, q_targets)

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # copy model parameters
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1
        if self.count % 50 == 0:
            print(f"---Agent update ep={self.count}---")


if __name__ == '__main__':
    handCards0 = [4, 4, 6, 18, 19, 22, 37, 38, 41, 50, 54]
    fulu_ = [[[20, 20, 20]], [], [], []]
    king_card = 39
    all_player_handcards = [[4, 4, 6, 18, 19, 22, 37, 38, 41, 50, 54], [3, 5, 5, 7, 17, 22, 23, 35, 40, 51, 51, 51, 53],
                            [6, 8, 9, 9, 17, 35, 36, 38, 38, 39, 49, 52, 52],
                            [2, 2, 6, 7, 8, 20, 23, 24, 25, 34, 34, 35, 41]]
    card_library = [19, 36, 17, 50, 41, 40, 55, 49, 25, 34, 55, 4, 55, 7, 54, 33, 40, 53, 55, 22, 39, 22, 49, 7, 33, 6,
                    5, 39, 52, 3, 53, 51, 34, 36, 1, 9, 19, 23, 24, 36, 41, 24, 2, 3, 4, 9, 18, 40, 35, 8, 50, 18, 2, 1,
                    5, 19, 37, 23, 21, 21, 37, 3, 39, 38, 52, 17, 8, 37, 18, 33, 1, 54, 49, 53, 50, 21, 1, 33, 24, 25,
                    21, 25, 54]
    # card_library = []
    # all_palyer_king_nums = [0, 0, 1, 0]
    # all_palyer_king_nums = [0, 0, 0, 0]
    discards_seq = [[], [], [20], []]
    remain_card_num = 83
    # remain_card_num = 0
    self_king_num = 0
    fei_king_nums = [0, 0, 0, 0]
    round_ = 2
    dealer_flag = [0, 0, 1, 0]
    # dealer_flag = [1, 0, 0, 0]

    features1 = calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                         discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                                         dealer_flag, search=True, global_state=False, dropout_prob=0)

    print(torch.tensor(features1).shape)
    # card_library = []

    handCards0 = [4, 4, 6, 18, 19, 22, 37, 38, 41, 50, 54]
    fulu_ = [[[20, 20, 20]], [], [], []]
    king_card = 39
    all_player_handcards = [[4, 4, 6, 18, 19, 22, 37, 38, 41, 50, 54], [3, 5, 5, 7, 17, 22, 23, 35, 40, 51, 51, 51, 53],
                            [6, 8, 9, 9, 17, 35, 36, 38, 38, 39, 49, 52, 52],
                            []]
    card_library = [19, 36, 17, 50, 41, 40, 55, 49, 25, 34, 55, 4, 55, 7, 54, 33, 40, 53, 55, 22, 39, 22, 49, 7, 33, 6,
                    5, 39, 52, 3, 53, 51, 34, 36, 1, 9, 19, 23, 24, 36, 41, 24, 2, 3, 4, 9, 18, 40, 35, 8, 50, 18, 2, 1,
                    5, 19, 37, 23, 21, 21, 37, 3, 39, 38, 52, 17, 8, 37, 18, 33, 1, 54, 49, 53, 50, 21, 1, 33, 24, 25,
                    21, 25, 54]
    # card_library = []
    # all_palyer_king_nums = [0, 0, 1, 0]
    # all_palyer_king_nums = [0, 0, 0, 0]
    discards_seq = [[], [], [20], []]
    remain_card_num = 83
    # remain_card_num = 0
    self_king_num = 0
    fei_king_nums = [0, 0, 0, 0]
    round_ = 2
    dealer_flag = [0, 0, 1, 0]

    features2 = calculate_king_sys_suphx2(handCards0, fulu_, king_card, all_player_handcards, card_library,
                                          discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                                          dealer_flag, search=True, global_state=False, dropout_prob=0)

    features1 = torch.tensor(features1, dtype=torch.float).view(1, 1330, 34, 1)
    features2 = torch.tensor(features2, dtype=torch.float).view(1, 1330, 34, 1)

    print(features1.sum())
    print(features2.sum())


    print(torch.equal(features1, features2))
    print((features1 == features2).sum())
    # exit(1)

    resnet50 = ResNet50(1330, 1)
    model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param" \
                 "/lr=0.001_AdamW_ep6_loss=527.990966796875_acc=0.6684.pth"
    state_dict = torch.load(model_path)
    resnet50.load_state_dict(state_dict)

    # tensor([[-7.0967, -8.4916, -6.3336, -5.8273, -5.7699, -5.5705, -6.7060,
    #          -6.1574, -6.1162, -5.8510, -5.9615, -5.8573, -7.1952, -6.2276,
    #          -6.1878, -6.5061, -5.9019, -5.9827, -6.1582, -6.0992, -6.2094,
    #          -6.1592, -5.5012, -6.2593, -6.1783, -6.0950, -5.8109, -6.4216,
    #          -5.8258, -6.4622, -6.2832, -6.1581, -2.3443, -10.2849]],

    # print(resnet50(features1))
    print(resnet50.predict(features1).item())

    resnet50_ = ResNet50(1330, 1)
    model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param" \
                 "/lr=0.001_AdamW_ep6_loss=527.990966796875_acc=0.6684.pth"
    state_dict = torch.load(model_path)
    resnet50_.load_state_dict(state_dict)

    print(resnet50.predict(features2).item())
    # print(feature_extract_model(x).shape)
    # print(resnet50(feature_extract_model(x)).shape)
    # print(resnet50(feature_extract_model(x)))
    # print(resnet50.predict(feature_extract_model(x)))
