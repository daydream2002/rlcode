import os
import numpy as np
import torch

from mah_tool import tool2
from sbln_learning.torch_v.replay_buffer import Transition
from sbln_learning.torch_v.rl_config import Config


def run_episode(env, agent, repalymemory, batch_size, train_eps, is_back=True):
    state, masks = env.reset()
    reward_total = 0
    state_action_list = []
    while True:
        action = agent.predict(state, masks)
        next_state, reward, done, info = env.step(action)
        state_action_list.append(Transition(state, action, reward, next_state, done))
        masks = info["mask"]
        state = next_state
        if done:
            if reward == -50:  # 由于非法出牌
                repalymemory.push(Transition(state, action, reward, next_state, done))
            else:
                win_resutl = env.mahjong.game.win_result  # 获取牌局结果集
                play0_score = win_resutl.get(0).get("score")  # 获取零号玩家的最终分数
                extra_sorce = 0
                if play0_score > 0:
                    extra_sorce = 1
                state_action_list_ = []

                if is_back:
                    for idx, item in enumerate(state_action_list):  # 更新动作奖励分数
                        reward_ = play0_score * ((idx + 1) / len(state_action_list)) + extra_sorce
                        state_action_list_.append(Transition(item.state, item.action, reward_, item.next_state, item.done))
                        reward_total += reward_
                    #     print(reward_)
                    # exit(1)
                    # 将动作奖励分数加入到经验回放中
                    repalymemory.push_list(state_action_list_)
                else:
                    repalymemory.push_list(state_action_list)

                # 获取获胜玩家idx
                win_idx = -1
                for idx in win_resutl:
                    if win_resutl.get(idx).get("win") == 1:
                        win_idx = idx
                        break

                if win_idx == -1 or win_idx == 0:
                    break

                # print(win_idx)
                state_action_list_ = []
                action_list = None
                win_play = None
                win_player_socre = None
                if win_idx == 1:
                    action_list = env.mahjong.player1.action_list
                    win_play = env.mahjong.player1
                    win_player_socre = win_resutl.get(win_idx).get("score")
                elif win_idx == 2:
                    action_list = env.mahjong.player2.action_list
                    win_play = env.mahjong.player2
                    win_player_socre = win_resutl.get(win_idx).get("score")
                elif win_idx == 3:
                    action_list = env.mahjong.player3.action_list
                    win_play = env.mahjong.player3
                    win_player_socre = win_resutl.get(win_idx).get("score")

                # print(len(action_list))
                for idx, item in enumerate(action_list):
                    if idx < len(action_list) - 1:
                        state_action_list_.append(Transition(item.state, item.action, 0,
                                                                  action_list[idx + 1].state, item.done))
                    else:
                        tmp, _ = tool2.get_suphx_1330_and_mask(env.mahjong.game.get_state(win_play, env.mahjong.game))
                        state_action_list_.append(Transition(item.state, item.action, 0, tmp, True))

                action_list_ = []
                for idx, item in enumerate(state_action_list_):  # 更新动作奖励分数
                    reward_ = win_player_socre * ((idx + 1) / len(state_action_list_))
                    action_list_.append(Transition(item.state, item.action, reward_, item.next_state, item.done))

                is_load_orther_aciton = False
                if is_load_orther_aciton:
                    repalymemory.push_list(action_list_)

                # print(action_list_)
                # print(len(action_list_))
                # exit(0)

            break
    if repalymemory.check(batch_size):
        # print(T_data)
        for idx in range(train_eps):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = repalymemory.sample(batch_size)
            T_data = Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            agent.update(T_data)
    return reward_total


def episode_evaluate(env, agent, render):
    reward_list = []
    play0_win_test = 0
    test_sum = 100
    for i in range(test_sum):
        state, mask = env.reset()
        state_action_list = []
        sorce_total = 0
        # reward_episode = 0
        while True:
            action = agent.predict(state, mask)
            next_state, reward, done, info = env.step(action)
            state_action_list.append(Transition(state, action, reward, next_state, done))
            # reward_episode += reward
            mask = info["mask"]
            state = next_state
            if done:
                if reward == -50:  # 由于非法出牌
                    reward_total = reward
                else:
                    win_resutl = env.mahjong.game.win_result  # 获取牌局结果集
                    if win_resutl.get(0).get("win") == 1:
                        play0_win_test += 1
                    play0_score = win_resutl.get(0).get("score")  # 获取零号玩家的最终分数
                    # print(f"play0_score:{play0_score}")
                    # for item in enumerate(state_action_list):  # 更新动作奖励分数
                    #     reward_ = play0_score * ((idx + 1) / len(state_action_list))
                    #     reward_total += reward_
                    sorce_total += play0_score
                break
            if render:
                env.render()
        reward_list.append(sorce_total)
    return np.mean(reward_list).item(), play0_win_test, test_sum


if __name__ == "__main__":

    # print("prepare for RL")
    cfg = Config()
    cfg.agent_name = "DQN_resnet50"
    cfg.replay_name = "ReplayMemoryWithImportance"
    cfg.save_path_root = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/DQN/noback"
    cfg.actor_model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param" \
                           "/lr=0.001_AdamW_ep9_loss=490.4792175292969_acc=0.6570.pth"
    cfg.num_episode = 20000
    cfg.actor_lr = 1e-4
    cfg.DQN_kwargs["epsilon"] = 0.01
    cfg.DQN_kwargs["target_update"] = 50
    cfg.buffer_size = 1000
    cfg.batch_size = 500
    cfg.is_global_state = False
    cfg.is_open_self_play = False
    cfg.start_ep = 0
    cfg.reset()

    train_log_interval = 50  # 打印train轮次间隔
    critic_loss_interval = 50
    test_interval = 100
    file_log_interval = 500
    save_model_interval = 1000
    test_sum = 100
    is_open_file_log = False
    dropout_prob_interval = 5000
    is_back = False

    env = cfg.env  # 创建麻将对打环境
    agent = cfg.agent
    replaymemory = cfg.replay
    batch_size = cfg.batch_size
    num_episodes = cfg.num_episode
    log_name = "rl_DQN_train_noback"
    reward_list = []
    play0_win_train = 0
    file_log = f"/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/" \
               f"{log_name}={cfg.replay_name}_ep{num_episodes}.txt"
    print(f"---train_start---")
    for idx in range(cfg.start_ep, num_episodes):
            # replaymemory.show()
        reward_episode = run_episode(env, agent, replaymemory, batch_size, cfg.DQN_kwargs["train_eps"], is_back)
        if env.mahjong.game.win_result.get(0).get("win") == 1:
            play0_win_train += 1
        if (idx + 1) % file_log_interval == 0:
            with open(file_log, "a") as log:
                log.write(f"--train--win: {play0_win_train} --sum_game: {idx + 1 - cfg.start_ep} "
                          f"--win_acc:{play0_win_train / (idx + 1 - cfg.start_ep):0.4f}\n\n")
                log.close()
            print(f"--win: {play0_win_train} --sum_game: {idx + 1 - cfg.start_ep} --win_acc:{play0_win_train / (idx + 1 - cfg.start_ep):0.4f}")

        if (idx + 1) % train_log_interval == 0:
            print(f"---ep{idx}_train_end---")
            print(f"--win: {play0_win_train} --sum_game: {idx + 1 - cfg.start_ep} "
                  f"--win_acc:{play0_win_train / (idx + 1 - cfg.start_ep):0.4f}")

        if (agent.count + 1) % dropout_prob_interval == 0 and cfg.is_global_state:  # 缩减可见信息
            env.dropout_prob = max(env.dropout_prob - 0.2, 0)
            print(f"---env.dropout_prob:{env.dropout_prob}---")

        if (idx + 1) % test_interval == 0:
            print("-----------------------------------")
            print(f"---ep{idx}_test_start---")
            test_reward, play0_win_test, test_sum = episode_evaluate(env, agent, False)
            if env.mahjong.game.win_result.get(0).get("win") == 1:
                play0_win_test += 1
            if is_open_file_log:
                with open(file_log, "a") as log:
                    log.write(f"==test--win: {play0_win_test} --sum_game: {test_sum} "
                              f"--win_acc:{play0_win_test / test_sum:0.4f} --ep{idx}_reward:{test_reward:0.4f}\n\n")
                    log.close()
            print(f"--win: {play0_win_test} --sum_game: {test_sum} --win_acc:{play0_win_test / test_sum:0.4f}")
            print(f"ep{idx}_reward:{test_reward:0.4f}")
            print("------------------------------------")
            if replaymemory.__len__() > batch_size:
                torch.save(agent.target_q_net.state_dict(),
                           os.path.join(cfg.save_path_root,
                                        f"RL_lr={agent.lr}_AdamW_ep{idx}_reward={test_reward:0.4f}_update={agent.count}"
                                        f"_mer={replaymemory.win_rate}_"
                                        f"win_acc:{play0_win_test / test_sum:0.4f}.pth"))
