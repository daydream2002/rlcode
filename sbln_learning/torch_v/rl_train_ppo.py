import os
import time

import gym
import numpy as np
import torch

from sbln_learning.torch_v.replay_buffer import Transition
from sbln_learning.torch_v.rl_config import Config


def run_episode(env, agent, repalymemory, is_clear_replay=True, is_back=True):
    if is_clear_replay:
        repalymemory.clearWin()
        repalymemory.clearLose()
    state, mask = env.reset()
    reward_total = 0
    state_action_list = []
    is_invalid_discard = False
    while True:
        action = agent.take_action(state, mask)
        next_state, reward, done, info = env.step(action)
        state_action_list.append(Transition(state, action, reward, next_state, done))
        state = next_state
        mask = info["mask"]
        # print(reward)
        if done:
            if reward == -50:  # 由于非法出牌
                repalymemory.push(Transition(state, action, reward, next_state, done))
                is_invalid_discard = True
            else:
                win_resutl = env.mahjong.game.win_result  # 获取牌局结果集
                play0_score = win_resutl.get(0).get("score")  # 获取零号玩家的最终分数
                if play0_score > 0:
                    extra_score = 1
                else:
                    extra_score = -1
                if is_back:
                    state_action_list_ = []
                    for idx, item in enumerate(state_action_list):  # 更新动作奖励分数
                        reward_ = play0_score * ((idx + 1) / len(state_action_list)) + extra_score
                        state_action_list_.append(Transition(item.state, item.action, reward_, item.next_state, item.done))
                        reward_total += reward_
                    repalymemory.push_list(state_action_list_)
                else:
                    repalymemory.push_list(state_action_list)
            break
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = repalymemory.sample(len(repalymemory))
    T_data = Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
    # print(T_data)
    agent.learn(T_data)
    return reward_total, is_invalid_discard


def episode_evaluate(env, agent, render, test_sum):
    reward_list = []
    play0_win_test = 0
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
            state = next_state
            mask = info["mask"]
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
    cfg.agent_name = "PPO_resnet50"
    cfg.replay_name = "ReplayMemoryWithWin"
    cfg.save_path_root = "./model_param"
    # cfg.actor_model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/PPO_self/linear" \
    #                  "/RL_lr_Actor=1e-07_AdamW_ep199999_gamma=0.99_reward=-0.2100_update=100000_win_acc:0.2400.pth"
    # cfg.critic_model_path = "/home/tonnn/xiu/xy/shangrao_mj_rl_v4_suphx_copy_01/sbln_learning/torch_v/model_param/PPO_self/linear" \
    #                  "/RL_lr_Critic=0.0001_AdamW_ep199999_reward=-0.2100_update=100000_win_acc:0.2400_loss=57.1771.pth"
    cfg.is_load_model = False
    cfg.is_global_state = False
    cfg.critic_lr = 1e-4
    cfg.actor_lr = 1e-7
    cfg.gamma = 0.99
    cfg.num_episode = 100000
    cfg.start_ep = 0
    cfg.is_open_self_play = True
    cfg.PPO_kwargs["ppo_epochs"] = 10
    cfg.PPO_kwargs["replay_review_ep"] = 1
    cfg.buffer_size = 100
    cfg.reset()

    train_log_interval = 500  # 打印train轮次间隔
    critic_loss_interval = 500
    test_interval = 5000
    file_log_interval = 5000
    save_model_interval = 5000
    test_sum = 100
    is_open_file_log = False
    dropout_prob_interval = 1000
    is_back = False  # 是否使用回推法
    # replay_review_ep = 5

    env = cfg.env  # 创建麻将对打环境
    agent = cfg.agent
    replaymemory = cfg.replay
    batch_size = cfg.batch_size
    num_episodes = cfg.num_episode
    log_name = f"rl_{cfg.agent_name}_train"
    reward_list = []
    play0_win_train = 0
    play0_invalid_discard = 0
    file_log = f"./log" \
               f"{log_name}={cfg.replay_name}_ep{num_episodes}.txt"
    print(f"---train_start---")
    print(f"---critic_lr:{cfg.critic_lr}_actor_lr:{cfg.actor_lr}_gamma:{cfg.gamma}_num_episode:{cfg.num_episode}---")
    for idx in range(cfg.start_ep, num_episodes):
        # if (idx + 1) % train_log_interval == 0:
        #     print(f"---ep{idx}_train_start---")
            # replaymemory.show()
        is_clear_replay = False
        if (idx + 1) % cfg.PPO_kwargs["replay_review_ep"] == 0:
            is_clear_replay = True

        reward_episode, is_invalid_discard = run_episode(env, agent, replaymemory, is_clear_replay, is_back)
        if is_invalid_discard:  # 非法出牌结束游戏
            play0_invalid_discard += 1
        if env.mahjong.game.win_result.get(0).get("win") == 1:
            play0_win_train += 1
        if (idx + 1 - cfg.start_ep) % file_log_interval == 0 and is_open_file_log:
            with open(file_log, "a") as log:
                log.write(f"--train--win: {play0_win_train} --sum_game: {idx + 1 - cfg.start_ep} "
                          f"--win_acc:{play0_win_train / (idx + 1 - cfg.start_ep):0.4f}\n\n")
        if (idx + 1 - cfg.start_ep) % train_log_interval == 0:
            print(f"---ep{idx}_train_end---")
            print(f"--win: {play0_win_train} --sum_game: {idx + 1 - cfg.start_ep} --invalid_discard:{play0_invalid_discard} "
                  f"--win_acc:{play0_win_train / (idx + 1 - cfg.start_ep):0.4f}")

        if (idx + 1 - cfg.start_ep) % critic_loss_interval == 0:
            print(f"critic_loss_last:{agent.last_critic_loss_sum:0.4f}")

        if cfg.is_global_state and (idx + 1 - cfg.start_ep) % dropout_prob_interval == 0:  # 缩减可见信息
            env.dropout_prob = max(env.dropout_prob - 0.2, 0)
            print(f"---env.dropout_prob:{env.dropout_prob}---")

        if (idx + 1 - cfg.start_ep) % test_interval == 0:
            print("-----------------------------------")
            print(f"---ep{idx}_test_start---")
            test_reward, play0_win_test, test_sum = episode_evaluate(cfg.env_test, agent, False, test_sum)
            if env.mahjong.game.win_result.get(0).get("win") == 1:
                play0_win_test += 1

            if is_open_file_log:
                with open(file_log, "a") as log:
                    log.write(f"==test--win: {play0_win_test} --sum_game: {test_sum} "
                              f"--win_acc:{play0_win_test / test_sum:0.4f} --ep{idx}_reward:{test_reward:0.4f}\n\n")

            print(f"--win: {play0_win_test} --sum_game: {test_sum} --win_acc:{play0_win_test / test_sum:0.4f}")
            print(f"ep{idx}_reward:{test_reward:0.4f}")
            print("------------------------------------")
            # if replaymemory.__len__() > batch_size:
            if agent.count % save_model_interval == 0:
                torch.save(agent.actor.state_dict(),
                           os.path.join(cfg.save_path_root,
                                        f"RL_lr_Actor={cfg.actor_lr}_AdamW_ep{idx}_gamma={cfg.gamma}"
                                        f"_reward={test_reward:0.4f}"
                                        f"_update={agent.count}"
                                        f"_win_acc:{play0_win_test / test_sum:0.4f}.pth"))
                torch.save(agent.critic.state_dict(),
                           os.path.join(cfg.save_path_root,
                                        f"RL_lr_Critic={cfg.critic_lr}_AdamW_ep{idx}_reward={test_reward:0.4f}"
                                        f"_update={agent.count}"
                                        f"_win_acc:{play0_win_test / test_sum:0.4f}"
                                        f"_loss={agent.last_critic_loss_sum:0.4f}.pth"))
                print("----save success----")
                cfg.env.mahjong.player1.brain1.load_state_dict(agent.actor.state_dict())
                cfg.env.mahjong.player2.brain1.load_state_dict(agent.actor.state_dict())
                cfg.env.mahjong.player3.brain1.load_state_dict(agent.actor.state_dict())
                print("other player state update")
