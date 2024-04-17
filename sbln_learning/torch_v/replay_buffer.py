from collections import deque, namedtuple

import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBufferWithNormal:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def push(self, args):
        self.buffer.append(args)

    def push_list(self, state_action_list):
        for item in state_action_list:
            self.push(item)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch_data = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch_data)
        return state, action, reward, next_state, done


class ReplayMemoryWithWin(object):
    def __init__(self, memory_size, win_rate):
        # self.memory = deque([], maxlen=memory_size)

        self.win_rate = win_rate
        # print(int(memory_size * self.win_rate))
        # print(int(memory_size * (1 - self.win_rate)))
        self.memory_win = deque([], maxlen=int(memory_size * self.win_rate))
        self.memory_lose = deque([], maxlen=int(memory_size * (1 - self.win_rate)))

    def sample(self, batch_size):
        # print(int(batch_size * self.win_rate))
        b1 = min(batch_size, len(self.memory_win))
        b2 = min(batch_size - b1, len(self.memory_lose))
        batch_data = random.sample(self.memory_win, b1)
        batch_data += random.sample(self.memory_lose, b2)
        state, action, reward, next_state, done = zip(*batch_data)
        return state, action, reward, next_state, done

    def clearWin(self):
        self.memory_win.clear()

    def clearLose(self):
        self.memory_lose.clear()

    def push(self, args):
        # *args: 把传进来的所有参数都打包起来生成元组形式
        # self.push(1, 2, 3, 4, 5)
        # args = (1, 2, 3, 4, 5)
        if args.reward > 0:
            self.memory_win.append(args)
        else:
            self.memory_lose.append(args)

    def push_list(self, state_action_list):
        for item in state_action_list:
            self.push(item)

    def check(self, batch_size):
        if len(self.memory_win) > int(batch_size * self.win_rate) + 10 and len(self.memory_lose) > int(
                batch_size * (1 - self.win_rate)) + 10:
            return True
        else:
            return False

    def show(self):
        # print(self.memory)
        pass

    def __len__(self):
        return len(self.memory_win) + len(self.memory_lose)

class ReplayMemoryWithImportance(object):

    def __init__(self, memory_size, win_rate):
        # self.memory = deque([], maxlen=memory_size)

        self.win_rate = win_rate
        # print(int(memory_size * self.win_rate))
        # print(int(memory_size * (1 - self.win_rate)))
        self.memory_win = deque([], maxlen=int(memory_size * self.win_rate))
        self.memory_lose = deque([], maxlen=int(memory_size * (1 - self.win_rate)))

    def sample(self, batch_size):
        # print(int(batch_size * self.win_rate))
        batch_data = random.sample(self.memory_win, int(batch_size * self.win_rate))
        batch_data += random.sample(self.memory_lose, int(batch_size * (1 - self.win_rate)))
        state, action, reward, next_state, done = zip(*batch_data)
        return state, action, reward, next_state, done

    def push(self, args):
        # *args: 把传进来的所有参数都打包起来生成元组形式
        # self.push(1, 2, 3, 4, 5)
        # args = (1, 2, 3, 4, 5)
        if args.reward > 0:
            self.memory_win.append(args)
        else:
            self.memory_lose.append(args)

    def push_list(self, state_action_list):
        for item in state_action_list:
            self.push(item)

    def check(self, batch_size):
        if len(self.memory_win) > int(batch_size * self.win_rate) + 10 and len(self.memory_lose) > int(
                batch_size * (1 - self.win_rate)) + 10:
            return True
        else:
            return False

    def show(self):
        # print(self.memory)
        pass

    def __len__(self):
        return len(self.memory_win) + len(self.memory_lose)