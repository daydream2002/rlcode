from collections import deque, namedtuple

from torch.utils.data import Dataset, DataLoader
from mah_tool.feature_extract_v10 import *


# def get_action_label():
#     """
#     用于生成弃牌所有标签，和对应的反对应（用于实现后面的多模型决策）
#     :return: 生成决策对应id, 以及id对应的标签 action_to_id, id_to_action
#     """
#     action_list = []
#     for i in range(34):  # 弃牌决策，因为所有的牌都能丢弃，所以是0-33
#         action_list.append(f"discard_{i}")
#
#     action_to_id = {}
#     for item in action_list:  # 生成标签对应的id， 使用字典的方式例如,{"discard_0": 0, "discard_1": 1, .....}
#         action_to_id[item] = len(action_to_id.keys())
#
#     id_to_action = {value: key for key, value in action_to_id.items()}
#     return action_to_id, id_to_action
#
#
# action_to_id, id_to_action = get_action_label()  # 先生成决策对应id, 以及id对应的标签 action_to_id, id_to_action


class DatasetTotal(Dataset):
    def __init__(self, root, mode):
        super(DatasetTotal, self).__init__()
        self.root = root
        file_names = os.listdir(root)  # 获取标签文件名
        # print(file_names)
        # self.sample_list = {}
        self.train_sample = []  # 制作训练数据文件
        self.test_sample = []  # 制作测试数据文件
        # lists = []
        for item in file_names:
            temp = [os.path.join(item, i) for i in os.listdir(os.path.join(self.root, item))]
            if len(temp) < 5000:
                continue
            # lists.append((len(temp), item))
            # print(f"{item}:{len(temp)}")
            # print(temp)
            temp_train = temp[: int(len(temp) * 0.8)]  # 就取一点来训练，另外的用来测试
            temp_test = temp[int(len(temp) * 0.8):]
            # temp_train = temp  # 就取一点来训练，另外的用来测试
            # temp_test = temp
            self.train_sample += temp_train
            self.test_sample += temp_test

        # lists = sorted(lists, key=lambda x: -x[0])
        # for item in lists:
        #     print(f"{item[1]}:{item[0]}")
        # 在将合并的文件进行打乱
        random.shuffle(self.train_sample)
        random.shuffle(self.test_sample)

        if mode == "train":
            self.sample = self.train_sample
        else:
            self.sample = self.test_sample
        # print(len(self.sample))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        df = open(os.path.join(self.root, self.sample[idx]), encoding="utf-8")
        data = json.load(df)
        featrues, size_len = get_feature_total(data)
        return featrues, torch.tensor(size_len), torch.tensor([data["label"]], dtype=torch.float)


class DatasetGlobal(Dataset):
    def __init__(self, root, mode):
        super(DatasetGlobal, self).__init__()
        self.root = root
        file_names = os.listdir(root)  # 获取标签文件名
        # print(file_names)
        # self.sample_list = {}
        self.train_sample = []  # 制作训练数据文件
        self.test_sample = []  # 制作测试数据文件
        # lists = []
        for item in file_names:
            temp = [os.path.join(item, i) for i in os.listdir(os.path.join(self.root, item))]
            if len(temp) < 5000:
                continue
            # lists.append((len(temp), item))
            # print(f"{item}:{len(temp)}")
            # print(temp)
            temp_train = temp[: int(len(temp) * 0.8)]  # 就取一点来训练，另外的用来测试
            temp_test = temp[int(len(temp) * 0.8):]
            # temp_train = temp  # 就取一点来训练，另外的用来测试
            # temp_test = temp
            self.train_sample += temp_train
            self.test_sample += temp_test

        # lists = sorted(lists, key=lambda x: -x[0])
        # for item in lists:
        #     print(f"{item[1]}:{item[0]}")
        # 在将合并的文件进行打乱
        random.shuffle(self.train_sample)
        random.shuffle(self.test_sample)

        if mode == "train":
            self.sample = self.train_sample
        else:
            self.sample = self.test_sample
        # print(len(self.sample))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        df = open(os.path.join(self.root, self.sample[idx]), encoding="utf-8")
        data = json.load(df)
        featrues = get_feature_global(data)
        return featrues, torch.tensor([data["label"]], dtype=torch.float)

if __name__ == '__main__':
    # root = "/home/tonnn/xiu/data/discard"
    # ds_train = DatasetDiscard(root, "train", action_to_id)
    # ds_test = DatasetDiscard(root, "test", action_to_id)
    # data_train = DataLoader(ds_train, batch_size=100)
    # print(data_train.__len__())
    # for x, targets in data_train:
    #     print(x.shape)
    #     print(targets)
    # reply_m = ReplayMemory(1000, 0.8)
    # reply_m.sample(100)

    root = "/home/tonnn/.nas/xy/output/globel"
    ds_train = DatasetGlobal(root, "train")
    ds_test = DatasetGlobal(root, "test")
    data_train = DataLoader(ds_train, batch_size=100)
    # print(len(data_train))
    for x, targets in data_train:
        print(x.shape)
        print(targets)
        exit(0)
