import torch
import torch.nn as nn

from sbln_learning.torch_v.policy_model.resnet_50 import BaseLay

device = torch.device("cuda")


class DoubleGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(DoubleGRUModel, self).__init__()
        self.device = device
        self.state0 = torch.ones([1, 1, hidden_size], requires_grad=True).to(self.device)
        self.state1 = torch.ones([1, 1, hidden_size], requires_grad=True).to(self.device)

        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Flatten()
        self.relu = nn.ReLU()
        self.Linear1 = nn.Linear(in_features=hidden_size, out_features=128)
        self.Linear2 = nn.Linear(in_features=128, out_features=1)

    def cul(self, input, len_size):
        pass

    def forward(self, input, len_size):
        bitch_output = []
        # print(input.shape)
        # print(len_size)
        for idx, item in enumerate(input):
            state0 = self.state0
            state1 = self.state1
            out1 = None
            for it in range(int(len_size[idx].item())):
                tmp = item[it].view(1, 1, item.shape[1])
                out0, state0 = self.gru1(tmp, state0)
                out1, state1 = self.gru2(out0, state1)
            out1 = self.fc(out1)
            out1 = self.Linear1(out1)
            out1 = self.Linear2(out1)
            bitch_output.append([out1.item()])

        return torch.tensor(bitch_output, dtype=torch.float, requires_grad=True).to(self.device)


class DoubleSimpleGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(DoubleSimpleGRUModel, self).__init__()
        # self.state0 = torch.ones([1, 1, hidden_size], requires_grad=True).to(self.device)
        # self.state1 = torch.ones([1, 1, hidden_size], requires_grad=True).to(self.device)
        self.device = device
        # self.simpleGRU = nn.Sequential(
        #     nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        #     nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        # )
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Flatten()
        self.relu = nn.ReLU()
        self.Linear1 = nn.Linear(in_features=30 * hidden_size, out_features=128)
        self.Linear2 = nn.Linear(in_features=128, out_features=1)
        self.dropout = nn.Dropout()

    def cul(self, input, len_size):
        pass

    def forward(self, inputs):
        outputs, _ = self.gru1(inputs)
        outputs, _ = self.gru2(outputs)
        # print(outputs)
        # exit(1)
        outputs = self.fc(outputs)
        # print(outputs.shape)
        outputs = self.Linear1(outputs)
        # outputs = self.dropout(outputs)
        outputs = self.Linear2(outputs)
        return outputs


class CNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(CNNModel, self).__init__()
        self.device = device
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        base_lay = BaseLay(256)
        lays = []
        for i in range(101):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        self.fc = nn.Flatten()
        self.relu = nn.ReLU()
        self.Linear1 = nn.Linear(in_features=256 * 34, out_features=512)
        self.Linear2 = nn.Linear(in_features=512, out_features=out_channels)

    def forward(self, inputs):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        outputs = self.fc(outputs)
        # print(outputs)
        # exit(1)
        # print(outputs.shape)
        outputs = self.Linear1(outputs)
        outputs = self.Linear2(outputs)
        return outputs


if __name__ == '__main__':
    x = torch.randint(2, (1, 1330, 34, 1), dtype=torch.float).to(device)
    # target = torch.randint(2, (10,), dtype=torch.int8)

    # model = DoubleSimpleGRUModel(1330 * 34, 512, device).to(device)
    model = CNNModel(1330, 1, device).to(device)
    # print(model)
    # len_size = torch.randint(6, 9, (100,), dtype=torch.int8).to(device)
    print(model(x)[0].item())
