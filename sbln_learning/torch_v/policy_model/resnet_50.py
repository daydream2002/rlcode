from torch import nn
import torch
import torch.functional as F

class BaseLay(nn.Module):
    def __init__(self, in_channels):
        super(BaseLay, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, intputs):
        return self.lay(intputs) + intputs


class BaseLayWithNB(nn.Module):
    def __init__(self, in_channels):
        super(BaseLayWithNB, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, intputs):
        return self.lay(intputs) + intputs

class BaseLayWithTanh(nn.Module):
    def __init__(self, in_channels):
        super(BaseLayWithTanh, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            # nn.BatchNorm2d(in_channels),
        )

    def forward(self, intputs):
        return self.lay(intputs) + intputs


class ResNet50(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        self.device = torch.device("cuda")
        base_lay = BaseLay(256)
        lays = []
        for i in range(50):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        self.final_lay = nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, inputs):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        outputs = self.final_lay(outputs)
        # print(outputs.shape)
        return outputs.view(outputs.shape[0], 34)

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs.argmax(dim=1)

    def predict_with_mask(self, inputs, mask):
        outputs = self.forward(inputs)
        # print(outputs)
        # print(mask)
        # exit(0)
        # outputs = torch.where(mask, outputs, torch.tensor(-1e4, dtype=torch.float).to(self.device))
        for idx, item in enumerate(mask):
            outputs[idx][item == 0] = -100000.0
        return outputs.argmax(dim=1)


class ResNet50WithNB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        base_lay = BaseLayWithNB(256)
        lays = []
        for i in range(50):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        self.nb = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.final_lay = nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, inputs):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        outputs = self.nb(outputs)
        outputs = self.relu(outputs)
        outputs = self.final_lay(outputs)

        # print(outputs)
        # print(outputs.shape)
        return outputs.view(outputs.shape[0], 34)

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs.argmax(dim=1)

    def predict_with_mask(self, inputs, mask):
        outputs = self.forward(inputs)
        for idx, item in enumerate(mask):
            outputs[idx][item == 0] = -1000.0
        return outputs.argmax(dim=1)


class ResNet50WithTanh(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        base_lay = BaseLayWithTanh(256)
        lays = []
        for i in range(50):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        # self.nb = nn.BatchNorm2d(out_channels)
        self.final_lay = nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, inputs):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        outputs = self.final_lay(outputs)
        # outputs = self.nb(outputs)
        # print(outputs)
        # print(outputs.shape)
        return outputs.view(outputs.shape[0], 34)

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs.argmax(dim=1)

    def predict_with_mask(self, inputs, mask):
        outputs = self.forward(inputs)
        for idx, item in enumerate(mask):
            outputs[idx][item == 0] = -1000.0
        return outputs.argmax(dim=1)


class ResNet50WithLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature_extract = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        base_lay = BaseLay(256)
        lays = []
        for i in range(50):
            lays.append(base_lay)
        self.lay = nn.Sequential(*lays)
        # self.nb = nn.BatchNorm2d(out_channels)
        self.final_lay = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 34 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
        )

    def forward(self, inputs):
        outputs = self.feature_extract(inputs)
        outputs = self.lay(outputs)
        outputs = self.final_lay(outputs)
        # outputs = self.nb(outputs)
        # print(outputs)
        # print(outputs.shape)
        return outputs

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs.argmax(dim=1)

    def predict_with_mask(self, inputs, mask):
        outputs = self.forward(inputs)
        for idx, item in enumerate(mask):
            outputs[idx][item == 0] = -1000.0
        return outputs.argmax(dim=1)

if __name__ == '__main__':
    x = torch.randint(2, (100, 1330, 34, 1), dtype=torch.float)
    resnet50 = ResNet50WithNB(1330, 1)
    print(resnet50(x).shape)
    # print(resnet50.predict(x))
