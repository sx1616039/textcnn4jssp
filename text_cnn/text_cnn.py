import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, hight, width, input_channel=3, out_channel=6, out_num=6):
        super(Actor, self).__init__()
        self.h = hight
        self.w = width
        self.feature1 = nn.Sequential(nn.Conv2d(input_channel, out_channel, kernel_size=(5, 5), padding=2), nn.ReLU())
        self.fc2 = nn.Linear(self.h * self.w * out_channel, self.h * self.w)
        self.action_head = nn.Linear(self.h * self.w, out_num)

    def forward(self, x):
        x = self.feature1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, hight, width, input_channel=3, out_channel=6, out_num=1):
        super(Critic, self).__init__()
        self.h = hight
        self.w = width
        self.feature1 = nn.Sequential(nn.Conv2d(input_channel, out_channel, kernel_size=(5, 5), padding=2), nn.ReLU())
        self.fc2 = nn.Linear(self.h * self.w * out_channel, self.h * self.w)
        self.state_value = nn.Linear(self.h * self.w, out_num)

    def forward(self, x):
        x = self.feature1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc2(x))
        state_value = self.state_value(x)
        return state_value
