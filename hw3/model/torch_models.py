import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch

class DQNCNN(nn.Module):
    def __init__(self, num_actions):

        super(DQNCNN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(in_channels=4, 
                                             out_channels=32, 
                                             kernel_size=8, 
                                             stride=1, 
                                             padding=1))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(4, 4))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(in_channels=32, 
                                             out_channels=64, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(in_channels=64, 
                                             out_channels=128, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1))
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, num_actions))
        self.layer4 = layer4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fc_input = x.view(x.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

class SimpleDQN(nn.Module):
    def __init__(self, num_actions, in_channels=4):
        super(SimpleDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
