import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from losses import relu_evidence
import numpy as np
import torchvision.models as models
import torchvision

class LeNet(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.use_dropout = dropout
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(28800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear1 = nn.Linear(64*block.expansion, num_classes)
        self.linear2 = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x.float())))
        out = self.layer1(out)
        out1 = nn.AdaptiveAvgPool2d((1,1))(out)
        out1 = out1.reshape(out1.size(0), -1)
        out1 = self.linear1(out1)
        u1 = calc_uncertainty(out1, self.num_classes)
        out = self.layer2(out)
        out = self.layer3(out)
        # check the uncertainty after layer3
        out2 = nn.AdaptiveAvgPool2d((1,1))(out)
        out2 = out2.reshape(out2.size(0), -1)
        out2 = self.linear2(out2)
        u2 = calc_uncertainty(out2, self.num_classes)

        out = self.layer4(out)
        # # out = F.avg_pool2d(out, 16)
        out = nn.AdaptiveAvgPool2d((1,1))(out)
        out = out.reshape(out.size(0), -1) #out = out.view((out.size(0), -1))
        out = self.linear(out)
        u_l = calc_uncertainty(out, self.num_classes)
        return out1, out2, out

class ResNet18_pretrained(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet18_pretrained, self).__init__()
        self.linear1 = nn.Linear(64, num_classes)
        self.linear2 = nn.Linear(128, num_classes)
        self.linear3 = nn.Linear(256, num_classes)
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
    
    def forward(self, x):
        arr_dict = {}
        outputs = self.model(x)
        layer1_output = self.model.layer1(self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x)))))
        out_1 = nn.AdaptiveAvgPool2d((1,1))(layer1_output)
        out_1 = out_1.reshape(out_1.size(0), -1) #out = out.view((out.size(0), -1))
        # out_1 = out_1.to(device)
        out_1 = self.linear1(out_1)
        u_1 = calc_uncertainty(out_1, self.num_classes)
        arr_dict[u_1] = out_1
        layer2_output = self.model.layer2(self.model.layer1(self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))))
        out_2 = nn.AdaptiveAvgPool2d((1,1))(layer2_output)
        out_2 = out_2.reshape(out_2.size(0), -1) #out = out.view((out.size(0), -1))
        # out_2 = out_2.to(device)
        out_2 = self.linear2(out_2)
        u_2 = calc_uncertainty(out_2, self.num_classes)
        arr_dict[u_2] = out_2
        layer3_output = self.model.layer3(self.model.layer2(self.model.layer1(self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x)))))))
        out_3 = nn.AdaptiveAvgPool2d((1,1))(layer3_output)
        out_3 = out_3.reshape(out_3.size(0), -1) #out = out.view((out.size(0), -1))
        # out_3 = out_3.to(device)
        out_3 = self.linear3(out_3)
        u_3 = calc_uncertainty(out_3, self.num_classes)
        arr_dict[u_3] = out_3
        u_max = max(u_1, u_2, u_3)
        outputs_u_max = arr_dict[u_max]
        return outputs_u_max, outputs

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def calc_uncertainty(outputs, num_classes):
        _, preds = torch.max(outputs, 1)
        evidence = relu_evidence(outputs)
        alpha = evidence + 1
        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        return u