
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from losses import *
from helpers import get_device, rotate_img, one_hot_embedding
import copy
from model import *
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from utils import progress_bar
from pytorch_pretrained_vit import ViT
from torchinfo import summary

start_time = time.time()
rows = 4 #サンプリングマトリックスの行数
cols = 4 #サンプリングマトリックスの列数
M =  rows*cols#生成されるビデオのフレーム数
blk_size = cols
sampling_rate = M // (rows*cols)
cube_size = 384 // cols
# 平均と標準偏差を指定
mean = 0.0
std_dev = 1.0
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PhiR = torch.rand(M, rows, cols)
PhiB = torch.rand(M, rows, cols)
PhiG = torch.rand(M, rows, cols)
PhiR = torch.where(PhiR > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
PhiB = torch.where(PhiB > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
PhiG = torch.where(PhiG > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
PhiWeightR = PhiR.unsqueeze(1).to(device)
PhiWeightB = PhiB.unsqueeze(1).to(device)
PhiWeightG = PhiG.unsqueeze(1).to(device)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

data_dir = '/home/kimishima/pytorch-classification-uncertainty/data/CUB'
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

dataset_size = len(dataset)
train_ratio = 0.7
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size)
dataloaders = {
    "train": train_loader,
    "val": val_loader,
}
best_acc = 0
total_epoch = 100
num_classes = 200
input_channel = 3
# model = ResNet18(num_classes, input_channels = input_channel)
model = ViT('B_16_imagenet1k', pretrained=True, in_channels = input_channel, image_size = cube_size, num_classes = num_classes)
# summary(model=model, input_size=(batch_size*M, 3, cube_size, cube_size))
# input()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0,1,2])
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = 1e-3,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
transform_cube_train = transforms.Compose([
        transforms.RandomResizedCrop(cube_size, scale=(0.8, 0.8)),
        transforms.RandomHorizontalFlip(),
    ])
transform_cube_val = transforms.Compose([
        transforms.RandomResizedCrop(cube_size, scale=(0.8, 0.8)),
        transforms.RandomHorizontalFlip(),
    ])
# for inputs, targets in train_loader:
#     img = inputs[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
#     plt.figure(figsize=(10, 4))
#     plt.subplot(131)
#     plt.imshow(img, cmap="gray")
#     plt.savefig("/home/19x3039_kimishima/pytorch-classification-uncertainty/images/scene15.jpg")
#     exit()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    loss_func_kd = KnowledgeDistillationLoss()
    train_loss = 0
    correct = 0
    total = 0
    num_class = 200
    correct_uncertainty = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets_ = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        reconstructed_cube_all = torch.randn(0, 3, cube_size, cube_size).to(device)
        y = torch.zeros(0, num_classes).to(device)
        targets = torch.zeros(0).int().to(device)
        cube_r = F.conv2d(inputs[:, 0:1, :, :], PhiWeightR, padding=0, stride=blk_size, bias=None)
        cube_g = F.conv2d(inputs[:, 1:2, :, :], PhiWeightG, padding=0, stride=blk_size, bias=None)
        cube_b = F.conv2d(inputs[:, 2:3, :, :], PhiWeightB, padding=0, stride=blk_size, bias=None)
        # img = inputs[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
        # plt.figure(figsize=(10, 4))
        # plt.subplot(131)
        # plt.imshow(img, cmap="gray")
        # plt.savefig("/home/kimishima/pytorch-classification-uncertainty/images/caltech101_original.jpg")
        for i in range(M):
            max_r = torch.max(cube_r[:,i,:,:])
            min_r = torch.min(cube_r[:,i,:,:])
            normarized_cube_r = (cube_r[:,i,:,:] - min_r) / (max_r - min_r)
            max_b = torch.max(cube_b[:,i,:,:])
            min_b = torch.min(cube_b[:,i,:,:])
            normarized_cube_b = (cube_b[:,i,:,:] - min_b) / (max_b - min_b)
            max_g = torch.max(cube_g[:,i,:,:])
            min_g = torch.min(cube_g[:,i,:,:])
            normarized_cube_g = (cube_g[:,i,:,:] - min_g) / (max_g - min_g)
            reconstructed_cube = torch.stack([normarized_cube_r, normarized_cube_g, normarized_cube_b], dim = 1)
            reconstructed_cube_all = torch.cat([reconstructed_cube_all, reconstructed_cube], dim = 0)
            y_ = one_hot_embedding(targets_, num_classes)
            y_ = y_.to(device)
            y = torch.cat([y, y_], dim = 0)
            targets = torch.cat([targets, targets_], dim = 0)
            # image = reconstructed_cube[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy() #ミニバッチに含まれる一つの画像を表示する
            # plt.figure(figsize=(10, 4))
            # plt.subplot(131)
            # plt.imshow(image)
            # plt.savefig("/home/kimishima/pytorch-classification-uncertainty/images/caltech101_cube_"+str(i)+".jpg")
            # exit()
        reconstructed_cube_all = transform_cube_train(reconstructed_cube_all)
        # outputs_1, outputs_u_max, outputs = model(reconstructed_cube_all)
        outputs = model(reconstructed_cube_all)
        loss = ce_loss(targets, outputs, num_class, epoch, 10, device)
        #loss = edl_mse_loss(outputs, y.float(), epoch, num_class, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
        # loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        _, preds = torch.max(outputs, 1)
        match = torch.reshape(torch.eq(preds, targets).float(), (-1, 1))
        acc = torch.mean(match)
        # evidence = relu_evidence(outputs)
        evidence = exp_evidence(outputs)
        alpha = evidence + 1
        uncertainty = num_class / torch.sum(alpha, dim=1, keepdim=True)
        mean_uncertainty = torch.mean(uncertainty)
        correct_uncertainty += mean_uncertainty.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    total_accuracy = correct / total
    total_uncertainty = correct_uncertainty / len(train_loader)
    return total_accuracy, total_uncertainty

        

def test(epoch, M, flag):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    loss_func_kd = KnowledgeDistillationLoss()
    total = 0
    num_class = 200
    correct_uncertainty = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets_ = inputs.to(device), targets.to(device)
            reconstructed_cube_all = torch.randn(0, 3, cube_size, cube_size).to(device)
            y = torch.zeros(0, num_classes).to(device)
            targets = torch.zeros(0).int().to(device)
            cube_r = F.conv2d(inputs[:, 0:1, :, :], PhiWeightR, padding=0, stride=blk_size, bias=None)
            cube_g = F.conv2d(inputs[:, 1:2, :, :], PhiWeightG, padding=0, stride=blk_size, bias=None)
            cube_b = F.conv2d(inputs[:, 2:3, :, :], PhiWeightB, padding=0, stride=blk_size, bias=None)
            for i in range(M):
                max_r = torch.max(cube_r[:,i,:,:])
                min_r = torch.min(cube_r[:,i,:,:])
                normarized_cube_r = (cube_r[:,i,:,:] - min_r) / (max_r - min_r)
                max_b = torch.max(cube_b[:,i,:,:])
                min_b = torch.min(cube_b[:,i,:,:])
                normarized_cube_b = (cube_b[:,i,:,:] - min_b) / (max_b - min_b)
                max_g = torch.max(cube_g[:,i,:,:])
                min_g = torch.min(cube_g[:,i,:,:])
                normarized_cube_g = (cube_g[:,i,:,:] - min_g) / (max_g - min_g)
                reconstructed_cube = torch.stack([normarized_cube_r, normarized_cube_g, normarized_cube_b], dim = 1)
                reconstructed_cube_all = torch.cat([reconstructed_cube_all, reconstructed_cube], dim = 0)
                y_ = one_hot_embedding(targets_, num_classes)
                y_ = y_.to(device)
                y = torch.cat([y, y_], dim = 0)
                targets = torch.cat([targets, targets_], dim = 0)
            reconstructed_cube_all = transform_cube_val(reconstructed_cube_all)
            # outputs_1, outputs_u_max, outputs = model(reconstructed_cube_all)
            outputs = model(reconstructed_cube_all)
            loss = ce_loss(targets, outputs, num_class, epoch, 10, device)
            #loss = edl_mse_loss(outputs, y.float(), epoch, num_class, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
            #loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
            # loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + proposed_kd_loss(outputs_u_max, y, param=3)
            #loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + proposed_kd_loss(relu_evidence(outputs), y, param=3) + loss_func_kd(outputs_u_max, y.float(), outputs)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            _, preds = torch.max(outputs, 1)
            match = torch.reshape(torch.eq(preds, targets).float(), (-1, 1))
            acc = torch.mean(match)
            # evidence = relu_evidence(outputs)
            evidence = exp_evidence(outputs)
            alpha = evidence + 1
            uncertainty = num_class / torch.sum(alpha, dim=1, keepdim=True)
            mean_uncertainty = torch.mean(uncertainty)
            correct_uncertainty += mean_uncertainty.item()
            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    total_acc = correct/total
    total_uncertainty = correct_uncertainty / len(val_loader)
    if flag == True and acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        pth_path = './checkpoint/CUB_blk_size_'+str(blk_size)+'_proposed_loss.pth'
        torch.save(model.state_dict(), pth_path)
        best_acc = acc
    print("Best Accuracy:", best_acc)
    return total_acc, total_uncertainty
    
train_acc_list = []
train_uncertainty_list = []
test_acc_list = []
test_uncertainty_list = []
for epoch in range(total_epoch):
    train_acc, train_u = train(epoch+1)
    test_acc, test_u = test(epoch+1, M, True)
    train_acc_list.append(train_acc)
    train_uncertainty_list.append(train_u)
    test_acc_list.append(test_acc)
    test_uncertainty_list.append(test_u)
    scheduler.step()

acc_dic = {}
acc_dic["train"] = train_acc_list
acc_dic["test"] = test_acc_list
torch.save(acc_dic, "/home/kimishima/pytorch-classification-uncertainty/CUB_acc_blk_"+str(blk_size)+".pkl")
uncertainty_dic = {}
uncertainty_dic["train"] = train_uncertainty_list
uncertainty_dic["test"] = test_uncertainty_list
torch.save(uncertainty_dic, "/home/kimishima/pytorch-classification-uncertainty/CUB_uncertainty_blk_"+str(blk_size)+".pkl")

# グラフの作成
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
x_value = list(range(1, len(train_acc_list)+1))
# Train Accuracyのプロット
axes[0].plot(x_value, train_acc_list, marker = "o", label='Train Accuracy')
axes[0].plot(x_value, test_acc_list, marker = "x", label='Test Accuracy')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Validation Accuracyのプロット
axes[1].plot(x_value, train_uncertainty_list, marker = "o", label='Train Uncertainty')
axes[1].plot(x_value, test_uncertainty_list, marker = "x", label='Test Uncertainty')
axes[1].set_title('Uncertainty')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Uncertainty')
axes[1].legend()

# 画像の保存と表示
plt.tight_layout()
plt.savefig("/home/kimishima/pytorch-classification-uncertainty/images/CUB_acc_uncertainty_proposed_loss_blk_"+str(blk_size)+".jpg")
plt.show()

pth_path = './checkpoint/CUB_blk_size_'+str(blk_size)+'_proposed_loss.pth'
model.load_state_dict(torch.load(pth_path))
model = model.to(device)
model.eval()

cs_result_file = open("/home/kimishima/pytorch-classification-uncertainty/CUB_result_blk_"+str(blk_size)+".txt", "w")
for i in range(1, M+1):
    print("SR=", i / M)
    acc_cs, u_cs = test(1, i, False)
    cs_result_file.write(f"SR {i / M}: Accuracy {acc_cs:.4f} Mean_Uncertainty {u_cs:.4f}\n")
