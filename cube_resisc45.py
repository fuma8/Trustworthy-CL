
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
from torcheval.metrics.functional import multiclass_auroc

rows = 4 #サンプリングマトリックスの行数
cols = 4 #サンプリングマトリックスの列数
M =  rows*cols#生成されるビデオのフレーム数
blk_size = cols
sampling_rate = M // (rows*cols)
cube_size = 256 // cols
# 平均と標準偏差を指定
mean = 0.0
std_dev = 1.0
batch_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PhiR = torch.rand(M, rows * cols)
PhiB = torch.rand(M, rows * cols)
PhiG = torch.rand(M, rows * cols)
# PhiR = torch.where(PhiR > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
# PhiB = torch.where(PhiB > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
# PhiG = torch.where(PhiG > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
PhiWeightR = PhiR.unsqueeze(1).to(device)
PhiWeightB = PhiB.unsqueeze(1).to(device)
PhiWeightG = PhiG.unsqueeze(1).to(device)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

data_dir = '/home/kimishima/pytorch-classification-uncertainty/data/RESISC45'
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
num_classes = 45
input_channel = 3
# model = ResNet18(num_classes, input_channels = input_channel)
model = ViT('B_16_imagenet1k', pretrained=True, in_channels = input_channel, image_size = cube_size, num_classes = num_classes)
# summary(model=model, input_size=(batch_size*M, 3, cube_size, cube_size))
# input()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0,1])
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = 1e-2,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
transform_cube_train = transforms.Compose([
        transforms.RandomResizedCrop(cube_size),
        transforms.RandomHorizontalFlip(),
    ])
transform_cube_val = transforms.Compose([
        transforms.CenterCrop(cube_size),
        transforms.RandomHorizontalFlip(),
    ])

def train(epoch, M, PhiR, PhiG, PhiB):
    print('\nEpoch: %d' % epoch)
    start = time.time()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    num_class = 45
    correct_uncertainty = 0
    auc_score = 0
    cnt = 0
    sampling_rate = M // (rows * cols)
    I = torch.eye(int(sampling_rate*rows*cols)).to(device)
    criterion = F.mse_loss
    gamma1 = torch.Tensor([0.001]).to(device)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets_ = targets.to(device)
        optimizer.zero_grad()
        PhiWeightR = PhiR.contiguous().view(M, 1, rows, cols).to(device)
        PhiWeightG = PhiG.contiguous().view(M, 1, rows, cols).to(device)
        PhiWeightB = PhiB.contiguous().view(M, 1, rows, cols).to(device)
        cube_r = F.conv2d(inputs[:, 0:1, :, :], PhiWeightR, padding=0, stride=blk_size, bias=None)
        cube_g = F.conv2d(inputs[:, 1:2, :, :], PhiWeightG, padding=0, stride=blk_size, bias=None)
        cube_b = F.conv2d(inputs[:, 2:3, :, :], PhiWeightB, padding=0, stride=blk_size, bias=None)
        # img = inputs[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
        # plt.figure(figsize=(10, 4))
        # plt.subplot(131)
        # plt.imshow(img, cmap="gray")
        # plt.savefig("/home/kimishima/pytorch-classification-uncertainty/images/RESISC45_original.jpg")
        for i in range(M):
            cnt += 1
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
            y_ = one_hot_embedding(targets, num_classes)
            y_ = y_.to(device)
            # image = reconstructed_cube[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy() #ミニバッチに含まれる一つの画像を表示する
            # plt.figure(figsize=(10, 4))
            # plt.subplot(131)
            # plt.imshow(image)
            # plt.savefig("/home/kimishima/pytorch-classification-uncertainty/images/RESISC45_cube_"+str(i)+".jpg")
            # exit()
            reconstructed_cube = transform_cube_train(reconstructed_cube)
            outputs = model(reconstructed_cube)
            phi_consR = torch.mm(PhiR, PhiR.t()).to(device)
            phi_consG = torch.mm(PhiG, PhiG.t()).to(device)
            phi_consB = torch.mm(PhiB, PhiB.t()).to(device)
            loss = ce_loss(targets_, outputs, num_class, epoch, 10, device)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets_.size(0)
            correct += predicted.eq(targets_).sum().item()
            auc_score += multiclass_auroc(input = outputs, target = targets_, num_classes = num_class, average = "macro").item() * 100
            evidence = exp_evidence(outputs)
            alpha = evidence + 1
            uncertainty = num_class / torch.sum(alpha, dim=1, keepdim=True)
            mean_uncertainty = torch.mean(uncertainty)
            correct_uncertainty += mean_uncertainty.item()
            progress_bar(cnt-1, len(train_loader)*M, 'Loss: %.3f | Acc: %.3f%% (%d/%d) | AUC: %.3f'
                        % (train_loss/(cnt), 100.*correct/total, correct, total, auc_score/cnt))
        loss.backward()
        optimizer.step()
    
    total_accuracy = correct / total
    total_uncertainty = correct_uncertainty / cnt
    total_auc  = auc_score / cnt
    return total_accuracy, total_uncertainty, total_auc, PhiR, PhiG, PhiB

        

def test(epoch, M, PhiR, PhiG, PhiB, flag):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_class = 45
    best_auc = 0
    correct_uncertainty = 0
    auc_score = 0
    cnt = 0
    sampling_rate = M // (rows * cols)
    I = torch.eye(M).to(device)
    criterion = F.mse_loss
    gamma1 = torch.Tensor([0.001]).to(device)
    PhiR = PhiR[:M, :]
    PhiB = PhiB[:M, :]
    PhiG = PhiG[:M, :]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets_ = targets.to(device)
            PhiWeightR = PhiR.contiguous().view(M, 1, rows, cols).to(device)
            PhiWeightG = PhiG.contiguous().view(M, 1, rows, cols).to(device)
            PhiWeightB = PhiB.contiguous().view(M, 1, rows, cols).to(device)
            cube_r = F.conv2d(inputs[:, 0:1, :, :], PhiWeightR, padding=0, stride=blk_size, bias=None)
            cube_g = F.conv2d(inputs[:, 1:2, :, :], PhiWeightG, padding=0, stride=blk_size, bias=None)
            cube_b = F.conv2d(inputs[:, 2:3, :, :], PhiWeightB, padding=0, stride=blk_size, bias=None)
            for i in range(M):
                cnt += 1
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
                y_ = one_hot_embedding(targets, num_classes)
                y_ = y_.to(device)
                reconstructed_cube = transform_cube_val(reconstructed_cube)
                outputs = model(reconstructed_cube)
                phi_consR = torch.mm(PhiR, PhiR.t()).to(device)
                phi_consG = torch.mm(PhiG, PhiG.t()).to(device)
                phi_consB = torch.mm(PhiB, PhiB.t()).to(device)
                loss = ce_loss(targets_, outputs, num_class, epoch, 10, device)
                #loss = ce_loss(targets_, outputs, num_class, epoch, 10, device) + torch.mul(criterion(phi_consR, I), gamma1) + torch.mul(criterion(phi_consG, I), gamma1) + torch.mul(criterion(phi_consB, I), gamma1)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets_.size(0)
                correct += predicted.eq(targets_).sum().item()
                _, preds = torch.max(outputs, 1)
                match = torch.reshape(torch.eq(preds, targets_).float(), (-1, 1))
                acc = torch.mean(match)
                evidence = exp_evidence(outputs)
                alpha = evidence + 1
                auc_score += multiclass_auroc(input = outputs, target = targets_, num_classes = num_class, average = "macro").item() * 100
                uncertainty = num_class / torch.sum(alpha, dim=1, keepdim=True)
                mean_uncertainty = torch.mean(uncertainty)
                correct_uncertainty += mean_uncertainty.item()
                progress_bar(cnt, len(val_loader)*M, 'Loss: %.3f | Acc: %.3f%% (%d/%d) | AUC: %.3f'
                            % (test_loss/(cnt), 100.*correct/total, correct, total, auc_score/cnt))

    # Save checkpoint.
    acc = 100.*correct/total
    total_acc = correct/total
    total_auc = auc_score/cnt
    total_uncertainty = correct_uncertainty / cnt
    if flag == True and acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'auc': total_auc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        pth_path = './checkpoint/RESISC45_blk_size_'+str(blk_size)+'_learnable_matrix_proposed_loss.pth'
        dict_path = './checkpoint/RESISC45_blk_size_'+str(blk_size)+'_learnable_matrix_proposed_loss_dict.pth'
        torch.save(model.state_dict(), pth_path)
        torch.save(state, dict_path)
        best_acc = acc
        best_auc = total_auc
    print("Best Accuracy:", best_acc)
    print("Best AUC:", best_auc)
    return total_acc, total_uncertainty, total_auc

train_acc_list = []
train_uncertainty_list = []
test_acc_list = []
test_uncertainty_list = []
train_auc_list = []
test_auc_list = []
for epoch in range(total_epoch):
    train_acc, train_u, train_auc, trained_PhiR, trained_PhiG, trained_PhiB = train(epoch+1, M, PhiR, PhiG, PhiB)
    test_acc, test_u, test_auc = test(epoch+1, M, trained_PhiR, trained_PhiG, trained_PhiB, True)
    PhiR = trained_PhiR
    PhiG = trained_PhiG
    PhiB = trained_PhiB
    train_acc_list.append(train_acc)
    train_uncertainty_list.append(train_u)
    test_acc_list.append(test_acc)
    test_uncertainty_list.append(test_u)
    train_auc_list.append(train_auc)
    test_auc_list.append(test_auc)
    scheduler.step()

acc_dic = {}
acc_dic["train"] = train_acc_list
acc_dic["test"] = test_acc_list
torch.save(acc_dic, "/home/kimishima/pytorch-classification-uncertainty/learnable_matrix_RESISC45_acc_blk_"+str(blk_size)+".pkl")
uncertainty_dic = {}
uncertainty_dic["train"] = train_uncertainty_list
uncertainty_dic["test"] = test_uncertainty_list
torch.save(uncertainty_dic, "/home/kimishima/pytorch-classification-uncertainty/learnable_matrix_RESISC45_uncertainty_blk_"+str(blk_size)+".pkl")
auc_dic = {}
auc_dic["train"] = train_auc_list
auc_dic["test"] = test_auc_list
torch.save(auc_dic, "/home/kimishima/pytorch-classification-uncertainty/learnable_matrix_RESISC45_auc_blk_"+str(blk_size)+".pkl")

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
plt.savefig("/home/kimishima/pytorch-classification-uncertainty/images/learnable_matrix_RESISC45_acc_uncertainty_proposed_loss_blk_"+str(blk_size)+".jpg")
plt.show()

pth_path = './checkpoint/RESISC45_blk_size_'+str(blk_size)+'_learnable_matrix_proposed_loss.pth'
model.load_state_dict(torch.load(pth_path))
model = model.to(device)
model.eval()

cs_result_file = open("/home/kimishima/pytorch-classification-uncertainty/learnable_matrix_RESISC45_result_blk_"+str(blk_size)+".txt", "w")
for i in range(1, M+1):
    print("SR=", i / M)
    acc_cs, u_cs, auc_cs = test(1, i, trained_PhiR, trained_PhiG, trained_PhiB, False)
    cs_result_file.write(f"SR {i / M}: Accuracy {acc_cs:.4f} Mean_Uncertainty {u_cs:.4f} AUC {auc_cs:.4f}\n")
