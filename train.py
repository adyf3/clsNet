import os
import time
import torch
import torchvision
from torch import nn, optim
from datas import transfer_data
from utils.config import opt
from enum import Enum
import logging 
from loss import *

# class Model(Enum):
#     vgg16 = 1
#     vgg19 = 2
#     resnet18 = 3
#     resnet50 = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# make datasets through files by using torchvision.datasets.ImageFolder
train_dataset = torchvision.datasets.ImageFolder(
    root='D:\scripts/data/train/',
    transform=transfer_data.transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=4)
val_dataset = torchvision.datasets.ImageFolder(
    root='D:\scripts/data/val/',
    transform=transfer_data.transform)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=4)

# logging.basiConfig(level = logging.INFO,filename = 'log.txt',filemode = 'a')
# logger = logging.getLogger(__name__)
# logger.info('start computing!')
# logger.info('Finish!')

def train():
    print(f'Train numbers:{len(train_dataset)}')
    model = torchvision.models.resnet50(pretrained=True).to(device)
    #freeze the paras before classififer
    for param in model_conv.parameters():
	    param.requires_grad = False

    model.fc = nn.Linear(2048, 2).to(device)  # 2048
    logger.info('start computing!')
    logger.info(model)
    #loss & optim
    #cost = torch.nn.CrossEntropyLoss().to(device)
    cost = FocalLoss(class_num = 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-8)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    #余弦退火
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=5, eta_min=4e-08)


    for epoch in range(1, opt.epochs + 1):
        print('Start epoch:%d' % epoch)
        model.train()
        start = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            images = inputs.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = cost(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
#             optimizer.step()
        lr_scheduler.step()
        if(epoch % opt.display_epoch == 0):
            end = time.time()
            print(f"Epoch: [{epoch}/{opt.epochs}],"
                  f"Loss:{loss.item():8f},"
                  f"Time:{(end-start)*opt.display_epoch:1f}sec!")
            model.eval()

            correct_prediction = 0
            total = 0
            for i, (inputs, labels) in enumerate(val_loader):
                images = inputs.to(device)
                labels = labels.to(device)
                # print prediction
                outputs = model(images)
                # equal prediction and acc
                _, predicted = torch.max(outputs.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()
                accuracy = correct_prediction / total
#                 break;
            print(f"Acc:{(correct_prediction / total):4f}")
            path = opt.save_model_path + str(accuracy) + '.pth'
            torch.save(model, path)
    print("Save model to{path}.format(path)")


if __name__ == '__main__':
    logging.basicConfig(level = logging.NOTSET)
    logger = logging.getLogger()
    torch.cuda.set_device(0)
    train()
