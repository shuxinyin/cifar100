import os
import time
import pickle
import argparse

import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

from data_set import Cifar100
from cifarmodel import Cifar100Net

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=2, help='epoch size')
parser.add_argument('--step_size', type=int, default=5, help='step size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
parser.add_argument('--dir_path', type=str, default='../data/cifar-100-data/', help='dir_path')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dir_path = args.dir_path  # data path

total, correct = 0, 0
model = Cifar100Net().cuda()

train_dataset = Cifar100(dir_path, train=True)
test_dataset = Cifar100(dir_path, train=False)

train_dataloder = DataLoader(train_dataset, batch_size=args.batch_size,
                             num_workers=0)
test_dataloder = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=0)

loss_func = CrossEntropyLoss()
opt = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8, eta_min=0)


def model_eval(model, test_acc_list):
    model.eval()
    total_num, total_correct = 0, 0
    for i, (image, label) in enumerate(test_dataloder):
        image, label = image.cuda(), label.cuda()
        model_out = model(image)
        pred = torch.argmax(model_out, dim=1)
        correct = (pred == label).sum()
        total_correct += correct
        #     print(model_out.size(),label.size(), image.size())
        total_num += label.size(0)
        acc_tmp = int(total_correct) / total_num
        test_acc_list.append(acc_tmp)

    print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * acc_tmp))


loss_list, train_acc_list = [], []
for epoch in range(args.epochs):
    steps = len(train_dataset) // args.batch_size
    for step, (image, label) in enumerate(train_dataloder):
        start = time.time()

        out = model(image.cuda())

        train_acc_list.append(correct)
        loss = loss_func(out, label.cuda())
        loss_list.append(loss.item())
        #         loss = loss_func(out, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        delta = time.time() - start
        if step % 10 == 0:
            print(f"[{epoch}][{step}/{steps}] loss:{loss:.2f} time:{delta:.2f}")
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label.cuda()).sum().item()
            print('Accuracy of the network on the %d train images: %.3f %%' % (total, 100.0 * correct / total))
            print(f"[EPOCH:{epoch}][{step}/{steps}] loss:{loss:.2f} Accuracy：{correct / total} time:{delta:.2f}")
            acc_ = correct / total
            model_eval(model, test_acc_list=[])  # 测试集预测
    print('{} scheduler: {}'.format(epoch, scheduler.get_last_lr()[0]))
    scheduler.step()
