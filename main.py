# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_pro import load_data_and_labels, Data
from config import opt
import models

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):

    setup_seed(opt.seed)
    opt.parse(kwargs)

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    x_text, y = load_data_and_labels("./data/rt-polarity.pos", "./data/rt-polarity.neg")
    x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=opt.test_size)

    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print("{} train data: {}, test data: {}".format(now(), len(train_data), len(test_data)))

    model = getattr(models, opt.model)(opt)
    print("{} init model finished".format(now()))
    if opt.use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)

    for epoch in range(opt.epochs):
        total_loss = 0.0
        model.train()
        for step, batch_data in enumerate(train_loader):
            x, labels = batch_data
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                labels = labels.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        acc = test(model, test_loader)
        print("{} {} epoch: loss: {}, acc: {}".format(now(), epoch, total_loss, acc))


def test(model, test_loader):
    correct = 0
    num_case = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x, labels = data
            num_case += len(labels)
            output = model(x)
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                output = output.cpu()
            predict = torch.max(output.data, 1)[1]
            correct += (predict == labels).sum().item()
        return correct * 1.0 / num_case


if __name__ == "__main__":
    import fire
    fire.Fire()
