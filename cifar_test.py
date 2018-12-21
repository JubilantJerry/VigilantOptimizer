#!/usr/bin/env python3

import numpy as np
import random
import torch
import torch.nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from vigilant.optim import Vigilant


class GlobalAvgPool(torch.nn.Module):
    def forward(self, features):
        (batch_size, channels, _, _) = features.size()
        return features.reshape(batch_size, channels, -1).mean(dim=2)


def main():
    random.seed(1337)
    np.random.seed(1337)
    torch.random.manual_seed(1337)
    train_cifar = datasets.CIFAR10(
        root='./data/cifar',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    test_cifar = datasets.CIFAR10(
        root='./data/cifar',
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    test_loader = torch.utils.data.DataLoader(
        test_cifar, batch_size=64)
    train_loader = torch.utils.data.DataLoader(
        train_cifar, batch_size=64, shuffle=True)

    network = torch.nn.Sequential(
        torch.nn.Conv2d(3, 128, 3),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Conv2d(128, 128, 3),
        torch.nn.BatchNorm2d(128),
        torch.nn.Dropout2d(),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Conv2d(128, 256, 3),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Conv2d(256, 256, 3),
        torch.nn.BatchNorm2d(256),
        torch.nn.Dropout2d(),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Conv2d(256, 256, 3),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Conv2d(256, 10, 3),
        GlobalAvgPool()
    ).cuda()

    for p in network.parameters():
        if p.dim() != 1:
            torch.nn.init.xavier_normal_(p)
    # network.load_state_dict(torch.load("net.pth"))

    ballpark_optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    final_optimizer = Vigilant(network.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(150):
        mean_loss = 0
        loss_count = 0

        network.train()

        optimizer = ballpark_optimizer if (i < 10) else final_optimizer

        for batch, labels in train_loader:
            batch, labels = batch.cuda(), labels.cuda()
            batch_size = batch.size()[0]
            predict_logits = network(batch)
            loss = criterion(predict_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("%.2f" % loss.item(), end=' ', flush=True)
            mean_loss += loss.item() * batch_size
            loss_count += batch_size

        mean_loss /= loss_count
        print("")
        print("Mean for epoch %d: %f" % (i, mean_loss))

        network.eval()

        mean_loss = 0
        mean_accuracy = 0
        count = 0
        for batch, labels in test_loader:
            batch, labels = batch.cuda(), labels.cuda()
            batch_size = batch.size()[0]
            predict_logits = network(batch)
            loss = criterion(predict_logits, labels)

            mean_loss += loss.item() * batch_size
            mean_accuracy += (
                predict_logits.argmax(dim=1) == labels).float().sum()
            count += batch_size

        mean_loss /= count
        mean_accuracy /= count
        print("Test: loss %f, accuracy %f" % (mean_loss, mean_accuracy))
        torch.save(network.state_dict(), "net_cifar.pth")


if __name__ == '__main__':
    main()
