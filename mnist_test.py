#!/usr/bin/env python3

import numpy as np
import random
import torch
import torch.nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from vigilant import Vigilant as Vigilant

DISPLAY_PERIOD = 32


def main():
    random.seed(1337)
    np.random.seed(1337)
    torch.random.manual_seed(1337)

    train_mnist = datasets.MNIST(
        root='./data/mnist',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()]))
    test_mnist = datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        test_mnist, batch_size=64)
    train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=64, shuffle=True)

    network = torch.nn.Sequential(
        torch.nn.Linear(28 * 28, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 500),
        torch.nn.Dropout(),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 10)
    ).cuda()

    for p in network.parameters():
        if p.dim() != 1:
            torch.nn.init.xavier_normal_(p)

    optimizer = Vigilant(network.parameters())
    # optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)

    criterion = torch.nn.CrossEntropyLoss()

    for i in range(20):
        mean_loss = 0
        loss_count = 0
        counter = 0

        network.train()
        print("  ", end='')

        for batch, labels in train_loader:
            batch, labels = batch.cuda(), labels.cuda()
            batch_size = batch.size()[0]
            batch = batch.reshape(batch_size, 28 * 28)
            predict_logits = network(batch)
            loss = criterion(predict_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("%.2f" % loss.item(), end=' ', flush=True)
            mean_loss += loss.item() * batch_size
            loss_count += batch_size

            counter += 1
            if counter == DISPLAY_PERIOD:
                counter = 0
                optimizer.show_step_factor()

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
            batch = batch.reshape(batch_size, 28 * 28)
            predict_logits = network(batch)
            loss = criterion(predict_logits, labels)

            mean_loss += loss.item() * batch_size
            mean_accuracy += (
                predict_logits.argmax(dim=1) == labels).float().sum()
            count += batch_size

        mean_loss /= count
        mean_accuracy /= count
        print("Test: loss %f, accuracy %f" % (mean_loss, mean_accuracy))
        torch.save(network, "net_mnist.pth")


if __name__ == '__main__':
    main()
