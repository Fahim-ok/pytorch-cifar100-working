# train.py
#!/usr/bin/env python3

""" train network using pytorch with split learning

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, WarmUpLR, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from dataset import CIFAR100Split  # Import the dataset splitting class

def train(epoch, train_loader, net, optimizer, loss_function, writer, warmup_scheduler):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch, test_loader, net, loss_function, writer):
    start = time.time()
    net.eval()
    device = next(net.parameters()).device  # Get the device of the model

    test_loss = 0.0
    correct = 0.0

    for (images, labels) in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move to the correct device

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))

    writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    # Define class splits for split learning
    tasks = [
        list(range(0, 20)),  # Task 1: Classes 0-19
        list(range(20, 40)),  # Task 2: Classes 20-39
        list(range(40, 60)),  # Task 3: Classes 40-59
        list(range(60, 80)),  # Task 4: Classes 60-79
        list(range(80, 100))  # Task 5: Classes 80-99
    ]

    net = get_network(args)

    loss_function = nn.CrossEntropyLoss()

    # Use TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

    # Train the model sequentially on each task
    for task_id, task_classes in enumerate(tasks):
        print(f"Training on Task {task_id + 1} with classes: {task_classes}")

        # Update the fully connected layer for the current task
        net.update_fc(num_classes=len(task_classes))
        if args.gpu:
            net = net.cuda()

        # Reinitialize the optimizer for the updated model
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        # Create training and test datasets for the current task
        train_dataset = CIFAR100Split(path=settings.CIFAR100_PATH, class_indices=task_classes, transform=None)
        test_dataset = CIFAR100Split(path=settings.CIFAR100_PATH, class_indices=task_classes, transform=None)

        train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=4)

        # Warm-up scheduler
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

        best_acc = 0.0  # Reset best accuracy for each task

        # Train for the specified number of epochs
        for epoch in range(1, settings.EPOCH + 1):
            if epoch > args.warm:
                train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
                train_scheduler.step(epoch)

            train(epoch, train_loader, net, optimizer, loss_function, writer, warmup_scheduler)
            acc = eval_training(epoch, test_loader, net, loss_function, writer)

            # Save the best model for the current task
            if best_acc < acc:
                weights_path = os.path.join(settings.CHECKPOINT_PATH, f"task_{task_id + 1}_best.pth")
                print(f"Saving best model for Task {task_id + 1} to {weights_path}")
                torch.save(net.state_dict(), weights_path)
                best_acc = acc

    writer.close()
