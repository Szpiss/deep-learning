#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader, resolve_device

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='inference device')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-num-workers', type=int, default=2, help='dataloader worker count')
    args = parser.parse_args()

    device = resolve_device(args)
    print('Using device:', device)
    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=args.num_workers,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights, map_location=device))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            image = image.to(device)
            label = label.to(device)

            if device.type == 'cuda':
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if device.type == 'cuda':
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
