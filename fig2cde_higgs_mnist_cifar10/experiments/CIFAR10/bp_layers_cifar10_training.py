# reference implementation of BP CIFAR10 training

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# for reproducibility inits
import random
import numpy as np


# give this to each dataloader
def dataloader_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# networks

class MLPNet(nn.Module):
    """
    3 layer fully connected network
    """
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten the data before continuing
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    """
    - 2 Convs with max pooling and relu
    - 2 Fully connected layers and relu
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a DNN on MNIST and check accuracy.')
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training")
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train.')
    parser.add_argument('--seed', default=7, type=int, help='Seed for reproducibility.')

    args = parser.parse_args()

    # reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = args.batch_size

    # load MNIST dataset
    use_cuda = torch.cuda.is_available()

    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # We exclude any data augmentation here that generates more training data

    # if not existing, download mnist dataset
    train_set = datasets.CIFAR10(root='./cifar10_data', train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10(root='./cifar10_data', train=False, transform=transform, download=True)

    val_size = 10000
    train_size = len(train_set) - val_size

    train_set, val_set = random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    num_workers = 0
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker)

    print(f'Total training batches: {len(train_loader)}')
    print(f'Total validation batches: {len(val_loader)}')
    print(f'Total testing batches: {len(test_loader)}')

    # training

    model = LeNet5()
    # model = MLPNet()

    if use_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01) #, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    val_accuracies = []
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # training
        correct_cnt, summed_loss = 0, 0
        total_cnt = 0
        summed_loss = 0
        model.train()
        for batch_idx, (x, target) in enumerate(tqdm(train_loader, desc="Batches")):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            out = model(x)

            loss = criterion(out, target)
            _, pred_label = torch.max(out, 1)
            total_cnt += x.shape[0]
            correct_cnt += (pred_label == target).sum()
            summed_loss += loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Epoch: {epoch}, batch index: {batch_idx + 1}, train loss: {summed_loss / total_cnt:.6f}, train acc: {correct_cnt * 1.0 / total_cnt:.3f}')

        # validation
        correct_cnt, summed_loss = 0, 0
        total_cnt = 0
        model.eval()
        for batch_idx, (x, target) in enumerate(val_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            out = model(x)

            loss = criterion(out, target)
            _, pred_label = torch.max(out, 1)
            total_cnt += x.shape[0]
            correct_cnt += (pred_label == target).sum()
            summed_loss = summed_loss + loss.detach().cpu().numpy()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
                print(f'Epoch: {epoch}, batch index: {batch_idx + 1}, val loss: {summed_loss / total_cnt:.6f}, val acc: {correct_cnt / total_cnt:.3f}')

        val_accuracies.append((correct_cnt / total_cnt).detach().cpu().numpy())

    # testing
    correct_cnt, summed_loss = 0, 0
    total_cnt = 0
    model.eval()
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        out = model(x)
        loss = criterion(out, target)
        _, pred_label = torch.max(out, 1)
        total_cnt += x.shape[0]
        correct_cnt += (pred_label == target).sum()
        summed_loss += loss.detach().cpu().numpy()

    np.savez_compressed(f"output/cifar10_bp_lenet5_{seed}.npz", val_accuracies=np.array(val_accuracies), test_accuracy=np.array([(correct_cnt / total_cnt).detach().cpu().numpy()]))

    print(f'\nTest loss: {summed_loss / total_cnt:.6f}, test acc: {correct_cnt / total_cnt:.3f}')

    # torch.save(model.state_dict(), f"{model.__class__.__name__}.pt")
