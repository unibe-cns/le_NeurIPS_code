# reference implementation of LE CIFAR10 training

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.append(".")
import model.latent_equilibrium_layers as nn
import model.layered_torch_utils as tu
from model.network_params import ModelVariant, TargetType

# for reproducibility inits
import random
import numpy as np


# give this to each dataloader
def dataloader_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# networks

def MLPNet(batch_size, lr_multiplier, tau=10.0, dt=0.1, beta=0.1, model_variant=ModelVariant.VANILLA, target_type=TargetType.RATE, presentation_steps=10, with_optimizer=False):
    """
    3 layer fully connected network
    """
    learning_rate = 0.125 * lr_multiplier / presentation_steps / dt

    act_func = tu.HardSigmoid

    fc1 = nn.Linear(28 * 28, 300, act_func)
    fc2 = nn.Linear(300, 100, act_func)
    fc3 = nn.Linear(100, 10, tu.Linear)

    network = nn.LESequential([fc1, fc2, fc3], learning_rate, [1.0, 0.2, 0.1], None, None,
                              tau, dt, beta, model_variant, target_type, with_optimizer=with_optimizer)

    return network


def LeNet5(batch_size, lr_multiplier, tau=10.0, dt=0.1, beta=0.1, model_variant=ModelVariant.VANILLA, target_type=TargetType.RATE, presentation_steps=10, with_optimizer=False):
    """
    - 2 Convs with max pooling and relu
    - 2 Fully connected layers and relu
    """
    learning_rate = 0.125 * lr_multiplier / presentation_steps / dt

    act_func = tu.HardSigmoid

    l1 = nn.Conv2d(3, 20, 5, batch_size, 32, act_func)
    l2 = nn.MaxPool2d(2)
    l3 = nn.Conv2d(20, 50, 5, batch_size, 14, act_func)
    l4 = nn.MaxPool2d(2)
    l5 = nn.Projection((batch_size, 50, 5, 5), 500, act_func)
    l6 = nn.Linear(500, 10, tu.Linear)

    network = nn.LESequential([l1, l2, l3, l4, l5, l6], learning_rate, [1.0, 0.2, 0.2, 0.2, 0.2, 0.1], None, None,
                              tau, dt, beta, model_variant, target_type, with_optimizer=with_optimizer)

    return network


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train lagrange mnist on all 10 classes and check accuracy.')
    parser.add_argument('--model_variant', default="vanilla", type=str, help="Model variant: vanilla, full_forward_pass")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training")
    parser.add_argument('--batch_learning_multiplier', default=64, type=int, help="Learning rate multiplier for batch learning")
    parser.add_argument('--n_updates', default=10, type=int, help="Number of update steps per sample/batch")
    parser.add_argument('--with_optimizer', action='store_true', help="Train network with Adam Optimizer")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
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
    lr_multiplier = args.batch_learning_multiplier

    # setup network parameters
    tau = 10.0
    dt = 0.1
    beta = 0.1
    model_variant = args.model_variant
    target_type = TargetType.RATE
    presentation_steps = args.n_updates

    # load CIFAR10 dataset
    use_cuda = torch.cuda.is_available()

    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # We exclude any data augmentation here that generates more training data
    target_transform = transforms.Compose([
        lambda x:torch.LongTensor([x]),
        lambda x: F.one_hot(x, 10),
        lambda x: x.squeeze()
    ])

    # if not existing, download mnist dataset
    train_set = datasets.CIFAR10(root='./cifar10_data', train=True, transform=transform, target_transform=target_transform, download=True)
    test_set = datasets.CIFAR10(root='./cifar10_data', train=False, transform=transform, target_transform=target_transform, download=True)

    val_size = 10000
    train_size = len(train_set) - val_size

    train_set, val_set = random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    num_workers = 0
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker, drop_last=True)

    print(f'Total training batches: {len(train_loader)}')
    print(f'Total validation batches: {len(val_loader)}')
    print(f'Total testing batches: {len(test_loader)}')

    # training

    model = LeNet5(batch_size, lr_multiplier, tau, dt, beta, model_variant, target_type, presentation_steps, args.with_optimizer)
    # model = MLPNet(batch_size, lr_multiplier, tau, dt, beta, model_variant, target_type, presentation_steps, args.with_optimizer)

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # training
        correct_cnt, summed_loss = 0, 0
        total_cnt = 0
        summed_loss = 0
        model.train()
        for batch_idx, (x, target) in enumerate(tqdm(train_loader, desc="Batches")):
            # optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()

            for update_i in range(presentation_steps):
                model.update(x, target)

                # for p in model.parameters():
                #     print("----------------")
                #     print(f"Layer with {p.shape} has gradient norm {p.grad.data.norm(2)}")
                # print("---------------------\n")

            loss = model.errors[-1]
            out = model.rho[-1]
            # loss = criterion(out, target)
            _, pred_label = torch.max(out, 1)
            total_cnt += x.shape[0]
            correct_cnt += (pred_label == torch.max(target, 1)[1]).sum()
            summed_loss += loss.detach().cpu().numpy()
            # loss.backward()
            # optimizer.step()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Epoch: {epoch}, batch index: {batch_idx + 1}, train loss: {(np.abs(summed_loss).sum(1)/total_cnt).mean(0):.6f}, train acc: {correct_cnt / total_cnt:.3f}')

        # validation
        correct_cnt, summed_loss = 0, 0
        total_cnt = 0
        model.eval()
        for batch_idx, (x, target) in enumerate(val_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()

            for update_i in range(presentation_steps):
                model.update(x, target)

            loss = model.errors[-1]
            out = model.rho[-1]
            _, pred_label = torch.max(out, 1)
            total_cnt += x.shape[0]
            correct_cnt += (pred_label == torch.max(target, 1)[1]).sum()
            summed_loss = summed_loss + loss.detach().cpu().numpy()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
                print(f'Epoch: {epoch}, batch index: {batch_idx + 1}, val loss:  {(np.abs(summed_loss).sum(1)/total_cnt).mean(0):.6f}, val acc: {correct_cnt / total_cnt:.3f}')

    # testing
    correct_cnt, summed_loss = 0, 0
    total_cnt = 0
    model.eval()
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()

        for update_i in range(presentation_steps):
            model.update(x, target)

        loss = model.errors[-1]
        out = model.rho[-1]
        _, pred_label = torch.max(out, 1)
        total_cnt += x.shape[0]
        correct_cnt += (pred_label == torch.max(target, 1)[1]).sum()
        summed_loss += loss.detach().cpu().numpy()

    print(f'\nTest loss: {(np.abs(summed_loss).sum(1)/total_cnt).mean(0):.6f}, test acc: {correct_cnt / total_cnt:.3f}')

    # torch.save(model.state_dict(), f"{model.__class__.__name__}.pt")
