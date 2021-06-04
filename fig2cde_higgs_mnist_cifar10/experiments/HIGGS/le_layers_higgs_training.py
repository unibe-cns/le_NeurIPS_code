# reference implementation of HIGGS training

import argparse

import os
import sys
sys.path.append(".")
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import model.latent_equilibrium_layers as nn
import model.layered_torch_utils as tu
from model.network_params import ModelVariant, TargetType
import pandas as pd

# for reproducibility inits
import random
import numpy as np


# give this to each dataloader
def dataloader_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# higgs dataset
def load_HIGGS(root, transform, n_row_limit=400_000):
    """Download the dataset manually from
    https://archive.ics.uci.edu/ml/datasets/HIGGS and copy it into
    `root`.
    """
    n_train_limit = int(0.9 * n_row_limit)
    data = pd.read_csv(
        os.path.join(root, "HIGGS.csv"), header=None, dtype="float32", nrows=n_row_limit
    )
    data_train = data[:n_train_limit].reset_index(drop=True).values
    data_test = data[n_train_limit:].reset_index(drop=True).values

    class HIGGSTrainDataSet:
        def __getitem__(self, idx):
            return data_train[idx][1:], int(data_train[idx][0])

        def __len__(self):
            return len(data_train)

    class HIGGSTestDataSet:
        def __getitem__(self, idx):
            return data_test[idx][1:], int(data_test[idx][0])

        def __len__(self):
            return len(data_test)

    return HIGGSTrainDataSet(), HIGGSTestDataSet()


# networks

def MLPNet(batch_size, lr_multiplier, tau=10.0, dt=0.1, beta=0.1, model_variant=ModelVariant.VANILLA, target_type=TargetType.RATE, presentation_steps=10, with_optimizer=False):
    """
    3 layer fully connected network
    """
    learning_rate = 0.125 * lr_multiplier / presentation_steps / dt

    act_func = tu.HardSigmoid

    # fc1 = nn.Linear(28, 300, act_func)
    # fc2 = nn.Linear(300, 100, act_func)
    # fc3 = nn.Linear(100, 1, tu.Linear)
    # layers = [fc1, fc2, fc3]
    # lr_factors = [1.0, 0.2, 0.1]

    fc1 = nn.Linear(28, 300, act_func)
    fc2 = nn.Linear(300, 300, act_func)
    fc3 = nn.Linear(300, 300, act_func)
    fc4 = nn.Linear(300, 1, tu.Linear)
    layers = [fc1, fc2, fc3, fc4]
    lr_factors = [1.0, 0.2, 0.1, 0.1]

    network = nn.LESequential(layers, learning_rate, lr_factors, None, None,
                              tau, dt, beta, model_variant, target_type, with_optimizer=with_optimizer)

    return network


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a Latent Equilibrium Neural Network on HIGGS and check accuracy.')
    parser.add_argument('--model_variant', default="vanilla", type=str, help="Model variant: vanilla, full_forward_pass")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training")
    parser.add_argument('--batch_learning_multiplier', default=64, type=int, help="Learning rate multiplier for batch learning")
    parser.add_argument('--n_updates', default=20, type=int, help="Number of update steps per sample/batch")
    parser.add_argument('--with_optimizer', action='store_true', help="Train network with Adam Optimizer")
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
    lr_multiplier = args.batch_learning_multiplier

    # setup network parameters
    tau = 10.0
    dt = 0.1
    beta = 0.1
    model_variant = args.model_variant
    target_type = TargetType.RATE
    presentation_steps = args.n_updates

    # load HIGGS dataset
    use_cuda = torch.cuda.is_available()

    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))]) # We exclude any data augmentation here that generates more training data
    target_transform = transforms.Compose([
        lambda x:torch.LongTensor([x]),
        lambda x: F.one_hot(x, 10),
        lambda x: x.squeeze()
    ])

    train_set, test_set = load_HIGGS(
        root="experiments/HIGGS/higgs_data/", transform=transforms.ToTensor()
    )

    val_size = int(0.05 * len(train_set))
    train_size = len(train_set) - val_size

    train_set, val_set = random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    n_epochs = args.epochs

    num_workers = 0
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=dataloader_seed_worker,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=dataloader_seed_worker,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=dataloader_seed_worker,
        drop_last=True
    )

    print(f"Total training batches: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")
    print(f"Total testing batches: {len(test_loader)}")

    # training

    model = MLPNet(batch_size, lr_multiplier, tau, dt, beta, model_variant, target_type, presentation_steps, args.with_optimizer)

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # criterion = nn.CrossEntropyLoss()

    # testing prior to training
    correct_cnt, summed_loss = 0, 0
    total_cnt = 0
    model.eval()
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        target = target.unsqueeze(-1)

        for update_i in range(presentation_steps):
            model.update(x, target)

        loss = model.errors[-1]
        out = model.rho[-1]
        # loss = criterion(out, target)
        pred_label = (out.squeeze().data >= 0.5).to(torch.int).unsqueeze(-1)
        total_cnt += x.shape[0]
        correct_cnt += (pred_label == target).sum().detach().cpu().numpy()
        summed_loss = summed_loss + torch.abs(loss).sum().detach().cpu().numpy()

    print(f'\nTest loss: {summed_loss/total_cnt:.6f}, test acc: {correct_cnt / total_cnt:.3f}')

    val_accuracies = []
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
            target = target.unsqueeze(-1)

            for update_i in range(presentation_steps):
                model.update(x, target)

            loss = model.errors[-1]
            out = model.rho[-1]
            # loss = criterion(out, target)
            pred_label = (out.squeeze().data >= 0.5).to(torch.int).unsqueeze(-1)
            total_cnt += x.shape[0]
            correct_cnt += (pred_label == target).sum().detach().cpu().numpy()
            summed_loss = summed_loss + torch.abs(loss).sum().detach().cpu().numpy()
            # loss.backward()
            # optimizer.step()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Epoch: {epoch}, batch index: {batch_idx + 1}, train loss: {(summed_loss/total_cnt):.6f}, train acc: {correct_cnt / total_cnt:.3f}')

        # validation
        correct_cnt, summed_loss = 0, 0
        total_cnt = 0
        model.eval()
        for batch_idx, (x, target) in enumerate(val_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            target = target.unsqueeze(-1)

            for update_i in range(presentation_steps):
                model.update(x, target)

            loss = model.errors[-1]
            out = model.rho[-1]
            # loss = criterion(out, target)
            pred_label = (out.squeeze().data >= 0.5).to(torch.int).unsqueeze(-1)
            total_cnt += x.shape[0]
            correct_cnt += (pred_label == target).sum().detach().cpu().numpy()
            summed_loss = summed_loss + torch.abs(loss).sum().detach().cpu().numpy()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
                print(f'Epoch: {epoch}, batch index: {batch_idx + 1}, val loss: {(summed_loss/total_cnt):.6f}, val acc: {correct_cnt / total_cnt:.3f}')

        val_accuracies.append(correct_cnt / total_cnt)

    # testing
    correct_cnt, summed_loss = 0, 0
    total_cnt = 0
    model.eval()
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        target = target.unsqueeze(-1)

        for update_i in range(presentation_steps):
            model.update(x, target)

        loss = model.errors[-1]
        out = model.rho[-1]
        # loss = criterion(out, target)
        pred_label = (out.squeeze().data >= 0.5).to(torch.int).unsqueeze(-1)
        total_cnt += x.shape[0]
        correct_cnt += (pred_label == target).sum().detach().cpu().numpy()
        summed_loss = summed_loss + torch.abs(loss).sum().detach().cpu().numpy()

    print(f'\nTest loss: {summed_loss/total_cnt:.6f}, test acc: {correct_cnt / total_cnt:.3f}')
    np.savez_compressed(f"output/higgs_le_{model_variant}_{presentation_steps}_{seed}.npz", val_accuracies=np.array(val_accuracies), test_accuracy=np.array([correct_cnt / total_cnt]))



    # torch.save(model.state_dict(), f"{model.__class__.__name__}.pt")
