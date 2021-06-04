# reference implementation of HIGGS training

import argparse
import os
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

# for reproducibility inits
import random
import numpy as np

# give this to each dataloader
def dataloader_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
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
class MLPNet(nn.Module):
    """
    X layer fully connected network
    """

    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


def train(model, optimizer, data_loader):
    model.train()
    for batch_idx, (x, target) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        out = model(x)

        loss = criterion(out.squeeze(), target.to(torch.float))
        loss.backward()
        optimizer.step()

    return model


def test(model, data_loader):
    correct_cnt = 0
    total_cnt = 0
    summed_loss = 0
    model.eval()
    for batch_idx, (x, target) in enumerate(tqdm(data_loader)):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        out = model(x)

        loss = criterion(out.squeeze(), target.to(torch.float))
        pred_label = (out.squeeze().data >= 0.5).to(torch.int)
        correct_cnt += (pred_label == target.data).sum()
        summed_loss += loss.detach().cpu().numpy()
        total_cnt += x.shape[0]

    return summed_loss, correct_cnt, total_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Neural Network on HIGGS and check accuracy.')
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
    parser.add_argument('--seed', default=7, type=int, help='Seed for reproducibility.')

    args = parser.parse_args()

    # reproducibility
    seed = args.seed

    # reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = args.batch_size

    use_cuda = torch.cuda.is_available()

    train_set, test_set = load_HIGGS(
        root="experiments/HIGGS/higgs_data/", transform=transforms.ToTensor()
    )

    val_size = int(0.05 * len(train_set))
    train_size = len(train_set) - val_size

    train_set, val_set = random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    # SGD (row_limit: 400_000, n_epochs=20)
    # lr=0.05: Final validation loss: 0.004345, acc: 0.715
    # lr=0.1: Final validation loss: 0.004203, acc: 0.725
    # lr=0.5: Final validation loss: 0.004201, acc: 0.726
    # lr=1.0: Final validation loss: 0.004280, acc: 0.720

    # ADAM (row_limit: 400_000, n_epochs=20)
    # lr=1e-4: Final validation loss: 0.004180, acc: 0.727
    # lr=5e-4: Final validation loss: 0.004025, acc: 0.746
    # lr=1e-3: Final validation loss: 0.004112, acc: 0.740

    n_epochs = args.epochs
    # for lr in [1e-4, 5e-4, 1e-3]:
    for lr in [0.5]:
        print()
        print("lr", lr)

        num_workers = 1
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=dataloader_seed_worker,
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=dataloader_seed_worker,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=dataloader_seed_worker,
        )

        print(f"Total training batches: {len(train_loader)}")
        print(f"Total validation batches: {len(val_loader)}")
        print(f"Total testing batches: {len(test_loader)}")

        torch.manual_seed(seed)
        model = MLPNet()

        if use_cuda:
            model = model.cuda()

        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        summed_loss, correct_cnt, total_cnt = test(model, val_loader)
        print(
            f"Initial loss: {summed_loss / total_cnt:.6f}, acc: {correct_cnt / total_cnt:.3f}"
        )
        history = np.zeros((n_epochs + 1, 2))
        history[0, 0] = summed_loss / total_cnt
        history[0, 1] = correct_cnt / total_cnt
        for epoch in range(n_epochs):
            model = train(model, optimizer, train_loader)
            summed_loss, correct_cnt, total_cnt = test(model, val_loader)
            history[epoch + 1, 0] = summed_loss / total_cnt
            history[epoch + 1, 1] = correct_cnt / total_cnt
            print(correct_cnt/total_cnt)
        print(
            f"Final loss: {summed_loss / total_cnt:.6f}, acc: {correct_cnt / total_cnt:.3f}"
        )

        # summed_loss, correct_cnt, total_cnt = test(model, test_loader)
        # print()
        # print(
        #     f"\nTest loss: {summed_loss / total_cnt:.6f}, test acc: {correct_cnt / total_cnt:.3f}"
        # )

        # torch.save(model.state_dict(), f"{model.__class__.__name__}.pt")
        # np.save("history_higgs.npy", history)