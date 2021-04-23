import torch
from torchvision import datasets, transforms
import torch.utils.data


def prepMNIST(batch_size=512):

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../../../data', train=True, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))     # mean and std
            ]),
        ), batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../../../data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ), batch_size=batch_size, shuffle=True
    )

    return train_loader, test_loader
