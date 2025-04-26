import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def get_cifar10_data(root='./data', batch_size=32, train_ratio=0.7):
    """
    Tải và chia dữ liệu CIFAR-10 thành train/test (70/30).
    Args:
        root (str): Thư mục lưu dữ liệu
        batch_size (int): Kích thước batch
        train_ratio (float): Tỷ lệ dữ liệu train
    Returns:
        trainloader, testloader: DataLoader cho train và test
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    
    # Chia dữ liệu thành train/test
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_size = int(train_ratio * dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    testloader = DataLoader(trainset, batch_size=batch_size, sampler=test_sampler)

    return trainloader, testloader