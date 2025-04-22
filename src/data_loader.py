import torchvision
import torchvision.transforms as transforms
import torch
import os

def get_cifar10_data(root='./data', batch_size=32):
    """
    Tải và chuẩn bị dữ liệu CIFAR-10.
    Args:
        root (str): Thư mục lưu dataset
        batch_size (int): Kích thước batch
    Returns:
        DataLoader: Dataloader cho tập huấn luyện
    """
    # Kiểm tra dataset
    if not os.path.exists(os.path.join(root, 'cifar-10-batches-py')):
        print("CIFAR-10 not found. Downloading...")
    else:
        print("CIFAR-10 dataset found!")

    # Transform dữ liệu
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Tải dataset
    trainset = torchvision.datasets.CIFAR-10(
        root=root, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    
    return trainloader