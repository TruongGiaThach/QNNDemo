import torch
import os

def get_device():
    """
    Lấy thiết bị (GPU nếu có, nếu không thì CPU).
    Returns:
        torch.device: Thiết bị
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_dataset(root='./data'):
    """
    Kiểm tra xem dataset CIFAR-10 có tồn tại không.
    Args:
        root (str): Thư mục dataset
    Returns:
        bool: True nếu dataset tồn tại
    """
    return os.path.exists(os.path.join(root, 'cifar-10-batches-py'))