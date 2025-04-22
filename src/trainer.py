import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

def train_model(model, trainloader, device, n_epochs=10, lr=0.001):
    """
    Huấn luyện mô hình QNN.
    Args:
        model: Mô hình QNN
        trainloader: DataLoader cho tập huấn luyện
        device: Thiết bị (cuda hoặc cpu)
        n_epochs (int): Số epoch
        lr (float): Learning rate
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    best_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        avg_loss = running_loss / len(trainloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping")
            break

    print('Finished Training')