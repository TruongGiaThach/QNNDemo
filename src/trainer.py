import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

def evaluate_model(model, testloader, device, criterion):
    """
    Đánh giá mô hình trên tập test.
    Args:
        model: Mô hình QNN
        testloader: DataLoader cho tập test
        device: Thiết bị (cuda hoặc cpu)
        criterion: Hàm loss
    Returns:
        test_loss, test_accuracy: Loss và accuracy trên tập test
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(testloader)
    test_accuracy = 100 * correct / total
    return test_loss, test_accuracy

def train_model(model, trainloader, testloader, device, n_epochs=10, lr=0.001):
    """
    Huấn luyện và đánh giá mô hình QNN.
    Args:
        model: Mô hình QNN
        trainloader: DataLoader cho tập huấn luyện
        testloader: DataLoader cho tập kiểm tra
        device: Thiết bị (cuda hoặc cpu)
        n_epochs (int): Số epoch
        lr (float): Learning rate
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    patience = 3
    counter = 0
    best_model_path = './best_model.pth'

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch + 1}/{n_epochs}")

        for i, data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'Batch Loss': f'{loss.item():.3f}'})

        avg_loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total

        print(f"\nEpoch {epoch + 1}/{n_epochs} - Avg Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
        
        # Đánh giá trên tập test
        test_loss, test_accuracy = evaluate_model(model, testloader, device, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Lưu mô hình nếu loss trên tập train cải thiện
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best loss updated to {best_loss:.4f}, model saved to {best_model_path}")
        else:
            counter += 1
            print(f"No improvement in loss. Patience counter: {counter}/{patience}")
        
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print('Finished Training')