from src.data_loader import get_cifar10_data
from src.quantum_circuit import create_enhanced_quantum_circuit, create_quantum_circuit
from src.model import QNN, EnhancedQNN
from src.trainer import train_model
from src.utils import get_device, check_dataset

def main():
    # Thiết lập tham số
    n_qubits = 4
    batch_size = 32
    n_epochs = 10
    data_root = './data'

    # Kiểm tra thiết bị và dataset
    device = get_device()
    print(f"Using device: {device}")
    if check_dataset(data_root):
        print("CIFAR-10 dataset ready")
    else:
        print("CIFAR-10 dataset will be downloaded")

    # Tải dữ liệu
    trainloader, testloader = get_cifar10_data(root=data_root, batch_size=batch_size)

    #Basic Model
    quantum_circuit = create_quantum_circuit(n_qubits)
    model = QNN(n_qubits, quantum_circuit).to(device)
    train_model(model, trainloader, testloader, device, n_epochs=n_epochs)

    # # Enhanced Model
    # quantum_circuit = create_enhanced_quantum_circuit(n_qubits, n_layers=2)
    # model = EnhancedQNN(n_qubits, quantum_circuit).to(device)
    # train_model(model, trainloader, testloader, device, n_epochs=n_epochs, lr=0.0005)

if __name__ == "__main__":
    main()