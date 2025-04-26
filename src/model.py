import torch
import torch.nn as nn

class QNN(nn.Module):
    """
    Mô hình Quantum Neural Network cho CIFAR-10.
    Args:
        n_qubits (int): Số lượng qubit
        quantum_circuit: Quantum circuit từ PennyLane
    """
    def __init__(self, n_qubits, quantum_circuit):
        super(QNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.pre_net = nn.Linear(16 * 15 * 15, n_qubits)
        self.q_params = nn.Parameter(torch.randn(n_qubits))
        self.post_net = nn.Linear(n_qubits, 10)
        self.quantum_circuit = quantum_circuit
        
    def forward(self, x):
        x = torch.relu(self.conv(x))  # [batch_size, 16, 15, 15]
        x = x.view(-1, 16 * 15 * 15)  # [batch_size, 16*15*15]
        x = torch.tanh(self.pre_net(x))  # [batch_size, n_qubits]
        # Process batch through quantum circuit
        batch_size = x.size(0)
        x_device = x.device  # Lưu device của tensor x
        x = [self.quantum_circuit(x[i], self.q_params) for i in range(batch_size)]
        # Convert list to tensor
        x = torch.tensor(x, dtype=torch.float32, device=x_device)
        x = self.post_net(x)  # [batch_size, 10]
        return x
    
class EnhancedQNN(nn.Module):
    """
    Mô hình Quantum Neural Network cải tiến cho CIFAR-10.
    Args:
        n_qubits (int): Số lượng qubit
        quantum_circuit: Quantum circuit từ PennyLane
    """
    def __init__(self, n_qubits, quantum_circuit):
        super(EnhancedQNN, self).__init__()
        # Tăng số tầng tích chập để trích xuất đặc trưng tốt hơn
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pre_net = nn.Linear(64 * 8 * 8, n_qubits * 2)  # Tăng chiều đầu ra
        self.quantum_circuit = quantum_circuit
        # Tăng độ phức tạp của post_net
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 128),  # n_qubits * 2 vì đo cả PauliZ và PauliX
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        # Xử lý qua các tầng tích chập
        x = torch.relu(self.conv1(x))  # [batch_size, 32, 32, 32]
        x = self.pool(x)  # [batch_size, 32, 16, 16]
        x = torch.relu(self.conv2(x))  # [batch_size, 64, 16, 16]
        x = self.pool(x)  # [batch_size, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)  # [batch_size, 64*8*8]
        x = torch.tanh(self.pre_net(x))  # [batch_size, n_qubits*2]
        # Process batch through quantum circuit
        x = self.quantum_circuit(x)
        x = self.post_net(x)  # [batch_size, 10]
        return x