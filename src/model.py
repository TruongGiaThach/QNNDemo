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
    
