import pennylane as qml

def create_quantum_circuit(n_qubits,n_layers=2, device_name="lightning.gpu"):
    """
    Tạo quantum circuit cho QNN.
    Args:
        n_qubits (int): Số lượng qubit
        device_name (str): Tên thiết bị quantum (mặc định lightning.gpu)
    Returns:
        QNode: Quantum circuit đã được định nghĩa
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(inputs[..., i], wires=i)
                qml.RY(weights[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])  
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit

def create_enhanced_quantum_circuit(n_qubits, n_layers=6, device_name="lightning.gpu"):
    """
    Tạo quantum circuit cải tiến cho QNN trên CIFAR-10.
    Args:
        n_qubits (int): Số lượng qubit (khuyến nghị 8)
        n_layers (int): Số tầng của mạch (khuyến nghị 6)
        device_name (str): Tên thiết bị quantum (mặc định lightning.gpu)
    Returns:
        QNode: Quantum circuit đã được định nghĩa
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        # Amplitude encoding cho dữ liệu đầu vào
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        
        # Strongly entangling layers
        for layer in range(n_layers):
            # Rotational gates với tham số học được
            for i in range(n_qubits):
                qml.CRX(weights[layer, i, 0], wires=[i, (i + 1) % n_qubits])
                qml.CRY(weights[layer, i, 1], wires=[i, (i + 1) % n_qubits])
            # Entangling gates
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Thêm CNOT vòng để tăng kết nối
            qml.CNOT(wires=[n_qubits - 1, 0])
        
        # Đo giá trị kỳ vọng trên PauliZ và PauliX
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] 

    circuit = qml.transforms.broadcast_expand(circuit)

    # Khởi tạo weights với kích thước phù hợp
    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)