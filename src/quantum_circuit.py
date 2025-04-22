import pennylane as qml

def create_quantum_circuit(n_qubits, device_name="lightning.gpu"):
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
        for i in range(n_qubits):
            qml.RX(inputs[..., i], wires=i)
            qml.RY(weights[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit