# QNN for CIFAR-10

# Quantum Neural Network (QNN) for CIFAR-10 Classification

This project implements a Quantum Neural Network (QNN) for CIFAR-10 classification using PyTorch and PennyLane, optimized for RTX 3060 (mobile).

## Setup

1. Install Python 3.10.
2. Create and activate a virtual environment:
    ```bash
    python3.10 -m venv qnn_env  # or ~/.pyenv/versions/3.10.6/bin/python -m venv venv
    source qnn_env/bin/activate  # Linux/macOS
    qnn_env\Scripts\activate     # Windows

    python -m pip install -r requirements.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118
    ```

## Project Structure

Below is the file folder tree and a brief description of each key component:

```plaintext
QNN-CIFAR10/
├── main.py                # Main script to run the program
├── README.md              # Project documentation
└── src/
    ├── data_loader.py     # Loads CIFAR-10 dataset
    ├── quantum_circuit.py # Defines quantum circuit
    ├── model.py           # Defines QNN model
    ├── trainer.py         # Training logic
    └── utils.py           # Utility functions
```

## Technical Notes

- **Dataset**: Downloaded to `./data` directory (163MB).
- **Batch size**: 32, optimized for RTX 3060 (6GB VRAM).
- **Quantum simulation**: Uses `lightning.gpu` for GPU acceleration.
- Reference: 
    -   [Quantum Convolutional Neural Networks for Classification](https://medium.com/@devmallyakarar/quantum-convolutional-neural-networks-for-classification-using-interaction-layers-d94649de42b5).
    -   https://pennylane.ai/qml/demos/tutorial_quanvolution 
    -   https://arxiv.org/pdf/1904.04767

## Quantum Convolutional Neural Network (QCNN)

### Overview
QCNN combines quantum layers with classical layers to leverage the advantages of quantum computing. Quantum convolution layers replace or complement classical convolution layers, reducing parameters and enhancing data representation.

### Interaction Layers
- Utilize quantum gates (e.g., CNOT, CZ, rotation gates) to create interactions between qubits.
- Enable the model to learn nonlinear relationships between features, which classical models may struggle with under parameter constraints.

### Model Structure
1. **Input**: CIFAR-10 images are preprocessed and encoded into quantum states.
2. **Quantum Convolution Layers**: Apply quantum transformations on small data regions (similar to kernels in classical CNNs).
3. **Interaction Layers**: Enhance dependencies between qubits.
4. **Output**: Measurements from the quantum circuit are passed to classical layers for predictions.

### Results
- QCNN with interaction layers achieves higher accuracy than classical CNNs of similar size on CIFAR-10.
- Quantum models use fewer parameters, making them suitable for resource-constrained quantum devices (NISQ - Noisy Intermediate-Scale Quantum).

### Applications
- Image classification, especially in resource-limited environments.
- Potential for expansion to other domains like natural language processing (NLP) or medical data classification.