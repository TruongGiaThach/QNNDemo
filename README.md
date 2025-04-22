# QNN for CIFAR-10

Quantum Neural Network (QNN) for CIFAR-10 classification using PyTorch and PennyLane, optimized for RTX 3060 (mobile).

## Setup

1. Install Python 3.10.
2. Create and activate a virtual environment:
    ```bash
    python3.10 -m venv qnn_env
    source qnn_env/bin/activate  # Linux/macOS
    qnn_env\Scripts\activate     # Windows
    ```

## Project Structure

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

-   Dataset is downloaded to `./data` directory (163MB)
-   Batch size: 32, optimized for RTX 3060 (6GB VRAM)
-   Uses `lightning.gpu` for quantum simulation on GPU
