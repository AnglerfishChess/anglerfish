import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from stockfish import Stockfish
import os
import json

# Ensure the Stockfish engine is properly configured
# STOCKFISH_PATH = "/path/to/stockfish"  # Replace with the actual path
# stockfish = Stockfish(STOCKFISH_PATH)

class ChessDataset(Dataset):
    """
    A custom PyTorch dataset for chess training data.
    Each data point consists of a chess position (FEN) and its evaluation.
    """
    def __init__(self, data_file: str):
        """

        :param data_file: Path to the dataset file (JSON format with FEN and evaluations).
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract a single data sample
        sample = self.data[idx]
        fen = sample['FEN']  # Chess position in FEN format
        evaluation = sample['Evaluation']  # A numeric evaluation (e.g., Stockfish's eval score)

        # Preprocess FEN and eval to tensor format
        input_tensor = self._encode_fen(fen)
        target_tensor = torch.tensor(evaluation, dtype=torch.float32)
        return input_tensor, target_tensor

    def _encode_fen(self, fen):
        """
        Encodes a FEN string into a numerical representation. Simplistic for illustration.
        """
        # A simple encoding, e.g., 1-hot encoding of board positions
        encoded = [ord(char) for char in fen]  # Example encoding. You should refine this.
        return torch.tensor(encoded, dtype=torch.float32)[:256]  # Truncate/pad to a fixed size


class ChessNet(nn.Module):
    """
    A simple neural network for chess position evaluation.
    """
    def __init__(self, input_size=256, hidden_size=128, output_size=1):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def ui_train_network(data_path, model_save_path, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Trains a neural network using chess positions and evaluations.

        data_path (str): Path to the training data file (JSON format).
        model_save_path (str): Path to save the trained PyTorch model.
        epochs (int): Number of epochs for training.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        None
    """
    # Load and prepare the dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found.")

    dataset = ChessDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the neural network, optimizer, and loss function
    model = ChessNet(input_size=256, hidden_size=128, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error for regression tasks

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")