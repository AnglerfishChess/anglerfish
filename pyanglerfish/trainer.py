import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import io
import zstandard as zstd
from stockfish import Stockfish
from pathlib import Path
from typing import Iterator

class ChessDataset(Dataset):
    """
    A PyTorch dataset for chess positions.

    :param data_file: Path to the dataset file (Zstandard-compressed JSON lines).
    """
    def __init__(self, data_file: Path):
        self.data_file = data_file

    def __len__(self) -> int:
        """
        Returns an arbitrary large number; the dataset is streamed.
        """
        return 10**9  # Just to satisfy PyTorch expectations

    def __getitem__(self, idx: int):
        """
        Fetches a single chess position and its evaluation.

        :param idx: Unused, since the dataset is streamed.
        :return: Encoded chess position tensor and its evaluation.
        """
        raise NotImplementedError("Streaming dataset should be used with an iterator!")

    def data_generator(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Streams the dataset line by line, decoding Zstandard and parsing JSON.

        :return: Generator yielding (input_tensor, target_tensor) pairs.
        """
        with self.data_file.open("rb") as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text_reader:
                    for line in text_reader:
                        data = json.loads(line)
                        fen = data["fen"]
                        best_eval = max(data["evals"], key=lambda x: x["depth"])
                        eval_value = best_eval["pvs"][0].get("cp", 1000 * best_eval["pvs"][0].get("mate", 0))
                        input_tensor = self._encode_fen(fen)
                        target_tensor = torch.tensor(eval_value, dtype=torch.float32)
                        yield input_tensor, target_tensor

    def _encode_fen(self, fen: str) -> torch.Tensor:
        """
        Encodes a FEN string into a numerical representation.

        :param fen: Chess position in FEN format.
        :return: Encoded tensor.
        """
        stockfish = Stockfish()
        stockfish.set_fen_position(fen)
        board = stockfish.get_board_visual()

        # Encoding: my pieces, opponent's pieces, castling, en passant, bishop pair
        encoded = torch.zeros(768, dtype=torch.float32)

        # TODO: Implement proper encoding

        return encoded


class ChessNet(nn.Module):
    """
    A simple neural network for chess position evaluation.
    """
    def __init__(self, input_size: int = 768, hidden_size: int = 256, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def ui_train_network(data_path: Path, model_save_path: Path, epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001):
    """
    Trains a neural network using chess positions and evaluations.

    :param data_path: Path to the training data file (Zstandard JSON lines).
    :param model_save_path: Path to save the trained PyTorch model.
    :param epochs: Number of training epochs.
    :param batch_size: Training batch size.
    :param learning_rate: Learning rate for the optimizer.
    """
    dataset = ChessDataset(data_path)
    data_loader = DataLoader(dataset.data_generator(), batch_size=batch_size, shuffle=True)

    model = ChessNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if model_save_path.exists():
        model.load_state_dict(torch.load(model_save_path))
        print("Loaded existing model weights.")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # Commented: Compute Stockfish evaluation directly
    # stockfish = Stockfish()
    # stockfish.set_fen_position("some_fen_here")
    # eval_30 = stockfish.get_evaluation(depth=30)
    # print("Stockfish eval at depth 30:", eval_30)
