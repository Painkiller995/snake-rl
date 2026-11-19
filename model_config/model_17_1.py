import torch
import torch.nn as nn
import torch.nn.functional as F

# Same values used in the Json
config = {
    "version": "v17.1",
    "board_size": 10,
    "frames": 2,
    "max_time_limit": 998,
    "supervised": 0,
    "n_actions": 4,
    "obstacles": 1,
    "buffer_size": 80000,
}


class DQNModel(nn.Module):
    """
    CNN that predicts a Q-value (score) for every possible action.
    Input: (batch, frames, board_size, board_size)
    """

    def __init__(self, board_size=10, frames=2, n_actions=4):
        super().__init__()

        # Input: 2 frames, output: 16 feature maps.
        # Output size = ((input + 2 * padding - kernel) / stride) + 1.
        # conv1: ((10 + 2 * 1 - 3) / 1) + 1 = 10.
        self.conv1 = nn.Conv2d(frames, 16, kernel_size=3, padding=1)

        # Input: 16 feature maps, output: 32 feature maps.
        # 3x3 filters detect patterns from the previous layer.
        # Output size = ((input + 2 * padding - kernel) / stride) + 1.
        # conv2: ((10 + 2 * 0 - 3) / 1) + 1 = 8.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        # Input: 32 feature maps, output: 64 feature maps.
        # 5x5 filters detect larger patterns.
        # Output size = ((input + 2 * padding - kernel) / stride) + 1.
        # conv3: ((8 + 2 * 0 - 5) / 1) + 1 = 4.
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)

        # conv1 reduces each spatial dimension by 0 (10x10 -> 10x10).
        # conv2 reduces each spatial dimension by 2 (10x10 -> 8x8).
        # conv3 reduces each spatial dimension by 4 (8x8 -> 4x4).
        # Total reduction: 0 + 2 + 4 = 6, so size = board_size - 6.
        size = board_size - 6

        # With the default 10x10 board, conv3 outputs 64 feature maps of size 4x4.
        # Flatten turns them into 64 * 4 * 4 = 1024 input features.
        # Fully connected layer: 1024 inputs -> 64 output neurons.
        # Each neuron learns to combine information from all 1024 inputs.
        self.fc1 = nn.Linear(64 * size * size, 64)

        # Converts 64 features into one Q-value per action.
        self.action_values = nn.Linear(64, n_actions)

        # Initialize weights for stable learning and biases at zero.
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.action_values]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Detect simple patterns, ReLU removes negative activations.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the feature maps into a single vector for each sample in the batch.
        x = torch.flatten(x, 1)

        # Combine features into a smaller set of features
        x = F.relu(self.fc1(x))

        # Return Q-values, one per action.
        return self.action_values(x)
