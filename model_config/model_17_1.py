import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):
    """
    CNN that predicts a Q-value (score) for every possible action.
    Input: (batch, frames, board_size, board_size)
    """

    def __init__(self, board_size=10, frames=2, n_actions=4):
        super().__init__()

        # Input: 2 frames, Output: 16 feature maps.
        # 3x3 filters scan the board
        # padding keeps its size at 10x10.
        self.conv1 = nn.Conv2d(frames, 16, kernel_size=3, padding=1)

        # Input: 16 maps, Output: 32 maps.
        # 3x3 filters detect patterns from the previous layer.
        # board shrinks from 10x10 to 8x8.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        # Input: 32 maps, Output: 64 maps.
        # 5x5 filters detect larger patterns.
        # board shrinks from 8x8 to 4x4.
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)

        # conv1: 3x3 kernel + padding=1 → reduces size by 0
        # conv2: 3x3 kernel, no padding → reduces size by 2
        # conv3: 5x5 kernel, no padding → reduces size by 4
        # Total reduction: 0 + 2 + 4 = 6 (10x10 → 4x4)
        size = board_size - 6

        # conv3 outputs 64 feature maps, each 4x4 (size = 4).
        # Flatten turns them into 64 * 4 * 4 = 1024 input features.
        # Fully connected layer: 1024 inputs → 64 output neurons.
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
