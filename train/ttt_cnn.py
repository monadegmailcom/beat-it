import torch
import torch.nn as nn
import torch.nn.functional as F


class TicTacToeCNN(nn.Module):
    def __init__(
            self, input_channels, board_size, num_actions, conv_channels,
            fc_hidden_size):
        super(TicTacToeCNN, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.input_channels = input_channels

        # Shared Body
        self.conv1 = nn.Conv2d(
            input_channels, conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.conv2 = nn.Conv2d(
            conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels * 2)

        # Calculate the flattened size after convolutions
        # For a 3x3 board and the conv layers above, the output size remains
        # 3x3
        # If you add more conv layers or change padding/stride, this needs
        # adjustment.
        self.flattened_size = (conv_channels * 2) * board_size * board_size

        # Policy Head
        self.fc_policy1 = nn.Linear(self.flattened_size, fc_hidden_size)
        self.fc_policy2 = nn.Linear(fc_hidden_size, num_actions)

        # Value Head
        self.fc_value1 = nn.Linear(self.flattened_size, fc_hidden_size)
        self.fc_value2 = nn.Linear(fc_hidden_size, 1)  # Single scalar value

    def forward(self, x):
        # The input x from C++ is flat (batch, 18). Reshape it for the
        # Conv2D layers.
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)

        # Shared body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # Flatten for the fully connected layers
        x = x.view(-1, self.flattened_size)

        # Policy head
        policy = F.relu(self.fc_policy1(x))
        # Raw logits, softmax will be applied in loss or outside
        policy = self.fc_policy2(policy)

        # Value head
        value = F.relu(self.fc_value1(x))
        value = torch.tanh(self.fc_value2(value))  # Output between -1 and 1

        return value, policy
