import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeCNN(nn.Module):
    def __init__(self, input_channels=3, board_size=3, num_actions=9):
        super(TicTacToeCNN, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.input_channels = input_channels

        # Shared Body
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 filters
        self.bn2 = nn.BatchNorm2d(64)

        # Calculate the flattened size after convolutions
        # For a 3x3 board and the conv layers above, the output size remains 3x3
        # If you add more conv layers or change padding/stride, this needs adjustment.
        self.flattened_size = 64 * board_size * board_size

        # Policy Head
        self.fc_policy1 = nn.Linear(self.flattened_size, 128)
        self.fc_policy2 = nn.Linear(128, num_actions)

        # Value Head
        self.fc_value1 = nn.Linear(self.flattened_size, 128)
        self.fc_value2 = nn.Linear(128, 1) # Single scalar value

    def forward(self, x):
        # The input x from C++ is flat (batch, 18). Reshape it for the Conv2D layers.
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)

        # Shared body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x))) # if using conv3
        x = x.view(-1, self.flattened_size)  # Flatten for the fully connected layers

        # Policy head
        policy = F.relu(self.fc_policy1(x))
        policy = self.fc_policy2(policy) # Raw logits, softmax will be applied in loss or outside

        # Value head
        value = F.relu(self.fc_value1(x))
        value = torch.tanh(self.fc_value2(value)) # Output between -1 and 1

        return value, policy