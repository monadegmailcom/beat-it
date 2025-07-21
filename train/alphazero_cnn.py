import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    The core building block of a ResNet. It contains a 'skip connection'
    that adds the input of the block to its output. This helps combat
    vanishing gradients and allows for much deeper networks.
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # The 'skip connection'
        residual = x
        # The main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Add the input to the output of the convolutions
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroCNN(nn.Module):
    def __init__(
            self, board_size, num_actions, input_channels, num_res_blocks,
            res_block_channels, fc_hidden_size):
        """
        A configurable ResNet-based architecture inspired by AlphaZero.

        Args:
        board_size (int): The width and height of the board.
        num_actions (int): The size of the policy output.
        input_channels (int): Number of input planes. For UTTT, this could be:
        num_res_blocks (int): The number of residual blocks in the network
            body.
        res_block_channels (int): The number of channels used in the residual
            blocks.
        fc_hidden_size (int): The number of neurons in the hidden layer of the
            value and policy heads.
        """
        super(AlphaZeroCNN, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.input_channels = input_channels

        # --- Network Body ---
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, res_block_channels, kernel_size=3, padding=1,
                bias=False),
            nn.BatchNorm2d(res_block_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(res_block_channels)
                                          for _ in range(num_res_blocks)])

        # --- Value Head ---
        self.value_head = nn.Sequential(
            nn.Conv2d(res_block_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1), nn.ReLU(inplace=True), nn.Flatten(),
            nn.Linear(1 * board_size * board_size, fc_hidden_size), nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, 1), nn.Tanh()
        )

        # --- Policy Head ---
        self.policy_head = nn.Sequential(
            nn.Conv2d(res_block_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2), nn.ReLU(inplace=True), nn.Flatten(),
            nn.Linear(2 * board_size * board_size, fc_hidden_size), nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, num_actions)
        )

    def forward(self, x):
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        return self.value_head(x), self.policy_head(x)
