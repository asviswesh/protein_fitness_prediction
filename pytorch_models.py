import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

# CNN v2
class MLDECNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, hidden_size, dropout_prob):
        super(MLDECNN, self).__init__()
        padding = (kernel_size - 1) // 2
        self.layer1 = nn.Conv1d(in_channels=1,
                                out_channels=32,
                                kernel_size=kernel_size,
                                padding=padding)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv1d(in_channels=32,
                                out_channels=32,
                                kernel_size=kernel_size,
                                padding=padding)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Conv1d(in_channels=32,
                                out_channels=32,
                                kernel_size=kernel_size,
                                padding=padding)
        self.flatten = nn.Flatten()
        fc_input_size = self._calculate_fc_input_size(input_size)
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def _calculate_fc_input_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)
            conv_output = self.layer1(dummy_input)
            conv_output = self.relu1(conv_output)
            conv_output = self.layer2(conv_output)
            conv_output = self.relu2(conv_output)
            conv_output = self.layer3(conv_output)
            conv_output = self.flatten(conv_output)
            fc_input_size = conv_output.view(conv_output.size(0), -1).size(1)
        return fc_input_size

    def forward(self, x):
        x = x.unsqueeze(1)
        conv_output = self.layer1(x)
        conv_output = self.relu1(conv_output)
        conv_output = self.layer2(conv_output)
        conv_output = self.relu2(conv_output)
        conv_output = self.layer3(conv_output)
        conv_output = self.flatten(conv_output)
        conv_output = self.fc1(conv_output)
        conv_output = self.dropout(conv_output)
        conv_output = self.fc2(conv_output)
        return conv_output

# CNN v1
class OneDimensionalCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneDimensionalCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        fc_input_size = self._calculate_fc_input_size(input_size)

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Reshape input to (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def _calculate_fc_input_size(self, input_size):
        # Simulate a forward pass through the convolutional layers to determine the output size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)
            conv_output = self.conv_layers(dummy_input)
            fc_input_size = conv_output.view(conv_output.size(0), -1).size(1)
        return fc_input_size

# Feed forward Neural Network with two hidden layers.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_l1, hidden_size_l2, num_output):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_l1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_l1, hidden_size_l2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_l2, num_output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
