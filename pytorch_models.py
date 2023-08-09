import numpy as np
from typing import List, Any
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


class OneDimensionalCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneDimensionalCNN, self).__init__()
        self.layer1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.layer2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
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
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

    def _calculate_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        dummy_output = self.conv_layers(dummy_input)
        return dummy_output.view(-1).size(0)
        
    
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
