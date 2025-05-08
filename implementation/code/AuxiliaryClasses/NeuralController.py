import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class NeuralController(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralController, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 16)  # Hidden layer with 16 neurons
        self.fc2 = nn.Linear(16, output_size)  # Output layer
        self.apply(initialize_weights)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)  # Output layer
        return torch.tanh(x) * 100  # Outputs actions for the robot

    def check_compatibility(self, w, z) -> Tuple[bool, float]:
        """Check if network matches environment dimensions."""
        # If they match exactly, OK
        if self.input_size == w and self.output_size == z:
            return True, 0.0

        # Otherwise return False plus the negative L₁-distance penalty
        penalty = - (abs(self.input_size - w) + abs(self.output_size - z))
        return False, penalty

    def adjust_to_environment(self, required_inputs, required_outputs):
        """Adjust network architecture to match environment dimensions."""


        # Adjust input layer if needed
        if self.input_size != required_inputs:
            self._adjust_input_layer(required_inputs)

        # Adjust output layer if needed
        if self.output_size != required_outputs:
            self._adjust_output_layer(required_outputs)

    def _adjust_input_layer(self, new_input_size):
        """Resize the input layer while preserving learned features."""
        old_weights = self.fc1.weight.data
        old_bias = self.fc1.bias.data

        # Create new layer with desired size
        new_layer = nn.Linear(new_input_size, self.fc1.out_features)

        # Initialize new weights
        if new_input_size > self.input_size:
            # Expand - copy existing weights and randomly initialize new ones
            new_layer.weight.data[:, :self.input_size] = old_weights
            new_layer.weight.data[:, self.input_size:] = torch.randn(
                (self.fc1.out_features, new_input_size - self.input_size)) * 0.01
            new_layer.bias.data = old_bias
        else:
            # Shrink - just keep the first N weights
            new_layer.weight.data = old_weights[:, :new_input_size]
            new_layer.bias.data = old_bias

        self.fc1 = new_layer
        self.input_size = new_input_size

    def _adjust_output_layer(self, new_output_size):
        """Resize the output layer while preserving learned features."""
        old_weights = self.fc2.weight.data
        old_bias = self.fc2.bias.data

        # Create new layer with desired size
        new_layer = nn.Linear(self.fc2.in_features, new_output_size)

        # Initialize new weights
        if new_output_size > self.output_size:
            # Expand - copy existing weights and randomly initialize new ones
            new_layer.weight.data[:self.output_size, :] = old_weights
            new_layer.weight.data[self.output_size:, :] = torch.randn(
                (new_output_size - self.output_size, self.fc2.in_features)) * 0.01
            new_layer.bias.data[:self.output_size] = old_bias
            new_layer.bias.data[self.output_size:] = torch.zeros(
                new_output_size - self.output_size)
        else:
            # Shrink - just keep the first N weights
            new_layer.weight.data = old_weights[:new_output_size, :]
            new_layer.bias.data = old_bias[:new_output_size]

        self.fc2 = new_layer
        self.output_size = new_output_size


# ---- Convert Weights to NumPy Arrays ----
def get_weights(model):
    """Extract weights from a PyTorch model as NumPy arrays."""
    return [p.detach().numpy() for p in model.parameters()]


# ---- Load Weights Back into a Model ----
def set_weights(model, new_weights):
    """Update PyTorch model weights from a list of NumPy arrays."""
    for param, new_w in zip(model.parameters(), new_weights):
        param.data = torch.tensor(new_w, dtype=torch.float32)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization
        nn.init.constant_(m.bias, 0.1)  # Small bias