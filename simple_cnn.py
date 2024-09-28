import numpy as np
import pandas as pd
import os
from PIL import Image

class CNN:
    """
        Implements a Convolutional Neural Network (CNN) with basic layers such as 2D convolution, ReLU activation, and max pooling.
        
        The `CNN` class provides the following methods:
        
        - `__init__()`: Initializes the layers, weights, and biases of the CNN.
        - `conv2d(input, kernel, bias)`: Performs 2D convolution on the input using the provided kernel and bias.
        - `relu(input)`: Applies the ReLU activation function to the input.
        - `max_pool(input)`: Implements max pooling on the input.
        - `forward(input)`: Performs the forward pass of the CNN on the input.
    """

    def __init__(self):
        # Initialize layers, weights, and biases
        self.conv1_weights = np.random.randn(3, 3, 3, 16).astype(np.float32) * 0.01
        self.conv1_bias = np.zeros((16, 1), dtype=np.float32)
        self.conv2_weights = np.random.randn(3, 3, 16, 32).astype(np.float32) * 0.01
        self.conv2_bias = np.zeros((32, 1), dtype=np.float32)
        self.fc_weights = np.random.randn(32 * 22 * 22, 1).astype(np.float32) * 0.01
        self.fc_bias = np.zeros((1, 1), dtype=np.float32)

    def conv2d(self, input, kernel, bias):
        h_out = input.shape[1] - kernel.shape[0] + 1
        w_out = input.shape[2] - kernel.shape[1] + 1
        output = np.zeros((input.shape[0], h_out, w_out, kernel.shape[3]))
        
        for i in range(h_out):
            for j in range(w_out):
                output[:, i, j, :] = np.sum(input[:, i:i+kernel.shape[0], j:j+kernel.shape[1], :, np.newaxis] * 
                                            kernel[np.newaxis, :, :, :], axis=(1, 2, 3)) + bias.T
        return output

    def relu(self, input):
        # Implement ReLU activation
        return np.maximum(0, input)

    def max_pool(self, input):
        h_out, w_out = input.shape[1] // 2, input.shape[2] // 2
        output = np.zeros((input.shape[0], h_out, w_out, input.shape[3]))
        
        for i in range(h_out):
            for j in range(w_out):
                output[:, i, j, :] = np.max(input[:, 2*i:2*i+2, 2*j:2*j+2, :], axis=(1, 2))
        return output

    def forward(self, input):
        # First convolutional layer
        conv1 = self.conv2d(input, self.conv1_weights, self.conv1_bias)
        relu1 = self.relu(conv1)
        pool1 = self.max_pool(relu1)
        
        # Second convolutional layer
        conv2 = self.conv2d(pool1, self.conv2_weights, self.conv2_bias)
        relu2 = self.relu(conv2)
        pool2 = self.max_pool(relu2)
        
        # Flatten and fully connected layer
        flattened = pool2.reshape(pool2.shape[0], -1)
        output = np.dot(flattened, self.fc_weights) + self.fc_bias.T
        
        return output