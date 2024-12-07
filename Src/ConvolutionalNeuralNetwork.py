import numpy as np

class ConvolutionalLayer:
    def __init__(self, num_filters, filter_size, input_depth, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_depth = input_depth
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)

    def _pad_input(self, input_data):
        if self.padding > 0:
            return np.pad(input_data,
                          ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant')
        return input_data

    def forward(self, input_data):
        self.input_data = self._pad_input(input_data)
        batch_size, depth, height, width = self.input_data.shape
        output_height = (height - self.filter_size) // self.stride + 1
        output_width = (width - self.filter_size) // self.stride + 1
        output = np.zeros((batch_size, self.num_filters, output_height, output_width))

        for n in range(batch_size):
            for f in range(self.num_filters):
                for i in range(0, output_height):
                    for j in range(0, output_width):
                        region = self.input_data[n, :,
                                                 i * self.stride:i * self.stride + self.filter_size,
                                                 j * self.stride:j * self.stride + self.filter_size]
                        output[n, f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        return np.maximum(0, output)  # ReLU activation


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input_data = input_data
        batch_size, depth, height, width = input_data.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, depth, output_height, output_width))

        for n in range(batch_size):
            for d in range(depth):
                for i in range(0, output_height):
                    for j in range(0, output_width):
                        region = input_data[n, d,
                                            i * self.stride:i * self.stride + self.pool_size,
                                            j * self.stride:j * self.stride + self.pool_size]
                        output[n, d, i, j] = np.max(region)
        return output


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, input_data):
        self.input_data = input_data
        output = np.dot(input_data, self.weights) + self.biases
        return np.maximum(0, output)  # ReLU activation


class LeNet5:
    def __init__(self, input_shape, num_classes):
        self.conv1 = ConvolutionalLayer(num_filters=6, filter_size=5, input_depth=input_shape[0], padding=2)
        self.pool1 = MaxPoolingLayer(pool_size=2, stride=2)
        self.conv2 = ConvolutionalLayer(num_filters=16, filter_size=5, input_depth=6)
        self.pool2 = MaxPoolingLayer(pool_size=2, stride=2)
        self.fc1 = FullyConnectedLayer(input_size=16 * 5 * 5, output_size=120)
        self.fc2 = FullyConnectedLayer(input_size=120, output_size=84)
        self.fc3 = FullyConnectedLayer(input_size=84, output_size=num_classes)

    def forward(self, input_data):
        x = self.conv1.forward(input_data)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        output = self.fc3.forward(x)  # No activation for output
        return output


