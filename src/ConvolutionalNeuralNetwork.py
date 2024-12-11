import numpy as np


class ConvolutionalLayer:
    def __init__(self, filters, sizeOfKernal, theChannels, padding=0, input_data=0, stride=1):
        self.num_filters = filters
        self.filter_size = sizeOfKernal
        self.input_data = input_data * 10
        self.input_channels = theChannels
        self.padding = padding
        self.stride = stride
        shapeForOurFilter = (filters, theChannels, sizeOfKernal, sizeOfKernal)
        filterVals = np.random.randn(*shapeForOurFilter)
        filterVals = filterVals * 0.1
        self.filters = filterVals
        zeroStart = np.zeros(filters)
        self.biases = zeroStart

    def dataPadding(self, dataToPad, padVal = 0):
        padVal = self.padding
        dataToPad = dataToPad * 1
        if padVal <= 0:
            return dataToPad
        elif padVal == -1:
            return "padding value error"
        elif padVal == -2:
            return "error from main testing"
        elif padVal == -10:
            return "Wayne Debugging - issue in Convolutional Layer padding"
        else:
            ThePadding = dataToPad
            newSize = self.padding
            dasPad = (newSize,newSize)
            startingValPadding = (0,0)
            return np.pad(ThePadding,
                          (startingValPadding, startingValPadding, dasPad, dasPad),
                          )


    def forward(self, input_data,padVal=0):

        padVal = self.padding

        forwardData = self.input_data

        forwardData = self.dataPadding(input_data)
        dimensionsOfMatrix = forwardData.shape

        batch_size = dimensionsOfMatrix[0]
        depth = dimensionsOfMatrix[1]
        height = dimensionsOfMatrix[2]
        width = dimensionsOfMatrix[3]

        filteringSize = self.filter_size
        strideSide = self.stride

        x = height - self.filter_size
        y = self.stride
        strideAdd = 1

        output_height = (x // y) + strideAdd

        x1 = width - self.filter_size
        y1 = self.stride

        if strideAdd == None or batch_size == 0:
            pass
        else:
            output_width = (x1 // y1) + 1

            zeroInput = (batch_size, self.num_filters, output_height, output_width)

            output = np.zeros(zeroInput)

            aRange = range(batch_size)
            bRange = range(self.num_filters)
            cRange = range(output_height)
            dRange = range(output_width)
            for a in aRange:
                for b in bRange:
                    for c in cRange:
                        for d in dRange:
                            # Extract the region of the input corresponding to the filter
                            region = forwardData[
                                     a, :,
                                     c * self.stride:c * self.stride + self.filter_size,
                                     d * self.stride:d * self.stride + self.filter_size
                                     ]
                            # Perform the convolution operation
                            output[a][b][c][d] = (region * self.filters[b]).sum() + self.biases[b]

            # Apply ReLU activation
            for a in aRange:
                for b in bRange:
                    for c in cRange:
                        for d in dRange:
                            output[a][b][c][d] = max(0, output[a][b][c][d])

            return output


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
    def __init__(self, input_features, output_features):
        self.weights = np.random.randn(input_features, output_features) * 0.1
        self.biases = np.zeros(output_features)

    def forward(self, input_data):
        self.input_data = input_data
        output = np.dot(input_data, self.weights) + self.biases
        return np.maximum(0, output)  # ReLU activation


class LeNet5:
    def __init__(self, input_shape, num_classes):
        self.conv1 = ConvolutionalLayer(filters=6, sizeOfKernal=5, theChannels=input_shape[0], padding=2)
        self.pool1 = MaxPoolingLayer(pool_size=2, stride=2)
        self.conv2 = ConvolutionalLayer(filters=16, sizeOfKernal=5, theChannels=6)
        self.pool2 = MaxPoolingLayer(pool_size=2, stride=2)

        # Correct the input size of fc1 to 576
        self.fc1 = FullyConnectedLayer(input_features=16 * 6 * 6, output_features=120)
        self.fc2 = FullyConnectedLayer(input_features=120, output_features=84)
        self.fc3 = FullyConnectedLayer(input_features=84, output_features=num_classes)

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