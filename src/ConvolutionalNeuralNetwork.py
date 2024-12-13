import numpy as np

class ConvolutionalLayer:
    def __init__(self, filters, sizeOfKernal, theChannels, padding=0, input_data=0, stride=1):
        self.filterQuantity = filters  # number of filters
        self.filter_size = sizeOfKernal  # filter height/width
        self.input_data = input_data * 10  # store input data scaling
        self.input_channels = theChannels  # input channels
        self.padding = padding  # zero-padding size
        self.stride = stride  # stride for convolution
        shapeForOurFilter = (filters, theChannels, sizeOfKernal, sizeOfKernal)
        filterVals = np.random.randn(*shapeForOurFilter)
        filterVals = filterVals * 0.1  # scale filters
        self.filters = filterVals
        zeroStart = np.zeros(filters)
        self.biases = zeroStart

    def dataPadding(self, dataToPad, padVal = 0):
        padVal = self.padding
        dataToPad = dataToPad * 1  # ensure copy
        if padVal <= 0:
            return dataToPad
        elif padVal == -1:
            return "padding value error"
        elif padVal == -2:
            return "error from main testing"
        elif padVal == -10:
            return "Wayne Debugging - issue in Convolutional Layer padding"
        else:
            # pad input data
            ThePadding = dataToPad
            newSize = self.padding
            dasPad = (newSize,newSize)
            startingValPadding = (0,0)
            return np.pad(ThePadding,
                          (startingValPadding, startingValPadding, dasPad, dasPad),
                          )

    def forward(self, input_data, padVal=0):
        padVal = self.padding
        forwardData = self.input_data
        forwardData = self.dataPadding(input_data)
        dimensionsOfMatrix = forwardData.shape

        batch_size = dimensionsOfMatrix[0]  # number of samples
        depth = dimensionsOfMatrix[1]  # number of channels
        height = dimensionsOfMatrix[2]
        width = dimensionsOfMatrix[3]

        filteringSize = self.filter_size
        strideSide = self.stride

        x = height - self.filter_size
        y = self.stride
        strideAdd = 1

        height = (x // y) + strideAdd
        x1 = width - self.filter_size
        y1 = self.stride

        if strideAdd == None or batch_size == 0:
            pass
        else:
            outPutWidth = (x1 // y1) + 1
            zeroInput = (batch_size, self.filterQuantity, height, outPutWidth)
            output = np.zeros(zeroInput)

            # convolve input with filters
            aRange = range(batch_size)
            bRange = range(self.filterQuantity)
            cRange = range(height)
            dRange = range(outPutWidth)
            for a in aRange:
                for b in bRange:
                    for c in cRange:
                        for d in dRange:
                            cRowBegin = c * self.stride
                            cRowEnd = c * self.stride + self.filter_size
                            dRowBegin = d * self.stride
                            dRowEnd = d * self.stride + self.filter_size
                            convolutionArea = forwardData[
                                              a, :,
                                              cRowBegin:cRowEnd,
                                              dRowBegin:dRowEnd
                                              ]
                            if convolutionArea is not None:
                                # Compute the convolution as the elementwise product summed over all dimensions
                                total = np.sum(convolutionArea * self.filters[b])
                                output[a, b, c, d] = total + self.biases[b]
                            else:
                                print("ERROR IN convLayer convolution!")

            # Apply ReLU activation
            output = np.maximum(0, output)

            # store for backward
            self.last_input = input_data
            self.last_output = output

            return output

    def backward(self, d_out):
        # backprop for convolution
        batch_size = d_out.shape[0]
        filt_count = self.filterQuantity
        padded_input = self.dataPadding(self.last_input)
        padded_input_shape = padded_input.shape
        in_channels = padded_input_shape[1]
        out_height = d_out.shape[2]
        out_width = d_out.shape[3]

        # ReLU gradient
        relu_mask = (self.last_output > 0).astype(float)
        d_out = d_out * relu_mask

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input_padded = np.zeros_like(padded_input)

        # Compute gradients for inputs
        rotated_filters = np.flip(self.filters, axis=(2, 3))  # Flip filters for gradient computation
        d_input_padded = np.zeros_like(padded_input)

        # Reshape d_out for broadcasting
        d_out_reshaped = d_out.transpose(1, 0, 2, 3).reshape(filt_count, batch_size, out_height, out_width)

        # Accumulate gradients for each filter
        for b in range(filt_count):
            for c in range(out_height):
                for d_i in range(out_width):
                    h_start = c * self.stride
                    h_end = h_start + self.filter_size
                    w_start = d_i * self.stride
                    w_end = w_start + self.filter_size

                    # Add contributions from this filter
                    d_input_padded[:, :, h_start:h_end, w_start:w_end] += (
                            d_out_reshaped[b, :, c, d_i][:, np.newaxis, np.newaxis, np.newaxis]
                            * rotated_filters[b][np.newaxis, :, :, :]
                    )

        # remove padding from input gradient
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_input_padded

        self.d_filters = d_filters
        self.d_biases = d_biases

        return d_input

    def update_params(self, learning_rate):
        # Update filters and biases
        self.filters -= learning_rate * self.d_filters
        self.biases -= learning_rate * self.d_biases


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size  # pooling window size
        self.stride = stride  # stride for pooling

    def forward(self, input_data, maxPoolingPadding=0):
        maxPoolingPadding = maxPoolingPadding
        forwardData = input_data
        dimensionsOfMatrix = forwardData.shape

        batch_size = dimensionsOfMatrix[0]
        depth = dimensionsOfMatrix[1]
        height = dimensionsOfMatrix[2]
        width = dimensionsOfMatrix[3]

        poolSize = self.pool_size
        strideSide = self.stride

        x = height - self.pool_size
        y = self.stride
        strideAdd = 1

        height = (x // y) + strideAdd
        x1 = width - self.pool_size
        y1 = self.stride

        if strideAdd == None or batch_size == 0:
            pass
        else:
            outPutWidth = (x1 // y1) + 1
            zeroInput = (batch_size, depth, height, outPutWidth)
            output = np.zeros(zeroInput)

            # store max indices for backward
            self.argmax_positions = np.zeros((batch_size, depth, height, outPutWidth, 2), dtype=int)

            aRange = range(batch_size)
            bRange = range(depth)
            cRange = range(height)
            dRange = range(outPutWidth)

            # find max in each pooling region
            for a in aRange:
                for b in bRange:
                    for c in cRange:
                        for d in dRange:
                            cRowBegin = c * self.stride
                            cRowEnd = c * self.stride + self.pool_size
                            dRowBegin = d * self.stride
                            dRowEnd = d * self.stride + self.pool_size
                            poolingArea = forwardData[
                                          a, b,
                                          cRowBegin:cRowEnd,
                                          dRowBegin:dRowEnd
                                          ]

                            if poolingArea is not None:
                                maxVal = -10000
                                max_x = 0
                                max_y = 0
                                xMax = range(poolingArea.shape[0])
                                yMax = range(poolingArea.shape[1])
                                for x_index in xMax:
                                    for y_index in yMax:
                                        if poolingArea[x_index][y_index] > maxVal:
                                            maxVal = poolingArea[x_index][y_index]
                                            max_x = x_index
                                            max_y = y_index
                                output[a][b][c][d] = maxVal
                                self.argmax_positions[a, b, c, d] = (cRowBegin + max_x, dRowBegin + max_y)
                            else:
                                print("ERROR IN poolingLayer max pooling!")

            self.last_input = input_data
            self.last_output = output

            return output

    def backward(self, d_out):
        # backprop through max pooling
        batch_size = d_out.shape[0]
        depth = d_out.shape[1]
        out_height = d_out.shape[2]
        out_width = d_out.shape[3]

        d_input = np.zeros_like(self.last_input)
        # distribute gradients back to where max was found
        for a in range(batch_size):
            for b in range(depth):
                for c in range(out_height):
                    for d_i in range(out_width):
                        (h_idx, w_idx) = self.argmax_positions[a, b, c, d_i]
                        d_input[a, b, h_idx, w_idx] += d_out[a, b, c, d_i]

        return d_input

    def update_params(self, learning_rate):
        # no params in max pooling
        pass


class FullyConnectedLayer:
    def __init__(self, input_features, output_features, scaleFactor = 0.1, padding = 0):
        multVal = scaleFactor
        weight1 = input_features
        weight2 = output_features
        self.weights = np.random.randn(weight1, weight2) * multVal
        biasFCL = output_features
        self.biases = np.zeros(biasFCL)

    def forward(self, input_data, scalingFactor = 1):
        scaling = scalingFactor
        self.input_data = input_data * scaling
        dot1 = input_data * scaling
        dot2 = self.weights * scaling
        output = np.dot(dot1, dot2)
        forwardBias = self.biases
        output += forwardBias

        # ReLU activation
        reluOutput = output.copy()
        xMax = range(reluOutput.shape[0])
        yMax = range(reluOutput.shape[1])
        for x_i in xMax:
            for y_i in yMax:
                if reluOutput[x_i][y_i] < 0:
                    if reluOutput is not None:
                        reluOutput[x_i][y_i] = 0
                    else:
                        print("ERROR in FCL, ConNeuralNetwork")

        self.last_input = input_data
        self.last_output = reluOutput
        return reluOutput

    def backward(self, d_out):
        # backprop through fully connected layer
        relu_mask = (self.last_output > 0).astype(float)
        d_out = d_out * relu_mask

        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.last_input)

        # Gradients with respect to biases
        d_biases += np.sum(d_out, axis=0)

        # Gradients with respect to weights
        d_weights += np.dot(self.input_data.T, d_out)

        # Gradients with respect to inputs
        d_input = np.dot(d_out, self.weights.T)

        self.d_weights = d_weights
        self.d_biases = d_biases

        return d_input

    def update_params(self, learning_rate):
        # update weights and biases
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i][j] -= learning_rate * self.d_weights[i][j]

        for i in range(self.biases.shape[0]):
            self.biases[i] -= learning_rate * self.d_biases[i]


class LeNet5:
    def __init__(self, input_shape, num_classes):
        # create layers of LeNet5
        self.conv1 = ConvolutionalLayer(filters=6, sizeOfKernal=5, theChannels=input_shape[0], padding=2)
        self.pool1 = MaxPoolingLayer(pool_size=2, stride=2)
        self.conv2 = ConvolutionalLayer(filters=16, sizeOfKernal=5, theChannels=6)
        self.pool2 = MaxPoolingLayer(pool_size=2, stride=2)

        # fc layers
        self.fc1 = FullyConnectedLayer(input_features=16 * 6 * 6, output_features=120)
        self.fc2 = FullyConnectedLayer(input_features=120, output_features=84)
        self.fc3 = FullyConnectedLayer(input_features=84, output_features=num_classes)

    def forward(self, input_data):
        # forward pass through LeNet5
        x = self.conv1.forward(input_data)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        output = self.fc3.forward(x)  # output layer
        self.last_input = input_data
        self.last_output = output
        return output

    def backward(self, d_out, learning_rate=0.01):
        # backprop through entire network
        d_out = self.fc3.backward(d_out)
        d_out = self.fc2.backward(d_out)
        d_out = self.fc1.backward(d_out)

        batch_size = d_out.shape[0]
        d_out = d_out.reshape(batch_size, 16, 6, 6)

        d_out = self.pool2.backward(d_out)
        d_out = self.conv2.backward(d_out)
        d_out = self.pool1.backward(d_out)
        d_out = self.conv1.backward(d_out)

        # update all learnable parameters
        self.conv1.update_params(learning_rate)
        self.conv2.update_params(learning_rate)
        self.fc1.update_params(learning_rate)
        self.fc2.update_params(learning_rate)
        self.fc3.update_params(learning_rate)

        return d_out
