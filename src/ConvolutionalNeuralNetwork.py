import numpy as np

class ConvolutionalLayer:
    def __init__(self, filters, sizeOfKernal, theChannels, padding=0, input_data=0, stride=1):
        self.filterQuantity = filters
        self.filter_size = sizeOfKernal
        self.input_data = input_data * 10
        self.inputChannels = theChannels
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
        #debugging to prevent errors
        if padVal <= 0:
            return dataToPad
        elif padVal == -1:
            return "padding value error"
        elif padVal == -2:
            return "error from main testing"
        elif padVal == -10:
            return "Wayne Debugging - issue in Convolutional Layer padding"
        else:
            #here is where we pad the data as long as we arent getting and error with the data
            ThePadding = dataToPad
            newSize = self.padding
            dasPad = (newSize,newSize)
            startingValPadding = (0,0)
            return np.pad(ThePadding,
                          (startingValPadding, startingValPadding, dasPad, dasPad),
                          )

    def forward(self, inputD, padVal=0):

        padVal = self.padding
        forwardData = self.input_data
        forwardData = self.dataPadding(inputD)
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

            #Here is where we perform convolution that is specified from the lecture slides
            #As well as the
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
            self.last_input = inputD
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

        #calculations for our relu function
        relu_mask = (self.last_output > 0).astype(float)
        d_out = d_out * relu_mask

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input_padded = np.zeros_like(padded_input)

        #making filters/updating them and padding
        rotated_filters = np.flip(self.filters, axis=(2, 3))  # Flip filters for gradient computation
        d_input_padded = np.zeros_like(padded_input)

        #Rechape from transpose/reschape functions
        d_out_reshaped = d_out.transpose(1, 0, 2, 3).reshape(filt_count, batch_size, out_height, out_width)

        #Looping for the d_input
        for b in range(filt_count):
            for c in range(out_height):
                for d_i in range(out_width):
                    h_start = c * self.stride
                    h_end = h_start + self.filter_size
                    w_start = d_i * self.stride
                    w_end = w_start + self.filter_size

                    #Adding to the d_input the necessary vals
                    d_input_padded[:, :, h_start:h_end, w_start:w_end] += (
                            d_out_reshaped[b, :, c, d_i][:, np.newaxis, np.newaxis, np.newaxis]
                            * rotated_filters[b][np.newaxis, :, :, :]
                    )

        #here we just want to get rid of the padding to get it ready to ship like a christmas gift
        #excited for the break lol
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        else:
            d_input = d_input_padded

        self.d_filters = d_filters
        self.d_biases = d_biases

        return d_input

    def update_params(self, learning_rate):
        # Perform our update here for bias and filters
        self.filters -= learning_rate * self.d_filters
        self.biases -= learning_rate * self.d_biases


class MaxPoolingLayer:
    def __init__(self, poolSize, stride):
        self.pool_size = poolSize
        self.stride = stride

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

            self.argmax_positions = np.zeros((batch_size, depth, height, outPutWidth, 2), dtype=int)

            aRange = range(batch_size)
            bRange = range(depth)
            cRange = range(height)
            dRange = range(outPutWidth)

            #Here we are just looking for the max in the region, gave some runtime errors in the past, still fixing
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

    def update_params(self, learning_rate, scale = 1):
        #why do we have this? Need to discuss
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


        d_biases += np.sum(d_out, axis=0)


        d_weights += np.dot(self.input_data.T, d_out)


        d_input = np.dot(d_out, self.weights.T)

        self.d_weights = d_weights
        self.d_biases = d_biases

        return d_input

    def update_params(self, learningRate):
        # here we look to update our weights as well as the bias
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i][j] -= learningRate * self.d_weights[i][j]

        for i in range(self.biases.shape[0]):
            self.biases[i] -= learningRate * self.d_biases[i]


class LeNet5:
    def __init__(self, shapeForInput, classQuantity, scale = 1):
        #Here we are making our layers, we have them generate as specified in the document
        self.conv1 = ConvolutionalLayer(filters=6, sizeOfKernal=5, theChannels=shapeForInput[0], padding=2)
        self.pool1 = MaxPoolingLayer(poolSize=2, stride=2)
        self.conv2 = ConvolutionalLayer(filters=16, sizeOfKernal=5, theChannels=6)
        self.pool2 = MaxPoolingLayer(poolSize=2, stride=2)

        inputScaled = shapeForInput * scale


        self.fc1 = FullyConnectedLayer(input_features=16 * 6 * 6, output_features=120)
        self.fc2 = FullyConnectedLayer(input_features=120, output_features=84)
        self.fc3 = FullyConnectedLayer(input_features=84, output_features=classQuantity)

    def forward(self, inputNumbers, scale = 1):
        scaledInput = inputNumbers * scale
        x = self.conv1.forward(inputNumbers)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        output = self.fc3.forward(x)
        self.last_input = inputNumbers
        self.last_output = output
        return output

    def backward(self, outputForD, learning_rate=0.01, scale = 1):
        # backprop through entire network
        scaledD = outputForD * scale
        outputForD = self.fc3.backward(outputForD)
        outputForD = self.fc2.backward(outputForD)
        outputForD = self.fc1.backward(outputForD)

        batch_size = outputForD.shape[0]
        outputForD = outputForD.reshape(batch_size, 16, 6, 6)

        outputForD = self.pool2.backward(outputForD)
        outputForD = self.conv2.backward(outputForD)
        outputForD = self.pool1.backward(outputForD)
        outputForD = self.conv1.backward(outputForD)

        #here we need to update our parameters if they need to be changed
        self.conv1.update_params(learning_rate)
        self.conv2.update_params(learning_rate)
        self.fc1.update_params(learning_rate)
        self.fc2.update_params(learning_rate)
        self.fc3.update_params(learning_rate)

        return outputForD
