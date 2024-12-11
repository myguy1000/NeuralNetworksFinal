import numpy as np


class ConvolutionalLayer:
    def __init__(self, filters, sizeOfKernal, theChannels, padding=0, input_data=0, stride=1):
        self.filterQuantity = filters
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

        height = (x // y) + strideAdd

        x1 = width - self.filter_size
        y1 = self.stride

        if strideAdd == None or batch_size == 0:
            pass
        else:
            outPutWidth = (x1 // y1) + 1

            zeroInput = (batch_size, self.filterQuantity, height, outPutWidth)

            output = np.zeros(zeroInput)

            aRange = range(batch_size)
            bRange = range(self.filterQuantity)
            cRange = range(height)
            dRange = range(outPutWidth)
            for a in aRange:
                for b in bRange:
                    for c in cRange:
                        for d in dRange:
                            # Extract the region of the input corresponding to the filter
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
                                total = 0
                                for x in range(convolutionArea.shape[0]):
                                    for y in range(convolutionArea.shape[1]):
                                        for z in range(convolutionArea.shape[2]):
                                            total += convolutionArea[x][y][z] * self.filters[b][x][y][z]

                                output[a][b][c][d] = total + self.biases[b]
                            else:
                                print("ERROR IN convLayer convolution!")

            # Apply ReLU activation
            for a in aRange:
                for b in bRange:
                    for c in cRange:
                        for d in dRange:
                            reLuMax = output[a][b][c][d]
                            if reLuMax is not None:
                                output[a][b][c][d] = max(0, reLuMax)
                            else:
                                print("We have an error in ConVLayer")

            return output


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data, maxPoolingPadding=0):

        maxPoolingPadding = maxPoolingPadding

        forwardData = input_data

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

            aRange = range(batch_size)
            bRange = range(depth)
            cRange = range(height)
            dRange = range(outPutWidth)

            for a in aRange:
                for b in bRange:
                    for c in cRange:
                        for d in dRange:
                            # Extract the region of the input corresponding to the pooling area
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
                                maxVal = -10000  # Start with a very small number
                                xMax = range(poolingArea.shape[0])
                                yMax = range(poolingArea.shape[1])
                                for x in xMax:
                                    for y in yMax:
                                        if poolingArea[x][y] > maxVal:
                                            maxVal = poolingArea[x][y]
                                output[a][b][c][d] = maxVal
                            else:
                                print("ERROR IN poolingLayer max pooling!")

            return output


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

        # Replace np.maximum with a custom ReLU implementation
        reluOutput = output.copy()
        xMax = range(reluOutput.shape[0])
        yMax = range(reluOutput.shape[1])
        for x in xMax:
            for y in yMax:
                if reluOutput[x][y] < 0:
                    if reluOutput is not None:
                        reluOutput[x][y] = 0
                    else:
                        print("ERROR in FCL, ConNeuralNetwork")

        return reluOutput


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