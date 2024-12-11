from ConvolutionalNeuralNetwork import *
# Example usage:
if __name__ == "__main__":
    input_data = np.random.randn(1, 1, 32, 32)  # Batch size = 1, grayscale image (1 channel)
    lenet = LeNet5(input_shape=(1, 32, 32), num_classes=10)
    output = lenet.forward(input_data)
    print("Output shape:", output.shape)  # Should be (1, 10)
