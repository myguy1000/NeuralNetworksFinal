
# Example usage:
if __name__ == "__main__":
    # Input: batch size of 1, depth of 1 (grayscale), 32x32 image
    input_data = np.random.randn(1, 1, 32, 32)
    lenet = LeNet5(input_shape=(1, 32, 32), num_classes=10)
    output = lenet.forward(input_data)
    print("Output shape:", output.shape)
