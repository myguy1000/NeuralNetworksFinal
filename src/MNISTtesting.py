import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from scipy.ndimage import zoom
from ConvolutionalNeuralNetwork import LeNet5

# Load and preprocess the MNIST dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
X = X / 255.0  # Normalize pixel values
X = X.reshape(-1, 28, 28)  # Reshape to 28x28 images

# Resize to 32x32
X_resized = np.array([zoom(image, (32/28, 32/28)) for image in X])  # Rescale to 32x32
X_resized = X_resized[:, np.newaxis, :, :]  # Add channel dimension (N, 1, 32, 32)

# Convert labels to integers
y = y.astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=42)

# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

model = LeNet5(input_shape=(1, 32, 32), num_classes=10)

# Training parameters
epochs = 10
batch_size = 64
learning_rate = 0.01