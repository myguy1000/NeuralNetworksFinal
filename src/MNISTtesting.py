import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from scipy.ndimage import zoom
from ConvolutionalNeuralNetwork import LeNet5
import matplotlib.pyplot as plt

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

# Placeholder for tracking loss
losses = []

for epoch in range(epochs):
    # Shuffle training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train_encoded = y_train_encoded[indices]

    epoch_loss = 0

    # Process batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train_encoded[i:i+batch_size]

        # Forward pass
        predictions = model.forward(X_batch)

        # Compute loss
        loss = np.mean((predictions - y_batch) ** 2)
        epoch_loss += loss

        # Compute gradient of loss
        d_out = 2 * (predictions - y_batch) / len(y_batch)

        # Backward pass and update weights
        model.backward(d_out)

    avg_loss = epoch_loss / (len(X_train) // batch_size)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Plot

plt.plot(range(1, epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.show()