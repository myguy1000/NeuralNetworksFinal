from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from ConvolutionalNeuralNetwork import *
from sklearn.model_selection import train_test_split
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Start of user hyperparamater setting

# Choose which dataset to run (only pick one)
mnist_f = False
cifar_f = True

# Define training parameters
epochs = 10
batch_size = 32
learning_rate = 0.001
reduced_data_size = .01 # ex. 0.01 means you're only using 1/100th of the data in both train and test
loss_track = []
accuracy_track = []

# End of user hyperparameter setting

num_channels = 0
model_selected = ""
if cifar_f:
    num_channels = 3
    data_selected = "cifar-10"
else: # mnist = True
    num_channels = 1
    data_selected = "mnist"
print("")
print("parameters selected")
print(f"data_selected: {data_selected}, epochs: {epochs}, learning_rate: {learning_rate}, reduced_data_size: {reduced_data_size * 100} percent")
print("")

# end of training parameters

# Load and preprocess the MNIST dataset using openml from scikit-learn
mnist = fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist["data"].to_numpy(), mnist["target"].to_numpy()

# Reshape and normalize images
X_mnist = X_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Resize images to 32x32
X_mnist = np.pad(X_mnist, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_mnist = encoder.fit_transform(y_mnist.reshape(-1, 1))

print("MNIST dataset loaded:")
print(f"Images shape: {X_mnist.shape}")
print(f"Labels shape: {y_mnist.shape}")
print("")
# Convert TensorFlow tensors to NumPy arrays (if needed)
X_mnist = np.array(X_mnist)
y_mnist = np.array(y_mnist)

# Load and preprocess the CIFAR-10 dataset using tf.keras.datasets.cifar10.load_data()
(X_cifar10_train, y_cifar10_train), (X_cifar10_test, y_cifar10_test) = cifar10.load_data()

# Normalize images
X_cifar10_train = X_cifar10_train.astype('float32') / 255.0
X_cifar10_test = X_cifar10_test.astype('float32') / 255.0

# One-hot encode labels
y_cifar10_train = tf.keras.utils.to_categorical(y_cifar10_train, 10)
y_cifar10_test = tf.keras.utils.to_categorical(y_cifar10_test, 10)

X_cifar10_train = np.array(X_cifar10_train)
y_cifar10_train = np.array(y_cifar10_train)
X_cifar10_test = np.array(X_cifar10_test)
y_cifar10_test = np.array(y_cifar10_test)


print("CIFAR-10 dataset loaded:")
print(f"ALL Training images shape: {X_cifar10_train.shape}")
print(f"ALL Training labels shape: {y_cifar10_train.shape}")
print(f"ALL Test images shape: {X_cifar10_test.shape}")
print(f"ALL Test labels shape: {y_cifar10_test.shape}")
print("")

## Training Loop

# Transpose data to match the input shape expected by LeNet5
X_train = []
y_train = []
X_test = []
y_teset = []
if mnist_f:
    X_train = np.transpose(X_mnist, (0, 3, 1, 2))  # (batch_size, channels, height, width)
    y_train = y_mnist
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1 - reduced_data_size, random_state=33)
    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=reduced_data_size, random_state=33)
else: # data = cifar-10
    X_train = np.transpose(X_cifar10_train, (0, 3, 1, 2))  # (batch_size, channels, height, width)
    X_test = np.transpose(X_cifar10_test, (0, 3, 1, 2))
    y_train = y_cifar10_train
    y_test = y_cifar10_test
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1 - reduced_data_size, random_state=42)
    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=reduced_data_size, random_state=42)

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1 - reduced_data_size, random_state=33)
#_, X_test, _, y_test = train_test_split(X_test, y_test, test_size=reduced_data_size, random_state=33)

print(f"REDUCED Training images shape: {X_train.shape}")
print(f"REDUCED Training labels shape: {y_train.shape}")
print(f"REDUCED Test images shape: {X_test.shape}")
print(f"REDUCED Test labels shape: {y_test.shape}")
print("")
# Initialize the LeNet5 model
input_shape = (num_channels, 32, 32)  # CIFAR-10 images: 3 channels, 32x32
num_classes = 10
model = LeNet5(input_shape, num_classes)

# Loss function: Cross-Entropy
def cross_entropy_loss(y_pred, y_true):
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]

run_start = time.time()
for epoch in range(epochs):
    start = time.time()
    print(f"Epoch {epoch+1}/{epochs}")
    # Shuffle the data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Mini-batch gradient descent
    batch_losses = []
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass
        y_pred = model.forward(X_batch)

        # Compute loss
        loss = cross_entropy_loss(y_pred, y_batch)
        batch_losses.append(loss)

        # Backward pass and parameter updates
        d_out = y_pred - y_batch  # Gradient of loss w.r.t. predictions
        model.backward(d_out, learning_rate)

    # Record and print the average loss for this epoch
    epoch_loss = np.mean(batch_losses)
    loss_track.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Calculate accuracy at the end of each epoch
    correct_predictions = 0
    for i in range(0, X_test.shape[0], batch_size):
        X_batch = X_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]

        y_pred = model.forward(X_batch)
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_batch, axis=1)
        correct_predictions += np.sum(predictions == labels)

    accuracy = correct_predictions / X_test.shape[0]
    accuracy_track.append(accuracy)
    print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    end = time.time()
    print("Training epoch time: ", end - start)

run_end = time.time()
print("Total training time: ", run_end - run_start)

for i in range(0, X_test.shape[0], batch_size):
    X_batch = X_test[i:i + batch_size]
    y_batch = y_test[i:i + batch_size]

    y_pred = model.forward(X_batch)
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_batch, axis=1)
    correct_predictions += np.sum(predictions == labels)

accuracy = correct_predictions / X_test.shape[0]
print(f"Test Accuracy: {accuracy:.2f}")

# Plot loss
plt.figure(figsize=(10, 6))
plt.bar(range(epochs), loss_track, label="Training Loss",color='blue')
plt.title("CIFAR Loss with 10 Epochs, 32 Batch Size, and 0.001 Learning Rate")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.bar(range(epochs), accuracy_track, label="Test Accuracy", color='blue')
plt.title("Accuracy with 10 Epochs, 32 Batch Size, and 0.001 Learning Rate")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()