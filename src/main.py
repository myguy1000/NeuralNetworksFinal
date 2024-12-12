import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
#import tensorflow as tf
#from tensorflow.keras.datasets import cifar10
from ConvolutionalNeuralNetwork import *

# Load and preprocess the MNIST dataset using openml from scikit-learn
mnist = fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist["data"].to_numpy(), mnist["target"].to_numpy()

# Reshape and normalize images
X_mnist = X_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Resize images to 32x32
X_mnist = np.pad(X_mnist, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# One-hot encode labels
encoder = OneHotEncoder()
y_mnist = encoder.fit_transform(y_mnist.reshape(-1, 1))

print("MNIST dataset loaded:")
print(f"Images shape: {X_mnist.shape}")
print(f"Labels shape: {y_mnist.shape}")

## Load CIFAR-10 dataset
#cifar10 = fetch_openml('CIFAR_10', version=1)
#
## Extract data and labels
#X_cifar10, y_cifar10 = cifar10['data'], cifar10['target']
#
## Reshape and normalize images
#X_cifar10 = X_cifar10.reshape(-1, 32, 32, 3).astype('float32') / 255.0
#
## One-hot encode labels
#y_cifar10 = tf.keras.utils.to_categorical(y_cifar10.astype(int), 10)
#
#print("CIFAR-10 dataset loaded:")
#print(f"Images shape: {X_cifar10.shape}")
#print(f"Labels shape: {y_cifar10.shape}")
#
## Convert TensorFlow tensors to NumPy arrays (if needed)
#X_mnist = np.array(X_mnist)
#y_mnist = np.array(y_mnist)
#X_cifar10_train = np.array(X_cifar10_train)
#y_cifar10_train = np.array(y_cifar10_train)
#X_cifar10_test = np.array(X_cifar10_test)
#y_cifar10_test = np.array(y_cifar10_test)
#
#
## example run
input_data = np.random.randn(1, 1, 32, 32)  # Batch size = 1, grayscale image (1 channel)
lenet = LeNet5(input_shape=(1, 32, 32), num_classes=10)
output = lenet.forward(input_data)
print("Output shape:", output.shape)  # Should be (1, 10)
