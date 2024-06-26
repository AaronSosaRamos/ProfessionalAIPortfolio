# -*- coding: utf-8 -*-
"""CB001 - Chapter 29.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ciuARkwemdLBF2X-DbGaN-Wi9rEKwGNu

# Computing Gradients Using Autodiff
"""

import tensorflow as tf

def f(w1, w2):
 return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = 5, 3
eps = 1e-6
(f(w1 + eps, w2) - f(w1, w2)) / eps
(f(w1, w2 + eps) - f(w1, w2)) / eps

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
 z = f(w1, w2)
gradients = tape.gradient(z, [w1, w2])

with tf.GradientTape() as tape:
 z = f(w1, w2)
 dz_dw1 = tape.gradient(z, w1) # => tensor 36.0

dz_dw1

with tf.GradientTape(persistent=True) as tape:
 z = f(w1, w2)
dz_dw1 = tape.gradient(z, w1) # => tensor 36.0
dz_dw2 = tape.gradient(z, w2) # => tensor 10.0, works fine now!
del tape

dz_dw1

dz_dw2

c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
 z = f(c1, c2)
gradients = tape.gradient(z, [c1, c2])

gradients

with tf.GradientTape() as tape:
 tape.watch(c1)
 tape.watch(c2)
 z = f(c1, c2)
gradients = tape.gradient(z, [c1, c2])

gradients

with tf.GradientTape(persistent=True) as hessian_tape:
 with tf.GradientTape() as jacobian_tape:
  z = f(w1, w2)
 jacobians = jacobian_tape.gradient(z, [w1, w2])
hessians = [hessian_tape.gradient(jacobian, [w1, w2])
 for jacobian in jacobians]
del hessian_tape

hessians

def f(w1, w2):
 return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)
with tf.GradientTape() as tape:
 z = f(w1, w2) # same result as without stop_gradient()
gradients = tape.gradient(z, [w1, w2]) # => returns [tensor 30., None]

gradients

@tf.custom_gradient
def my_better_softplus(z):
 exp = tf.exp(z)
 def my_softplus_gradients(grad):
  return grad / (1 + 1 / exp)
 return tf.math.log(exp + 1), my_softplus_gradients

x = tf.Variable([100.])
with tf.GradientTape() as tape:
  z = my_better_softplus(x)

tape.gradient(z, [x])

"""# Custom Training Loops

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Set up data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Custom training loop
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')

# Train the model
train(model, train_loader, criterion, optimizer, epochs=5)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

# Load and preprocess the MNIST dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define the model architecture with L2 regularization
l2_reg = regularizers.l2(0.05)
model = keras.models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(30, activation="elu", kernel_initializer="he_normal", kernel_regularizer=l2_reg),
    layers.Dense(10, kernel_regularizer=l2_reg)
])

# Custom function to generate random batches
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

# Custom function to print training progress
def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)

# Training parameters
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.SparseCategoricalAccuracy()]

# Training loop
for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train, y_train, batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = loss_fn(y_batch, y_pred)
            loss = tf.add_n([main_loss] + model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)

        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)

    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

# Load and preprocess the MNIST dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define the model architecture with L2 regularization
l2_reg = regularizers.l2(0.05)
model = keras.models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(30, activation="elu", kernel_initializer="he_normal", kernel_regularizer=l2_reg),
    layers.Dense(10, kernel_regularizer=l2_reg)
])

# Custom function to generate random batches
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

# Custom function to print training progress
def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)

# Training parameters
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.SparseCategoricalAccuracy()]

# Training loop
for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train, y_train, batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = loss_fn(y_batch, y_pred)
            loss = tf.add_n([main_loss] + model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Apply weight constraints if defined
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)

        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)

    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()

"""# TensorFlow Functions and Graphs

"""

def cube(x):
   return x ** 3

cube(2)

cube(tf.constant(2.0))

tf_cube = tf.function(cube)
tf_cube

tf_cube(2)

tf_cube(tf.constant(2.0))

@tf.function
def tf_cube(x):
  return x ** 3

tf_cube.python_function(2)

"""# Autograph and Tracing and TF Function Rules

"""

tf.autograph.to_code(tf_cube.python_function)

import tensorflow as tf

def compute_jacobian(func, x):
    """
    Compute the Jacobian matrix of a function `func` at point `x` using TensorFlow.

    Args:
        func: A callable function that takes a TensorFlow tensor `x` as input.
        x: A TensorFlow tensor representing the point at which to compute the Jacobian.

    Returns:
        jacobian: A TensorFlow tensor representing the Jacobian matrix.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = func(x)
    jacobian = tape.jacobian(y, x)
    return jacobian

def compute_hessian(func, x):
    """
    Compute the Hessian matrix of a function `func` at point `x` using TensorFlow.

    Args:
        func: A callable function that takes a TensorFlow tensor `x` as input.
        x: A TensorFlow tensor representing the point at which to compute the Hessian.

    Returns:
        hessian: A TensorFlow tensor representing the Hessian matrix.
    """
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            y = func(x)
        gradients = tape1.gradient(y, x)
    hessian = tape2.jacobian(gradients, x)
    return hessian

# Example usage:
# Define a function for which to compute Jacobian and Hessian
def quadratic_function(x):
    return tf.reduce_sum(x**2)

# Define the point at which to compute Jacobian and Hessian
x_value = tf.constant([1.0, 2.0])

# Compute Jacobian
jacobian_result = compute_jacobian(quadratic_function, x_value)
print("Jacobian:")
print(jacobian_result)

# Compute Hessian
hessian_result = compute_hessian(quadratic_function, x_value)
print("\nHessian:")
print(hessian_result)

# Convert functions to autograph-compatible code
jacobian_code = tf.autograph.to_code(compute_jacobian)
hessian_code = tf.autograph.to_code(compute_hessian)

print("\nAutograph-compatible code for compute_jacobian:")
print(jacobian_code)

print("\nAutograph-compatible code for compute_hessian:")
print(hessian_code)

"""#Loading and Preprocessing Data with TensorFlow

# The Data API
"""

X = tf.range(10)

dataset = tf.data.Dataset.from_tensor_slices(X)

dataset

for item in dataset:
  print(item)

"""# Chaining Transformations"""

dataset = dataset.repeat(3).batch(7)
for item in dataset:
  print(item)