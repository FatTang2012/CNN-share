import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Loss function with L2 regularization
def cross_entropy_loss(y_true, y_pred, weights, lambda_reg):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    # L2 regularization
    l2_regularization = 0.5 * lambda_reg * np.sum(weights ** 2)
    loss += l2_regularization / m
    return loss

# Gradient of cross-entropy loss function with L2 regularization
def grad_cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    grad = y_pred.copy()
    grad[range(m), y_true] -= 1
    grad /= m
    return grad

# Model architecture using Keras
def keras_model(input_size, hidden_size, output_size):
    model = Sequential([
        Dense(hidden_size, activation='relu', input_shape=(input_size,)),
        Dense(output_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Model architecture without Keras
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def backward(self, X, y, learning_rate, lambda_reg):
        m = X.shape[0]
        
        dZ2 = grad_cross_entropy_loss(y, self.A2)
        dW2 = np.dot(self.A1.T, dZ2) + lambda_reg * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = np.dot(X.T, dZ1) + lambda_reg * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# One-hot encode labels
def one_hot_encode(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

num_classes = 10
y_train_one_hot = one_hot_encode(y_train, num_classes)

# Train the Keras model
input_size = X_train.shape[1]
hidden_size = 256
output_size = num_classes
learning_rate = 0.1
lambda_reg = 0.01  # L2 regularization parameter
num_epochs = 20

keras_model = keras_model(input_size, hidden_size, output_size)
keras_model.fit(X_train, y_train, epochs=num_epochs)
keras_model.save('nn_model.h5')

# Train the model without Keras
model = NeuralNetwork(input_size, hidden_size, output_size)
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model.forward(X_train)
    
    # Compute and print loss
    loss = cross_entropy_loss(y_train.flatten(), y_pred, model.W1, lambda_reg)  # L2 regularization
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss}')
    
    # Backpropagation with L2 regularization
    model.backward(X_train, y_train.flatten(), learning_rate, lambda_reg)

# Evaluate the Keras model
keras_model = load_model('nn_model.h5')
keras_model.evaluate(X_test, y_test)

# Evaluate the model without Keras
y_pred_test = model.forward(X_test)
predicted_labels = np.argmax(y_pred_test, axis=1)
accuracy = np.mean(predicted_labels == y_test.flatten())
print(f'Test Accuracy (Without Keras): {accuracy}')
