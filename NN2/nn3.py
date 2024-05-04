import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Tanh activation function
def tanh(x):
    return np.tanh(x)

# Derivative of tanh function
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# One-hot encoding
def one_hot_encoding(labels, num_classes):
    num_samples = labels.shape[0]
    one_hot_labels = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

# Neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
    def forward(self, x):
        # Forward pass
        self.hidden_input = np.dot(x, self.weights_input_hidden)
        self.hidden_output = tanh(self.hidden_input)
        self.output = tanh(np.dot(self.hidden_output, self.weights_hidden_output))
        return self.output
    
    def train(self, x, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(x)
            
            # Calculate loss
            loss = np.mean(np.square(y - output))
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')
            
            # Backpropagation
            error = y - output
            d_output = error * tanh_derivative(output)
            d_hidden = np.dot(d_output, self.weights_hidden_output.T) * tanh_derivative(self.hidden_output)
            
            # Update weights
            self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, d_output)
            self.weights_input_hidden += learning_rate * np.dot(x.T, d_hidden)

# Load MNIST data
(train_data, train_labels), (_, _) = mnist.load_data()

# Flatten images
train_data = train_data.reshape(train_data.shape[0], -1)

# Preprocess data
train_data = train_data / 255.0  # Normalize pixel values
train_labels = one_hot_encoding(train_labels, 10)  # One-hot encode labels

# Initialize neural network
input_size = train_data.shape[1]
hidden_size = 128
output_size = 10
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train neural network
learning_rate = 0.1
epochs = 10
nn.train(train_data, train_labels, learning_rate, epochs)

print("Neural network training completed.")

# Define the model architecture
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='tanh'),
    Dense(10, activation='tanh')
])

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data.reshape(-1, 28, 28), train_labels, epochs=10)

# Save the model to a .h5 file
model.save('nn2_model.h5')

