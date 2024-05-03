import numpy as np
from tensorflow.keras.datasets import mnist

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Tránh tràn số
    return exp_x / np.sum(exp_x, axis=0)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))

def max_pooling2d(pooling_size, img):
    pool_height, pool_width = pooling_size
    height, width = img.shape[:2]
    if pool_height > height or pool_width > width:
        raise ValueError("Pooling size cannot be larger than image size")
    output_height = height // pool_height
    output_width = width // pool_width
    result = np.zeros((output_height, output_width))
    for row in range(output_height):
        for col in range(output_width):
            result[row, col] = np.max(img[row * pool_height: (row + 1) * pool_height,
                                     col * pool_width: (col + 1) * pool_width])
    return result

def conv2d(kernel, img, s, p, active_func):
    height, width = img.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]
    output_height = (height + 2 * p - kernel_height) // s + 1
    output_width = (width + 2 * p - kernel_width) // s + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Invalid output dimensions. Adjust padding or kernel size.")
    matrix = np.zeros((output_height, output_width))
    for row in range(output_height):
        for col in range(output_width):
            sub_img = img[row * s: row * s + kernel_height, col * s: col * s + kernel_width]
            matrix[row, col] = np.sum(sub_img * kernel)
            if active_func == "relu":
                matrix[row, col] = relu(matrix[row, col])
    return matrix

def flatten(array):
    return array.reshape(-1)

def dense(x, weights, biases):
    return np.dot(x, weights) + biases

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28))
test_images = test_images.astype('float32') / 255
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

np.random.seed(0)
kernel1 = np.random.randn(3, 3)
bias1 = np.zeros((1,))
kernel2 = np.random.randn(3, 3)
bias2 = np.zeros((1,))
weights_fc = np.random.randn(100, 10)
biases_fc = np.zeros((10,))

flattened_shape = 121
weights_fc = np.random.randn(flattened_shape, 10)

epochs = 10
learning_rate = 0.001
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for i in range(len(train_images)):
        x = train_images[i]
        y_true = train_labels[i]
        
        conv1 = conv2d(kernel1, x, s=1, p=0, active_func="relu")
        pool1 = max_pooling2d((2, 2), conv1)
        conv2 = conv2d(kernel2, pool1, s=1, p=0, active_func="relu")
        flattened = flatten(conv2)
        logits = dense(flattened, weights_fc, biases_fc)
        probs = softmax(logits)
        
        loss = cross_entropy_loss(y_true, probs)
        
        d_logits = probs - y_true
        d_flattened = np.dot(d_logits, weights_fc.T)
        d_conv2 = d_flattened.reshape(conv2.shape)
        d_pool1 = np.zeros(pool1.shape)
        d_conv1 = np.zeros(conv1.shape)

        for row in range(d_pool1.shape[0]):
            for col in range(d_pool1.shape[1]):
                d_pool1[row, col] = np.max(d_conv2[row * 2: row * 2 + 2, col * 2: col * 2 + 2])

        for row in range(d_conv1.shape[0]):
            for col in range(d_conv1.shape[1]):
                d_conv1[row, col] = np.sum(d_pool1[row // 2: row // 2 + 1, col // 2: col // 2 + 1])

        d_weights_fc = np.dot(flattened, d_logits.T)
        d_biases_fc = np.sum(d_logits, axis=0)
        weights_fc -= learning_rate * d_weights_fc
        biases_fc -= learning_rate * d_biases_fc

        d_kernel2 = conv2d(pool1, d_conv2, s=1, p=0, active_func="relu")
        d_bias2 = np.sum(d_conv2)
        kernel2 -= learning_rate * d_kernel2
        bias2 -= learning_rate * d_bias2

        d_kernel1 = conv2d(x, d_conv1, s=1, p=0, active_func="relu")
        d_bias1 = np.sum(d_conv1)
        kernel1 -= learning_rate * d_kernel1
        bias1 -= learning_rate * d_bias1
