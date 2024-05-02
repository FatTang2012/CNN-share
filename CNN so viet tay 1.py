import numpy as np
from tensorflow.keras.datasets import mnist

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Tránh tràn số
    return exp_x / np.sum(exp_x, axis=0)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))

def max_pooling2d(pooling_size, kernel_size, img, p, s, active_func):
    matrix = conv2d(kernel_size, img, s, p, active_func)
    result = np.zeros((int(matrix.shape[0] / pooling_size[0]), int(matrix.shape[1] / pooling_size[1])))
    for row in range(int(matrix.shape[0] / pooling_size[0])):
        for col in range(int(matrix.shape[1] / pooling_size[1])):
            result[row, col] = np.max(matrix[row * pooling_size[0]: (row + 1) * pooling_size[0],
                                     col * pooling_size[1]: (col + 1) * pooling_size[1]])
    return result

def conv2d(kernel, img, s, p, active_func):
    print("Image shape:", img.shape)  # Thêm dòng này để kiểm tra kích thước thực sự của ảnh
    height, width = img.shape[:2]
    kernel_height, kernel_width = kernel.shape
    matrix = np.zeros((int((height - kernel_height + 2 * p) / s + 1), int((width - kernel_width + 2 * p) / s + 1)))
    for row in range(int((height - kernel_height + 2 * p) / s + 1)):
        for col in range(int((width - kernel_width + 2 * p) / s + 1)):
            sub_img = img[row * s: row * s + kernel_height, col * s: col * s + kernel_width]
            if sub_img.shape != kernel.shape:
                continue
            matrix[row, col] = np.sum(sub_img * kernel)
            if active_func == "relu":
                matrix[row, col] = relu(matrix[row, col])
    return matrix

def flatten(array):
    return array.flatten()

def dense(x, weights, biases):
    return np.dot(x, weights) + biases

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

np.random.seed(0)
kernel1 = np.random.randn(3, 3)
bias1 = np.zeros((1,))
kernel2 = np.random.randn(3, 3)
bias2 = np.zeros((1,))
kernel3 = np.random.randn(3, 3)
bias3 = np.zeros((1,))
weights_fc = np.random.randn(64, 10)
biases_fc = np.zeros((10,))

epochs = 10
learning_rate = 0.001
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for i in range(len(train_images)):
        x = train_images[i]
        y_true = train_labels[i]
        # Lan truyền xuôi
        conv1 = conv2d(kernel1, x, s=1, p=0, active_func="relu")
        pool1 = max_pooling2d((2, 2), (2, 2), conv1, p=0, s=1, active_func="relu")
        conv2 = conv2d(kernel2, pool1, s=1, p=0, active_func="relu")
        pool2 = max_pooling2d((2, 2), (2, 2), conv2, p=0, s=1, active_func="relu")
        conv3 = conv2d(kernel3, pool2, s=1, p=0, active_func="relu")
        flattened = flatten(conv3)
        logits = dense(flattened, weights_fc, biases_fc)
        probs = softmax(logits)
        
        # Tính độ lỗi
        loss = cross_entropy_loss(y_true, probs)
        
        # Lan truyền ngược
        d_logits = probs - y_true
        d_flattened = np.dot(d_logits, weights_fc.T)
        d_conv3 = d_flattened.reshape(conv3.shape)
        d_pool2 = np.zeros(pool2.shape)
        d_conv2 = np.zeros(conv2.shape)
        d_pool1 = np.zeros(pool1.shape)
        d_conv1 = np.zeros(conv1.shape)

        # Cập nhật trọng số của fully connected layer
        d_weights_fc = np.dot(flattened.T, d_logits)
        d_biases_fc = np.sum(d_logits, axis=0)
        weights_fc -= learning_rate * d_weights_fc
        biases_fc -= learning_rate * d_biases_fc

        # Cập nhật trọng số và bias của convolutional layer 3
        d_kernel3 = conv2d(flatten(pool2).T, d_conv3, s=1, p=0, active_func="relu")
        d_bias3 = np.sum(d_conv3)
        kernel3 -= learning_rate * d_kernel3
        bias3 -= learning_rate * d_bias3

        # Cập nhật trọng số và bias của convolutional layer 2
        d_kernel2 = conv2d(flatten(pool1).T, d_conv2, s=1, p=0, active_func="relu")
        d_bias2 = np.sum(d_conv2)
        kernel2 -= learning_rate * d_kernel2
        bias2 -= learning_rate * d_bias2

        # Cập nhật trọng số và bias của convolutional layer 1
        d_kernel1 = conv2d(flatten(x).T, d_conv1, s=1, p=0, active_func="relu")
        d_bias1 = np.sum(d_conv1)
        kernel1 -= learning_rate * d_kernel1
        bias1 -= learning_rate * d_bias1
