import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

# Function to predict digit
def predict_digit():
    # Load the trained model
    model = tf.keras.models.load_model('mnist_cnn_model.h5')

    # Load and preprocess the input image
    image = cv2.imread('D:\\Document move here\\Learning\\Hoc ki\\N2\\HK2 N2\\mang than kinh\\full set number\\ve.png', cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (28, 28))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)

    # Predict the digit
    prediction = model.predict(input_image)
    predicted_digit = np.argmax(prediction)

    # Create a popup window
    popup = tk.Tk()
    popup.wm_title("Predicted Digit")

    # Display predicted digit
    label = tk.Label(popup, text="Predicted Digit: " + str(predicted_digit))
    label.pack(side="top", fill="x", pady=10)

    # Close popup window
    button = tk.Button(popup, text="Close", command=popup.destroy)
    button.pack()

    popup.mainloop()

# Call the predict_digit function
predict_digit()
