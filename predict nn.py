import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

# Load trained model
model = load_model('nn_model.h5')

# Function to preprocess image for inference
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (28, 28))
    normalized_image = resized_image / 255.0
    flattened_image = normalized_image.flatten()  
    preprocessed_image = flattened_image.reshape(1, -1)  
    return preprocessed_image

# Function to predict digit from image
def predict_digit(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    digit = np.argmax(prediction)
    return digit

# Function to display predicted digit in a popup window
def display_result(image_path, predicted_digit):
    # Create a new Tkinter window
    window = tk.Tk()
    window.title("Digit Recognition Result")

    # Load and display the image
    img = Image.open(image_path)
    img = img.resize((200, 200))  # Không sử dụng giảm nhiễu
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(window, image=img)
    panel.image = img
    panel.pack(side="top", padx=10, pady=10)

    # Display the predicted digit
    label = tk.Label(window, text=f"Predicted digit: {predicted_digit}", font=("Helvetica", 16))
    label.pack(side="bottom", padx=10, pady=10)

    # Run the Tkinter event loop
    window.mainloop()

# Load image from file
image_path = 'D:\\Document move here\\Learning\\Hoc ki\\N2\\HK2 N2\\mang than kinh\\full set number\\ve.png'
image = cv2.imread(image_path)

# Predict digit from image
predicted_digit = predict_digit(image)

# Display the result in a popup window
display_result(image_path, predicted_digit)
