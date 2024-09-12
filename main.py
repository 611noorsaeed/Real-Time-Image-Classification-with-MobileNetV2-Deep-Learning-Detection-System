import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNetV2 model
try:
    model = MobileNetV2(weights='imagenet')
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Function to upload and process an image
def upload_image():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if not file_path:
        return  # If no file is selected, return

    # Open the selected image file
    img = Image.open(file_path)
    img = img.resize((224, 224))  # Resize image to 224x224 for MobileNetV2
    img_tk = ImageTk.PhotoImage(img)  # Convert image for Tkinter

    # Update the image label with the uploaded image
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Preprocess the image for MobileNetV2
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    # Make prediction using MobileNetV2
    preds = model.predict(img_array)

    # Decode predictions and display the top 3 results
    top_preds = decode_predictions(preds, top=3)[0]
    result_text = f"Predicted Category:\n{top_preds[0][1]} ({top_preds[0][2] * 100:.2f}%)"
    top_preds_text = "\n".join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in top_preds])

    result_label.config(text=result_text)
    top_preds_label.config(text=top_preds_text)

# Create the Tkinter window============================================================================ui design========
root = tk.Tk()
root.title("Detection System MobileNetV2")
# Set the window size
root.geometry("590x700+100+20")

# Create and place the title label
title_label = tk.Label(root, text="Detection System MobileNetV2", font=("Arial", 24, "bold"))
title_label.pack(pady=20)

# Create widgets
upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 16))
upload_button.pack(pady=20)

img_label = tk.Label(root)
img_label.pack(pady=20)

result_label = tk.Label(root, text="Predicted Category:", font=("Arial", 16))
result_label.pack(pady=20)

top_preds_label = tk.Label(root, text="", font=("Arial", 12))
top_preds_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
