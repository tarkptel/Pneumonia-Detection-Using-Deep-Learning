import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

# Load the trained model
model = tf.keras.models.load_model("pneumonia_detection.h5")

# Define the prediction function
def predict_xray(image):
    # Convert PIL image to OpenCV format (numpy array)
    image = np.array(image)
    
    # Resize image to 150x150 (as per your training)
    image = cv2.resize(image, (150, 150))
    
    # Reshape and normalize
    image = image.reshape(1, 150, 150, 3) / 255.0  # Normalization (if used in training)
    
    # Make prediction
    prediction = model.predict(image)[0]  # Get probabilities for both classes
    
    # Class labels
    labels = ["The Patient is Normal.", "The Patient has Pneumonia."]
    
    # Get predicted class and confidence scores
    predicted_class = np.argmax(prediction)  # Class with highest probability
    confidence = prediction[predicted_class] * 100  # Convert to percentage
    
    return f"{labels[predicted_class]} ({confidence:.2f}% confidence)"

# Create Gradio UI
iface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil"),  # Accepts image input
    outputs="text",  # Returns class label with confidence
    title="Pneumonia Detection",
    description="Upload a chest X-ray image, and the model will predict if the patient has pneumonia or is normal, along with confidence scores."
)

# Launch the app
iface.launch()
