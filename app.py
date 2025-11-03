import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("malaria_parasite.keras")

def predict_malaria(image):
    """Predicts if a given image contains malaria parasites."""
    # Preprocess the image
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)

    # Make a prediction
    prediction = model.predict(image)

    # Convert sigmoid output to class label
    if prediction[0][0] < 0.5:
        label = "Parasitized"
    else:
        label = "Uninfected"

    return label

# Create gradio interface
iface = gr.Interface(fn=predict_malaria, inputs=gr.Image(), outputs=gr.Label())

if __name__ == "__main__":
    iface.launch()