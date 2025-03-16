import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img, load_img
import os

# Load the trained generator model (cache to avoid reloading)
@st.cache_resource
def load_generator():
    return tf.keras.models.load_model("generator.h5")

generator = load_generator()

# Set latent space size
latent_dim = 100

# Streamlit UI
st.title("✨ Anime Face Generator ")
st.write("Press the button below to generate new AI-created anime faces.")

# Resize sample image
sample_image_path = "sample.png"
if os.path.exists(sample_image_path):
    sample_img = load_img(sample_image_path, target_size=(200, 125))  
    st.image(sample_img, caption="Sample Generated Face (Resized)", use_column_width=False)


# Initialize session state for storing generated images
if "generated_images" not in st.session_state:
    st.session_state.generated_images = None

# Button to generate new images
if st.button("Generate New Faces 🚀"):
    # Generate random noise
    random_noise = tf.random.normal([100, latent_dim])

    # Generate new images using the trained model
    generated_images = generator(random_noise, training=False)

    # Denormalize images (convert from [-1,1] to [0,255])
    generated_images = (generated_images * 127.5) + 127.5  
    st.session_state.generated_images = generated_images.numpy().astype("uint8")

# Display the generated images if available
if st.session_state.generated_images is not None:
    fig, axes = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(100):
        ax = axes[i // 10, i % 10]
        img = array_to_img(st.session_state.generated_images[i])
        ax.imshow(img)
        ax.axis("off")

    st.pyplot(fig)
