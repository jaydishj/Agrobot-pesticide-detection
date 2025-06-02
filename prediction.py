import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# Load your trained CNN model (load once)
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('e:/PESTICIDE PROJECT/pesticide_cnn_model.keras')

model = load_model()

class_labels = ['insecticide', 'fungicide', 'herbicide', 'bactericide', 'Rodenticide', 'Nematicide', 'Miticide']

pesticide_info = {
    "insecticide": {
        "description": "Used to kill or repel insects that damage crops.",
        "mode_of_action": "Targets insect nervous system or development.",
        "examples": "Malathion, Imidacloprid, Pyrethroids",
        "application": "Applied to leaves, soil, or as systemic treatments."
    },
    "fungicide": {
        "description": "Used to prevent or eliminate fungal diseases in plants.",
        "mode_of_action": "Disrupts fungal cell walls or reproduction.",
        "examples": "Mancozeb, Copper sulfate, Azoxystrobin",
        "application": "Sprayed on foliage or seed-treated."
    },
    "herbicide": {
        "description": "Used to control or kill unwanted weeds and plants.",
        "mode_of_action": "Inhibits photosynthesis or amino acid production.",
        "examples": "Glyphosate, Atrazine, Paraquat",
        "application": "Sprayed before or after crop emergence."
    },
    "bactericide": {
        "description": "Used to control bacterial infections in plants.",
        "mode_of_action": "Disrupts bacterial cell membranes or replication.",
        "examples": "Copper compounds, Streptomycin",
        "application": "Foliar sprays or seed treatments."
    },
    "Rodenticide": {
        "description": "Used to control rodents like rats and mice.",
        "mode_of_action": "Anticoagulants or neurological poisons.",
        "examples": "Brodifacoum, Zinc phosphide",
        "application": "Pellets, baits placed in fields or storage."
    },
    "Nematicide": {
        "description": "Used to eliminate nematodes (microscopic worms) in soil.",
        "mode_of_action": "Neurotoxins or enzyme inhibitors.",
        "examples": "Aldicarb, Fluensulfone",
        "application": "Soil fumigation or drenches."
    },
    "Miticide": {
        "description": "Used to control mites and ticks affecting crops.",
        "mode_of_action": "Inhibits respiration or nervous system.",
        "examples": "Abamectin, Bifenazate",
        "application": "Foliar sprays or greenhouse misting."
    }
}

def predict_pesticide(img):
    img = img.resize((128, 128))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_class = class_labels[predicted_index]
    info = pesticide_info[predicted_class]

    return predicted_class, confidence, info

def welcome_page():
    st.title("🌾 Welcome to Pesticide Classification AI System")
    st.markdown("""
    ### The Impact of Pesticide Misuse on Agriculture
    
    Pesticides play a crucial role in protecting crops from pests and diseases. However, misuse or overuse can harm the environment, human health, and reduce biodiversity.
    
    **Why is pesticide classification important?**
    - Ensures correct pesticide use to maximize crop protection.
    - Minimizes harmful environmental impacts.
    - Supports sustainable agriculture and food safety.
    
    ---
    
    ### Did you know?
    - Worldwide, over 2 million tons of pesticides are used annually.
    - Improper pesticide application causes contamination of soil and water.
    - Accurate identification helps farmers choose the right pesticide and dosage.
    
    
    
    ---
    
    Use the **Classification** page from the sidebar to upload pesticide images and get instant classification with detailed info.
    """)

def classification_page():
    st.title("🧪 Pesticide Classification & Detailed Information")

    uploaded_file = st.file_uploader("Upload an image of pesticide label or spray pattern", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_container_width=True)

        with st.spinner('Analyzing...'):
            predicted_class, confidence, info = predict_pesticide(img)

        st.success(f"🌾 Predicted Pesticide Class: **{predicted_class.upper()}**")
        st.write(f"Confidence Score: **{confidence:.2f}**")

        if confidence < 0.7:
            st.warning("⚠️ Confidence is low. Please verify the prediction manually.")

        st.markdown("### Detailed Information")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Mode of Action:** {info['mode_of_action']}")
        st.markdown(f"**Common Examples:** {info['examples']}")
        st.markdown(f"**Typical Application:** {info['application']}")

        # Thank you message with developer name and logo
        st.markdown("---")
        st.markdown("### 🙏 Thank You for Using This Application!")
        st.markdown("Developed by **Jaydish Kennedy**")

        # Display your logo (update the path accordingly)
        try:
            logo_path = 'c:/Users/Admin/Downloads/ChatGPT Image Apr 4, 2025, 03_01_29 PM (1).png'  # Change this to your actual logo file path
            logo_img = Image.open(logo_path)
            st.image(logo_img, width=150)
        except Exception as e:
            st.error("Logo image not found. Please check the path.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Classification"])

if page == "Welcome":
    welcome_page()
elif page == "Classification":
    classification_page()
