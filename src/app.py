import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from utils import apply_distortion
import traceback

# Set page configuration: sets the title of the page and adjusts the layout to wide
st.set_page_config(page_title="Multimodal LLM Road Safety Platform", layout="wide")

# Display the title of the app
st.title("Multimodal LLM Road Safety Platform")

# Input field for the user to enter their Gemini API key (masked as a password)
api_key = st.text_input("Enter your Gemini API key:", type="password")

# Check if the API key is provided
if api_key:
    # Set the environment variable with the provided API key
    os.environ['GEMINI_API_KEY'] = api_key
    # Configure the Google Generative AI client with the API key
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    # Display success message if API key is set correctly
    st.success("API key set successfully!")

    # Sidebar for additional settings
    st.sidebar.title("Settings")

    # Dropdown menu in the sidebar to select the AI model
    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["gemini-1.5-flash-latest", "gemini-1.5-pro"]
    )

    # Add a new option in the sidebar to choose analysis mode: Single or Bulk
    analysis_mode = st.sidebar.radio("Analysis Mode", ["Single", "Bulk"])

    # If 'Single' mode is selected, display the corresponding message
    if analysis_mode == "Single":
        st.sidebar.subheader("Distortions")

        distortion_type = st.sidebar.selectbox(
            "Choose Distortion:",
            ["None", "Blur", "Brightness", "Contrast", "Sharpness", "Color", "Rain", "Overlay", "Warp"]
        )

        # Initialize
        overlay_image = None
        warp_params = {}

        # Check if the user selected a distortion type other than 'None'
        if distortion_type != "None":
            intensity = st.sidebar.slider("Distortion Intensity", 0.0, 1.0, 0.5)
            if distortion_type == "Overlay":
                overlay_image = st.sidebar.file_uploader("Upload overlay image", type=["png", "jpg", "jpeg"])
            elif distortion_type == "Warp":
                warp_params['wave_amplitude'] = st.sidebar.slider("Wave Amplitude", 0.0, 50.0, 20.0)
                warp_params['wave_frequency'] = st.sidebar.slider("Wave Frequency", 0.0, 0.1, 0.04)
                warp_params['bulge_factor'] = st.sidebar.slider("Bulge Factor", -50.0, 50.0, 30.0)
        else:
            # Set default intensity when no distortion is selected
            intensity = 1.0
            overlay_image = None

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Initialize
        image = None
        processed_image = None
        
        # Check if an image file was uploaded
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                
                # Apply distortion if distortion type is not 'None'
                if distortion_type != "None":
                    processed_image = apply_distortion(image, distortion_type, intensity, overlay_image, warp_params)
                    # Check if the image was successfully processed
                    if processed_image is not None:
                        # Display original and processed images side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Image", use_column_width=True)
                        with col2:
                            st.image(processed_image, caption="Processed Image", use_column_width=True)
                    else:
                        st.error("Failed to process the image. The distortion function returned None.")
                else:
                    # If no distortion is selected, display the original image
                    st.image(image, caption="Original Image", use_column_width=True)
                    processed_image = image  # If no distortion, use the original image
            except Exception as e:
                st.error(f"An error occurred while processing the image: {str(e)}")
                st.error(traceback.format_exc())
    else:  # If not in 'Single' mode, handle Bulk Analysis
        st.write("Bulk analysis mode selected.")
else:
    # Display warning if the API key is not provided
    st.warning("Please enter your API key to proceed.")