import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from utils import apply_distortion, get_gemini_response
import traceback
import pandas as pd
from io import StringIO

# Set page configuration
st.set_page_config(page_title="Multimodal LLM Road Safety Platform", layout="wide")

# Sidebar for settings
st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model:",
    ["gemini-1.5-flash-latest", "gemini-1.5-pro"]
)

st.title("Multimodal LLM Road Safety Platform")

api_key = st.text_input("Enter your Gemini API key:", type="password")

if api_key:
    os.environ['GEMINI_API_KEY'] = api_key
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])

    # Add a new option in the sidebar for analysis mode
    analysis_mode = st.sidebar.radio("Analysis Mode", ["Single", "Bulk"])

    if analysis_mode == "Single":
        st.sidebar.subheader("Distortions")

        distortion_type = st.sidebar.selectbox(
            "Choose Distortion:",
            ["None", "Blur", "Brightness", "Contrast", "Sharpness", "Color", "Rain", "Overlay", "Warp"]
        )

        overlay_image = None
        warp_params = {}

        if distortion_type != "None":
            intensity = st.sidebar.slider("Distortion Intensity", 0.0, 1.0, 0.5)
            if distortion_type == "Overlay":
                overlay_image = st.sidebar.file_uploader("Upload overlay image", type=["png", "jpg", "jpeg"])
            elif distortion_type == "Warp":
                warp_params['wave_amplitude'] = st.sidebar.slider("Wave Amplitude", 0.0, 50.0, 20.0)
                warp_params['wave_frequency'] = st.sidebar.slider("Wave Frequency", 0.0, 0.1, 0.04)
                warp_params['bulge_factor'] = st.sidebar.slider("Bulge Factor", -50.0, 50.0, 30.0)
        else:
            intensity = 1.0
            overlay_image = None

        input_text = st.text_input("Input Prompt:", key="input")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        image = None
        processed_image = None
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                
                if distortion_type != "None":
                    processed_image = apply_distortion(image, distortion_type, intensity, overlay_image, warp_params)
                    if processed_image is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Image", use_column_width=True)
                        with col2:
                            st.image(processed_image, caption="Processed Image", use_column_width=True)
                    else:
                        st.error("Failed to process the image. The distortion function returned None.")
                else:
                    st.image(image, caption="Original Image", use_column_width=True)
                    processed_image = image  # If no distortion, use the original image
            except Exception as e:
                st.error(f"An error occurred while processing the image: {str(e)}")
                st.error(traceback.format_exc())

        submit = st.button("Analyse")

        if submit:
            if input_text or processed_image:
                try:
                    response = get_gemini_response(input_text, processed_image, model_choice)
                    
                    st.subheader("User Input")
                    st.write(input_text if input_text else "[No text input]")
                    
                    st.subheader("AI Response")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                
                st.warning("Please clear or change the input if you wish to analyze a different image or prompt.")
            else:
                st.warning("Please provide either an input prompt, an image, or both.")

    else:  # Bulk Analysis
        st.subheader("Bulk Analysis")
        
        analysis_source = st.radio("Choose analysis source:", ["Upload Files", "Specify Folder Path"])
        
        if analysis_source == "Upload Files":
            # File uploader for multiple images
            uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        else:
            # Folder path input with instructions
            st.write("To specify a folder path:")
            st.write("1. Open a File Explorer or Finder window on your computer.")
            st.write("2. Navigate to the folder containing your images.")
            st.write("3. Copy the full path of that folder.")
            st.write("4. Paste the path into the text box below.")
            
            folder_path = st.text_input("Enter folder path containing images:")
            
            if folder_path:
                if os.path.isdir(folder_path):
                    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    uploaded_files = [os.path.join(folder_path, f) for f in image_files]
                    st.success(f"Found {len(uploaded_files)} images in the specified folder.")
                    
                    # Display a sample of found images
                    if uploaded_files:
                        st.write("Sample of found images:")
                        sample_size = min(5, len(uploaded_files))
                        sample_images = uploaded_files[:sample_size]
                        cols = st.columns(sample_size)
                        for i, img_path in enumerate(sample_images):
                            with cols[i]:
                                st.image(Image.open(img_path), caption=os.path.basename(img_path), use_column_width=True)
                else:
                    st.error("Invalid folder path. Please check and try again.")
                    uploaded_files = []
            else:
                uploaded_files = []

        # Clear all image settings if the number of files changes
        if 'previous_file_count' not in st.session_state:
            st.session_state.previous_file_count = 0
        
        if len(uploaded_files) != st.session_state.previous_file_count:
            st.session_state.image_settings = []
            st.session_state.previous_file_count = len(uploaded_files)
        
        # Create or update image settings
        if 'image_settings' not in st.session_state:
            st.session_state.image_settings = []
        
        if uploaded_files:
            # Ensure image_settings has the same length as uploaded_files
            while len(st.session_state.image_settings) < len(uploaded_files):
                st.session_state.image_settings.append({
                    "distortion": "None",
                    "intensity": 0.5,
                    "input_text": ""
                })
            
            # Remove extra settings if files were removed
            st.session_state.image_settings = st.session_state.image_settings[:len(uploaded_files)]
            
            for i, (file, settings) in enumerate(zip(uploaded_files, st.session_state.image_settings)):
                file_name = file.name if hasattr(file, 'name') else os.path.basename(file)
                with st.expander(f"Settings for {file_name}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Load and process image
                        image = Image.open(file) if isinstance(file, str) else Image.open(file)
                        processed_image = apply_distortion(
                            image, 
                            settings["distortion"],
                            settings["intensity"]
                        )
                        st.image(processed_image, caption=f"Preview: {file_name}", use_column_width=True)
                    
                    with col2:
                        # Distortion selection
                        settings["distortion"] = st.selectbox(
                            "Distortion",
                            ["None", "Blur", "Brightness", "Contrast", "Sharpness", "Color", "Rain", "Warp"],
                            key=f"distortion_{i}",
                            index=["None", "Blur", "Brightness", "Contrast", "Sharpness", "Color", "Rain", "Warp"].index(settings["distortion"])
                        )
                        
                        # Intensity slider
                        settings["intensity"] = st.slider(
                            "Intensity", 
                            0.0, 1.0, 
                            settings["intensity"],
                            key=f"intensity_{i}"
                        )
                        
                        # Input text
                        settings["input_text"] = st.text_input(
                            "Input text", 
                            value=settings["input_text"],
                            key=f"input_{i}"
                        )
        
        # Button to start bulk analysis
        if st.button("Run Bulk Analysis") and uploaded_files:
            results = []
            progress_bar = st.progress(0)
            
            for i, (file, settings) in enumerate(zip(uploaded_files, st.session_state.image_settings)):
                try:
                    image = Image.open(file) if isinstance(file, str) else Image.open(file)
                    file_name = file.name if hasattr(file, 'name') else os.path.basename(file)
                    
                    # Apply distortion
                    processed_image = apply_distortion(image, settings["distortion"], settings["intensity"])
                    
                    # Get AI response
                    response = get_gemini_response(settings["input_text"], processed_image, model_choice)
                    
                    # Add result to list
                    results.append({
                        "Image": file_name,
                        "Distortion": settings["distortion"],
                        "Intensity": settings["intensity"],
                        "Input Text": settings["input_text"],
                        "AI Response": response
                    })
                    
                    # Show AI response
                    st.write(f"AI Response for {file_name}:")
                    st.write(response)
                    st.markdown("---")  # Add a separator between images
                    
                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if results:
                results_df = pd.DataFrame(results)
                st.subheader("Analysis Results")
                st.dataframe(results_df)
                
                # Convert DataFrame to CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="bulk_analysis_results.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No results were generated. Please check your inputs and try again.")
        elif not uploaded_files:
            st.warning("Please upload at least one image or specify a valid folder path to proceed with bulk analysis.")

else:
    st.warning("Please enter your API key to proceed.")