import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from utils import apply_distortion, get_gemini_response
import traceback
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Multimodal LLM Road Safety Platform", layout="wide")

# Define your CSS
css = """
<style>
    .stTextArea textarea {
        font-size: 1rem;
        padding-top: 0;
        margin-top: 0;
    }
    .stTextArea div[data-baseweb="textarea"] {
        margin-top: 0;
    }
</style>
"""

# Apply the CSS
st.markdown(css, unsafe_allow_html=True)

# Initialize session state variables
if 'use_system_instructions' not in st.session_state:
    st.session_state.use_system_instructions = False

if 'show_more_image_opts' not in st.session_state:
    st.session_state.show_more_image_opts = False

if 'system_instructions' not in st.session_state:
    st.session_state.system_instructions = """
    You are an AI assistant specialized in analyzing road safety images. Your task is to:
    1. Describe the scene(s) objectively, noting visible road features, signage, and potential hazards.
    2. Identify potential safety issues or concerns based on what you can see in the image(s).
    3. Suggest improvements or preventive measures for any identified issues.
    4. Comment on the overall safety of the scene(s) depicted.
    5. If multiple images are provided, note any significant differences or patterns, but do not assume they are necessarily sequential or related unless explicitly stated.
    6. Analyze each image individually, whether it's a single frame or part of a set.
    7. If any distortions or unusual visual effects are present, mention them only if they are clearly visible and relevant to safety analysis.
    Please provide your analysis in a clear, concise manner, focusing on road safety aspects. Adapt your response to the number and nature of the images provided.
    """

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "gemini-1.5-flash-latest"

# Add this near the top of your file, after the imports
PREDEFINED_PROMPTS = [
    "Analyze the road safety features visible in this image.",
    "Identify potential hazards for pedestrians in this scene.",
    "Evaluate the effectiveness of traffic signs and signals in this image.",
    "Assess the road conditions and suggest improvements for safety.",
    "Examine the intersection design and comment on its safety aspects.",
    "Identify any issues with road markings or lane divisions.",
    "Analyze the safety considerations for cyclists in this environment.",
    "Evaluate the lighting conditions and their impact on road safety.",
    "Assess the visibility and placement of traffic lights in this scene.",
    "Identify any potential blind spots or visual obstructions for drivers.",
]

# Rest of your app code starts here
st.title("Multimodal LLM Road Safety Platform")

# Sidebar
st.sidebar.title("Settings")

st.session_state.model_choice = st.sidebar.selectbox(
    "Choose Model:",
    ["gemini-1.5-flash-latest", "gemini-1.5-pro"],
    index=["gemini-1.5-flash-latest", "gemini-1.5-pro"].index(st.session_state.model_choice)
)

st.sidebar.subheader("System Instructions")

# Add the toggle button
st.session_state.use_system_instructions = st.sidebar.toggle(
    "Check System Instructions", value=st.session_state.use_system_instructions
)

# Only show the text area if system instructions are enabled
if st.session_state.use_system_instructions:
    st.session_state.system_instructions = st.sidebar.text_area(
        "Customize AI Instructions",
        st.session_state.system_instructions,
        height=200
    )
else:
    st.sidebar.info("System instructions are disabled.")

st.session_state.api_key = st.text_input("Enter your Gemini API key:", type="password", value=st.session_state.api_key)

if st.session_state.api_key:
    os.environ['GEMINI_API_KEY'] = st.session_state.api_key
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])

    # Add a new option in the sidebar for analysis mode
    analysis_mode = st.sidebar.radio("Analysis Mode", ["Single", "Bulk"])

    if analysis_mode == "Single":
        st.sidebar.subheader("Distortions")

        overlay_image = None
        warp_params = {}
        blur_intensity = 0.0
        brightness_intensity = 0.0
        contrast_intensity = 0.0
        sharpness_intensity = 0.0
        saturation_intensity = 1.0
        hue_shift = 1.0

        rain_intensity = st.sidebar.slider(
            "Rain Intensity", 0.0, 1.0, 0.0
        )

        overlay_intensity = st.sidebar.slider(
            "Overlay Intensity", 0.0, 1.0, 1.0
        )

        overlay_image = st.sidebar.file_uploader(
            "Upload overlay image", 
            type=["png", "jpg", "jpeg"]
        )

        warp_params['wave_amplitude'] = st.sidebar.slider(
            "Wave Amplitude", 0.0, 50.0, 0.0
        )
        warp_params['wave_frequency'] = st.sidebar.slider(
            "Wave Frequency", 0.0, 0.1, 0.0
        )
        warp_params['bulge_factor'] = st.sidebar.slider(
            "Bulge Factor", -50.0, 50.0, 1.0
        )

        # Add show more image opts
        st.session_state.show_more_image_opts = st.sidebar.toggle(
            "More image adjustment options",
            value=False
        )

        if st.session_state.show_more_image_opts:
            blur_intensity = st.sidebar.slider(
                "Blur Intensity", 0.0, 1.0, 0.0
            )
            brightness_intensity = st.sidebar.slider(
                "Brightness Intensity", 0.0, 1.0, 0.0
            )
            contrast_intensity = st.sidebar.slider(
                "Contrast Intensity", 0.0, 1.0, 0.0
            )
            sharpness_intensity = st.sidebar.slider(
                "Sharpness Intensity", 0.0, 1.0, 0.0
            )
            saturation_intensity = st.sidebar.slider(
                "Saturation Intensity", 0.0, 2.0, 1.0
            )
            hue_shift = st.sidebar.slider(
                "Hue Intensity", 0.0, 2.0, 1.0
            )

        prompt_option = st.radio("Choose prompt type:", ["Predefined", "Custom"])
        if prompt_option == "Predefined":
            input_text = st.selectbox("Select a predefined prompt:", PREDEFINED_PROMPTS)
        else:
            input_text = st.text_input("Input Custom Prompt:", key="input")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        image = None
        processed_image = None

        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                processed_image = apply_distortion(
                    image,
                    blur_intensity=blur_intensity,
                    brightness_intensity=brightness_intensity,
                    contrast_intensity=contrast_intensity,
                    sharpness_intensity=sharpness_intensity,
                    saturation_intensity=saturation_intensity,
                    hue_shift=hue_shift,
                    rain_intensity=rain_intensity,
                    overlay_image=overlay_image,
                    overlay_intensity=overlay_intensity,
                    warp_params=warp_params,
                )
                
                if processed_image is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_column_width=True)
                    with col2:
                        caption = "Processed Image"
                        st.image(processed_image, caption=caption, use_column_width=True)
                else:
                    st.error("Failed to process the image. The distortion function returned None.")
            except Exception as e:
                st.error(f"An error occurred while processing the image: {str(e)}")
                st.error(traceback.format_exc())

        submit = st.button("Analyse")




        if submit:
            if input_text or processed_image:
                try:
                    response = get_gemini_response(
                        input_text, 
                        processed_image, 
                        st.session_state.model_choice, 
                        st.session_state.system_instructions if st.session_state.use_system_instructions else None
                    )
                    
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
                    "input_text": "",
                    "overlay_image": None,
                    "warp_params": {
                        "wave_amplitude": 20.0,
                        "wave_frequency": 0.04,
                        "bulge_factor": 30.0
                    },
                    "saturation": 1.0,
                    "hue_shift": 0.0
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
                            settings["intensity"] if settings["distortion"] != "Color" else None,
                            settings["overlay_image"],
                            settings["warp_params"],
                            settings["saturation"] if settings["distortion"] == "Color" else None,
                            settings["hue_shift"] if settings["distortion"] == "Color" else None
                        )
                        st.image(processed_image, caption=f"Preview: {file_name}", use_column_width=True)
                    
                    with col2:
                        # Distortion selection
                        settings["distortion"] = st.selectbox(
                            "Distortion",
                            ["None", "Blur", "Brightness", "Contrast", "Sharpness", "Color", "Rain", "Overlay", "Warp"],
                            key=f"distortion_{i}",
                            index=["None", "Blur", "Brightness", "Contrast", "Sharpness", "Color", "Rain", "Overlay", "Warp"].index(settings["distortion"])
                        )
                        
                        if settings["distortion"] == "Color":
                            settings["saturation"] = st.slider("Color Saturation", 0.0, 2.0, settings["saturation"], key=f"saturation_{i}")
                            settings["hue_shift"] = st.slider("Hue Shift", -0.5, 0.5, settings["hue_shift"], key=f"hue_shift_{i}")
                        elif settings["distortion"] != "None":
                            settings["intensity"] = st.slider("Intensity", 0.0, 1.0, settings["intensity"], key=f"intensity_{i}")
                        
                        # Overlay image uploader
                        if settings["distortion"] == "Overlay":
                            settings["overlay_image"] = st.file_uploader("Upload overlay image", type=["png", "jpg", "jpeg"], key=f"overlay_{i}")
                        
                        # Warp parameters
                        if settings["distortion"] == "Warp":
                            settings["warp_params"]["wave_amplitude"] = st.slider("Wave Amplitude", 0.0, 50.0, settings["warp_params"]["wave_amplitude"], key=f"wave_amplitude_{i}")
                            settings["warp_params"]["wave_frequency"] = st.slider("Wave Frequency", 0.0, 0.1, settings["warp_params"]["wave_frequency"], key=f"wave_frequency_{i}")
                            settings["warp_params"]["bulge_factor"] = st.slider("Bulge Factor", -50.0, 50.0, settings["warp_params"]["bulge_factor"], key=f"bulge_factor_{i}")
                        
                        # Input text
                        prompt_option = st.radio("Choose prompt type:", ["Predefined", "Custom"], key=f"prompt_option_{i}")
                        if prompt_option == "Predefined":
                            settings["input_text"] = st.selectbox(
                                "Select a predefined prompt:", 
                                PREDEFINED_PROMPTS,
                                key=f"predefined_prompt_{i}"
                            )
                        else:
                            settings["input_text"] = st.text_input(
                                "Input custom text", 
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
                    processed_image = apply_distortion(
                        image, 
                        settings["distortion"], 
                        settings["intensity"] if settings["distortion"] != "Color" else None,
                        settings["overlay_image"], 
                        settings["warp_params"],
                        settings["saturation"] if settings["distortion"] == "Color" else None,
                        settings["hue_shift"] if settings["distortion"] == "Color" else None
                    )
                    
                    # Get AI response
                    response = get_gemini_response(
                        settings["input_text"], 
                        processed_image, 
                        st.session_state.model_choice, 
                        st.session_state.system_instructions if st.session_state.use_system_instructions else None
                    )
                    
                    # Add result to list
                    results.append({
                        "Image": file_name,
                        "Distortion": settings["distortion"],
                        "Intensity": settings["intensity"] if settings["distortion"] != "Color" else None,
                        "Saturation": settings["saturation"] if settings["distortion"] == "Color" else None,
                        "Hue Shift": settings["hue_shift"] if settings["distortion"] == "Color" else None,
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