import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from utils import apply_distortions, get_gemini_response
import traceback
import pandas as pd
from io import StringIO
import io
import json

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
    st.session_state.use_system_instructions = True

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

# Predefined Prompts
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

# Distortion Types
DISTORTION_TYPES = ["None", "Blur", "Brightness", "Contrast", "Sharpness", "Color", "Rain", "Overlay", "Warp"]

# Expected JSON Fields
EXPECTED_JSON_FIELDS = [
    "scene_description",
    "safety_features",
    "potential_hazards",
    "traffic_signs_effectiveness",
    "road_conditions",
    "suggested_improvements",
    "intersection_design",
    "road_markings_issues",
    "cyclist_safety",
    "lighting_conditions",
    "traffic_lights_visibility",
    "blind_spots",
    "overall_safety"
]

# Title
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
st.session_state.use_system_instructions = st.sidebar.checkbox("Use System Instructions", value=st.session_state.use_system_instructions)

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

        selected_distortions = st.sidebar.multiselect(
            "Choose Distortions:",
            DISTORTION_TYPES[1:],  # Exclude "None" from the options
            default=None,
            placeholder="Choose an option"
        )

        distortions = []

        if selected_distortions:
            for distortion_type in selected_distortions:
                with st.sidebar.expander(f"{distortion_type} Settings"):
                    if distortion_type == "Color":
                        saturation = st.slider(f"{distortion_type} Saturation", 0.0, 2.0, 1.0)
                        hue_shift = st.slider(f"{distortion_type} Hue Shift", -0.5, 0.5, 0.0)
                        distortions.append({
                            'type': distortion_type,
                            'saturation': saturation,
                            'hue_shift': hue_shift
                        })
                    elif distortion_type == "Overlay":
                        intensity = st.slider(f"{distortion_type} Intensity", 0.0, 1.0, 0.5)
                        overlay_image = st.file_uploader(f"Upload {distortion_type} image", type=["png", "jpg", "jpeg"], key=f"overlay_single_{distortion_type}")
                        if overlay_image:
                            overlay_img = Image.open(overlay_image).convert("RGBA")
                            overlay_img.thumbnail((300, 300))  # Resize to a manageable size
                            img_byte_arr = io.BytesIO()
                            overlay_img.save(img_byte_arr, format='PNG')
                            overlay_image_bytes = img_byte_arr.getvalue()
                            st.success("Overlay image uploaded successfully.")
                        else:
                            overlay_image_bytes = None
                        distortions.append({
                            'type': distortion_type,
                            'intensity': intensity,
                            'overlay_image': overlay_image_bytes
                        })
                    elif distortion_type == "Warp":
                        intensity = st.slider(f"{distortion_type} Intensity", 0.0, 1.0, 0.5)
                        wave_amplitude = st.slider(f"{distortion_type} Wave Amplitude", 0.0, 50.0, 20.0)
                        wave_frequency = st.slider(f"{distortion_type} Wave Frequency", 0.0, 0.1, 0.04)
                        bulge_factor = st.slider(f"{distortion_type} Bulge Factor", -50.0, 50.0, 30.0)
                        distortions.append({
                            'type': distortion_type,
                            'intensity': intensity,
                            'warp_params': {
                                'wave_amplitude': wave_amplitude,
                                'wave_frequency': wave_frequency,
                                'bulge_factor': bulge_factor
                            }
                        })
                    else:
                        intensity = st.slider(f"{distortion_type} Intensity", 0.0, 1.0, 0.5)
                        distortions.append({
                            'type': distortion_type,
                            'intensity': intensity
                        })

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

                if distortions:
                    processed_image = apply_distortions(image, distortions)
                    if processed_image is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Image", use_column_width=True)
                        with col2:
                            caption = f"Processed Image ({', '.join([d['type'] for d in distortions])})"
                            st.image(processed_image, caption=caption, use_column_width=True)
                    else:
                        st.error("Failed to process the image. The distortion function returned None.")
                else:
                    st.image(image, caption="Original Image", use_column_width=True)
                    processed_image = image  # If no distortions, use the original image
            except Exception as e:
                st.error(f"An error occurred while processing the image: {str(e)}")
                st.error(traceback.format_exc())

        submit = st.button("Analyse")

        if submit:
            if input_text or processed_image:
                try:
                    text_response, json_response = get_gemini_response(
                        input_text,
                        processed_image,
                        st.session_state.model_choice,
                        st.session_state.system_instructions if st.session_state.use_system_instructions else None,
                        EXPECTED_JSON_FIELDS  # Add this line
                    )

                    st.subheader("User Input")
                    st.write(input_text if input_text else "[No text input]")

                    st.subheader("AI Response")
                    st.write(text_response)

                    # Remove the JSON Response display here

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

                st.warning("Please clear or change the input if you wish to analyze a different image or prompt.")
            else:
                st.warning("Please provide either an input prompt, an image, or both.")

    else:  # Bulk Analysis
        st.subheader("Bulk Analysis Settings")

        use_centralized_distortions = st.checkbox("Use centralized distortion settings for all images", value=False)

        if use_centralized_distortions:
            st.subheader("Centralized Distortion Settings")
            centralized_distortions = st.multiselect(
                "Choose Distortions for all images:",
                DISTORTION_TYPES[1:],  # Exclude "None" from the options
                key="centralized_distortions"
            )

            # Store centralized distortion settings in st.session_state
            if 'centralized_distortion_settings' not in st.session_state:
                st.session_state.centralized_distortion_settings = {}
            centralized_distortion_settings = st.session_state.centralized_distortion_settings

            for distortion_type in centralized_distortions:
                with st.expander(f"{distortion_type} Settings"):
                    if distortion_type == "Color":
                        saturation = st.slider(
                            f"{distortion_type} Saturation",
                            0.0,
                            2.0,
                            centralized_distortion_settings.get(distortion_type, {}).get('saturation', 1.0)
                        )
                        hue_shift = st.slider(
                            f"{distortion_type} Hue Shift",
                            -0.5,
                            0.5,
                            centralized_distortion_settings.get(distortion_type, {}).get('hue_shift', 0.0)
                        )
                        centralized_distortion_settings[distortion_type] = {
                            'saturation': saturation,
                            'hue_shift': hue_shift
                        }
                    elif distortion_type == "Overlay":
                        intensity = st.slider(
                            f"{distortion_type} Intensity",
                            0.0,
                            1.0,
                            centralized_distortion_settings.get(distortion_type, {}).get('intensity', 0.5)
                        )

                        overlay_image = st.file_uploader(
                            f"Upload {distortion_type} image",
                            type=["png", "jpg", "jpeg"],
                            key=f"overlay_image_{distortion_type}"
                        )

                        if overlay_image is not None:
                            overlay_img = Image.open(overlay_image).convert("RGBA")
                            overlay_img.thumbnail((300, 300))  # Resize to a manageable size
                            img_byte_arr = io.BytesIO()
                            overlay_img.save(img_byte_arr, format='PNG')
                            centralized_distortion_settings[distortion_type] = {
                                'intensity': intensity,
                                'overlay_image': img_byte_arr.getvalue()
                            }
                            st.success("Overlay image uploaded successfully.")
                        else:
                            # Clear the overlay image if no file is uploaded
                            if distortion_type in centralized_distortion_settings:
                                centralized_distortion_settings[distortion_type]['overlay_image'] = None
                            else:
                                centralized_distortion_settings[distortion_type] = {
                                    'intensity': intensity,
                                    'overlay_image': None
                                }
                            st.info("No overlay image selected.")
                    elif distortion_type == "Warp":
                        intensity = st.slider(
                            f"{distortion_type} Intensity",
                            0.0,
                            1.0,
                            centralized_distortion_settings.get(distortion_type, {}).get('intensity', 0.5)
                        )
                        wave_amplitude = st.slider(
                            f"{distortion_type} Wave Amplitude",
                            0.0,
                            50.0,
                            centralized_distortion_settings.get(distortion_type, {}).get('warp_params', {}).get('wave_amplitude', 20.0)
                        )
                        wave_frequency = st.slider(
                            f"{distortion_type} Wave Frequency",
                            0.0,
                            0.1,
                            centralized_distortion_settings.get(distortion_type, {}).get('warp_params', {}).get('wave_frequency', 0.04)
                        )
                        bulge_factor = st.slider(
                            f"{distortion_type} Bulge Factor",
                            -50.0,
                            50.0,
                            centralized_distortion_settings.get(distortion_type, {}).get('warp_params', {}).get('bulge_factor', 30.0)
                        )
                        centralized_distortion_settings[distortion_type] = {
                            'intensity': intensity,
                            'warp_params': {
                                'wave_amplitude': wave_amplitude,
                                'wave_frequency': wave_frequency,
                                'bulge_factor': bulge_factor
                            }
                        }
                    else:
                        intensity = st.slider(
                            f"{distortion_type} Intensity",
                            0.0,
                            1.0,
                            centralized_distortion_settings.get(distortion_type, {}).get('intensity', 0.5)
                        )
                        centralized_distortion_settings[distortion_type] = {
                            'intensity': intensity
                        }

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
                    col1, col2 = st.columns(2)

                    with col1:
                        # Load and display original image
                        image = Image.open(file) if isinstance(file, str) else Image.open(file)
                        st.image(image, caption="Original Image", use_column_width=True)

                    with col2:
                        if use_centralized_distortions:
                            st.write("Using centralized distortion settings")
                            settings['distortions'] = centralized_distortions
                            for distortion_type in centralized_distortions:
                                if distortion_type in centralized_distortion_settings:
                                    for key, value in centralized_distortion_settings[distortion_type].items():
                                        settings[f"{distortion_type}_{key}"] = value
                        else:
                            # Multiple distortion selection
                            settings['distortions'] = st.multiselect(
                                "Choose Distortions:",
                                DISTORTION_TYPES[1:],  # Exclude "None" from the options
                                default=settings.get('distortions', []),
                                key=f"distortions_{i}"
                            )

                        # Distortion settings using tabs
                        if settings['distortions']:
                            tabs = st.tabs(settings['distortions'])
                            for tab, distortion_type in zip(tabs, settings['distortions']):
                                with tab:
                                    if distortion_type == "Color":
                                        settings[f"{distortion_type}_saturation"] = st.slider(
                                            "Saturation",
                                            0.0,
                                            2.0,
                                            settings.get(f"{distortion_type}_saturation", 1.0),
                                            key=f"saturation_{i}_{distortion_type}"
                                        )
                                        settings[f"{distortion_type}_hue_shift"] = st.slider(
                                            "Hue Shift",
                                            -0.5,
                                            0.5,
                                            settings.get(f"{distortion_type}_hue_shift", 0.0),
                                            key=f"hue_shift_{i}_{distortion_type}"
                                        )
                                    elif distortion_type == "Overlay":
                                        settings[f"{distortion_type}_intensity"] = st.slider(
                                            "Intensity",
                                            0.0,
                                            1.0,
                                            settings.get(f"{distortion_type}_intensity", 0.5),
                                            key=f"intensity_{i}_{distortion_type}"
                                        )

                                        # Add the file uploader for the overlay image
                                        overlay_image = st.file_uploader(
                                            f"Upload {distortion_type} image",
                                            type=["png", "jpg", "jpeg"],
                                            key=f"overlay_upload_{i}_{distortion_type}"
                                        )

                                        if overlay_image is not None:
                                            overlay_img = Image.open(overlay_image).convert("RGBA")
                                            overlay_img.thumbnail((300, 300))  # Resize to a manageable size
                                            img_byte_arr = io.BytesIO()
                                            overlay_img.save(img_byte_arr, format='PNG')
                                            settings[f"{distortion_type}_overlay_image"] = img_byte_arr.getvalue()
                                            st.success("Overlay image uploaded successfully.")
                                        elif f"{distortion_type}_overlay_image" not in settings:
                                            settings[f"{distortion_type}_overlay_image"] = None
                                            st.info("No overlay image selected.")
                                    elif distortion_type == "Warp":
                                        settings[f"{distortion_type}_intensity"] = st.slider(
                                            "Intensity",
                                            0.0,
                                            1.0,
                                            settings.get(f"{distortion_type}_intensity", 0.5),
                                            key=f"intensity_{i}_{distortion_type}"
                                        )
                                        settings[f"{distortion_type}_wave_amplitude"] = st.slider(
                                            "Wave Amplitude",
                                            0.0,
                                            50.0,
                                            settings.get(f"{distortion_type}_wave_amplitude", 20.0),
                                            key=f"wave_amplitude_{i}_{distortion_type}"
                                        )
                                        settings[f"{distortion_type}_wave_frequency"] = st.slider(
                                            "Wave Frequency",
                                            0.0,
                                            0.1,
                                            settings.get(f"{distortion_type}_wave_frequency", 0.04),
                                            key=f"wave_frequency_{i}_{distortion_type}"
                                        )
                                        settings[f"{distortion_type}_bulge_factor"] = st.slider(
                                            "Bulge Factor",
                                            -50.0,
                                            50.0,
                                            settings.get(f"{distortion_type}_bulge_factor", 30.0),
                                            key=f"bulge_factor_{i}_{distortion_type}"
                                        )
                                    else:
                                        settings[f"{distortion_type}_intensity"] = st.slider(
                                            "Intensity",
                                            0.0,
                                            1.0,
                                            settings.get(f"{distortion_type}_intensity", 0.5),
                                            key=f"intensity_{i}_{distortion_type}"
                                        )

                        # Apply distortions and display processed image
                        distortions_list = []
                        for distortion_type in settings['distortions']:
                            distortion_params = {"type": distortion_type}
                            if distortion_type == "Color":
                                distortion_params["saturation"] = settings.get(f"{distortion_type}_saturation", 1.0)
                                distortion_params["hue_shift"] = settings.get(f"{distortion_type}_hue_shift", 0.0)
                            elif distortion_type == "Overlay":
                                distortion_params["intensity"] = settings.get(f"{distortion_type}_intensity", 0.5)
                                overlay_image_bytes = settings.get(f"{distortion_type}_overlay_image")
                                if overlay_image_bytes:
                                    distortion_params["overlay_image"] = Image.open(io.BytesIO(overlay_image_bytes)).convert("RGBA")
                                else:
                                    distortion_params["overlay_image"] = None
                            elif distortion_type == "Warp":
                                distortion_params["intensity"] = settings.get(f"{distortion_type}_intensity", 0.5)
                                distortion_params["warp_params"] = {
                                    "wave_amplitude": settings.get(f"{distortion_type}_wave_amplitude", 20.0),
                                    "wave_frequency": settings.get(f"{distortion_type}_wave_frequency", 0.04),
                                    "bulge_factor": settings.get(f"{distortion_type}_bulge_factor", 30.0)
                                }
                            else:
                                distortion_params["intensity"] = settings.get(f"{distortion_type}_intensity", 0.5)
                            distortions_list.append(distortion_params)

                        # Only apply distortions if there are valid distortions to apply
                        if any(d for d in distortions_list if d.get("overlay_image") is not None or d["type"] != "Overlay"):
                            processed_image = apply_distortions(image, distortions_list)
                        else:
                            processed_image = image

                        st.image(processed_image, caption="Processed Image", use_column_width=True)

                        # Input text
                        st.markdown("### Prompt Settings")
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
                                value=settings.get("input_text", ""),
                                key=f"input_{i}"
                            )

                st.markdown("---")  # Add a separator between images

        # Button to start bulk analysis
        if st.button("Run Bulk Analysis") and uploaded_files:
            results = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                try:
                    image = Image.open(file)
                    file_name = file.name if hasattr(file, 'name') else os.path.basename(file)

                    # Apply distortions
                    distortions_list = []
                    if use_centralized_distortions:
                        for distortion_type in centralized_distortions:
                            distortion_params = {"type": distortion_type}
                            if distortion_type == "Overlay":
                                distortion_params["intensity"] = centralized_distortion_settings[distortion_type]['intensity']
                                # Convert bytes back to PIL Image for overlay
                                if centralized_distortion_settings[distortion_type]['overlay_image']:
                                    overlay_bytes = centralized_distortion_settings[distortion_type]['overlay_image']
                                    distortion_params["overlay_image"] = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
                                else:
                                    distortion_params["overlay_image"] = None
                            elif distortion_type == "Color":
                                distortion_params.update(centralized_distortion_settings[distortion_type])
                            elif distortion_type == "Warp":
                                distortion_params["intensity"] = centralized_distortion_settings[distortion_type]['intensity']
                                distortion_params["warp_params"] = centralized_distortion_settings[distortion_type]['warp_params']
                            else:
                                distortion_params["intensity"] = centralized_distortion_settings[distortion_type]['intensity']
                            distortions_list.append(distortion_params)
                    else:
                        settings = st.session_state.image_settings[i]
                        for distortion_type in settings['distortions']:
                            distortion_params = {"type": distortion_type}
                            if distortion_type == "Color":
                                distortion_params["saturation"] = settings.get(f"{distortion_type}_saturation", 1.0)
                                distortion_params["hue_shift"] = settings.get(f"{distortion_type}_hue_shift", 0.0)
                            elif distortion_type == "Overlay":
                                distortion_params["intensity"] = settings.get(f"{distortion_type}_intensity", 0.5)
                                overlay_image_bytes = settings.get(f"{distortion_type}_overlay_image")
                                if overlay_image_bytes:
                                    distortion_params["overlay_image"] = Image.open(io.BytesIO(overlay_image_bytes)).convert("RGBA")
                                else:
                                    distortion_params["overlay_image"] = None
                            elif distortion_type == "Warp":
                                distortion_params["intensity"] = settings.get(f"{distortion_type}_intensity", 0.5)
                                distortion_params["warp_params"] = {
                                    "wave_amplitude": settings.get(f"{distortion_type}_wave_amplitude", 20.0),
                                    "wave_frequency": settings.get(f"{distortion_type}_wave_frequency", 0.04),
                                    "bulge_factor": settings.get(f"{distortion_type}_bulge_factor", 30.0)
                                }
                            else:
                                distortion_params["intensity"] = settings.get(f"{distortion_type}_intensity", 0.5)
                            distortions_list.append(distortion_params)

                    # Only apply distortions if there are valid distortions to apply
                    if any(d for d in distortions_list if d.get("overlay_image") is not None or d["type"] != "Overlay"):
                        processed_image = apply_distortions(image, distortions_list)
                    else:
                        processed_image = image

                    # Modify this part to include intensity information
                    distortions_info = []
                    for d in distortions_list:
                        if d['type'] == 'Color':
                            distortions_info.append(f"{d['type']} (Saturation: {d['saturation']:.2f}, Hue Shift: {d['hue_shift']:.2f})")
                        elif d['type'] == 'Warp':
                            distortions_info.append(f"{d['type']} (Intensity: {d['intensity']:.2f}, Wave Amp: {d['warp_params']['wave_amplitude']:.2f}, Wave Freq: {d['warp_params']['wave_frequency']:.2f}, Bulge: {d['warp_params']['bulge_factor']:.2f})")
                        else:
                            distortions_info.append(f"{d['type']} (Intensity: {d['intensity']:.2f})")

                    # Get AI response
                    text_response, json_response = get_gemini_response(
                        settings["input_text"],
                        processed_image,
                        st.session_state.model_choice,
                        st.session_state.system_instructions if st.session_state.use_system_instructions else None,
                        EXPECTED_JSON_FIELDS  # Add this line
                    )

                    # Create a result dictionary with basic info
                    result = {
                        "Image": file_name,
                        "Distortions": ', '.join(distortions_info),
                        "Input Text": settings["input_text"],
                        "AI Response": text_response,
                        "JSON Response": json.dumps(json_response, indent=2)
                    }

                    # Add result to list
                    results.append(result)

                    # Show AI response
                    st.write(f"AI Response for {file_name}:")
                    st.write(text_response)

                    st.markdown("---")  # Add a separator between images

                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
                    st.error(traceback.format_exc())

                progress_bar.progress((i + 1) / len(uploaded_files))

            if results:
                # Create DataFrame
                results_df = pd.DataFrame(results)

                # Add JSON fields as separate columns
                for field in EXPECTED_JSON_FIELDS:
                    results_df[field] = results_df['JSON Response'].apply(
                        lambda x: json.loads(x).get(field, '')
                    )
                    # Check if the field contains a list and join it into a string
                    if results_df[field].dtype == 'object':
                        results_df[field] = results_df[field].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else x
                        )

                # Remove empty columns
                results_df = results_df.dropna(axis=1, how='all')

                # Remove columns that are entirely empty strings
                results_df = results_df.loc[:, (results_df != '').any()]

                # Reorder columns
                base_columns = ["Image", "Distortions", "Input Text", "AI Response", "JSON Response"]
                json_columns = [col for col in EXPECTED_JSON_FIELDS if col in results_df.columns]
                columns_order = base_columns + json_columns

                # Only include columns that exist in the DataFrame
                columns_order = [col for col in columns_order if col in results_df.columns]
                results_df = results_df[columns_order]

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