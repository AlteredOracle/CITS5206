import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from utils import apply_distortion, get_gemini_response
import pandas as pd


# Define constants
NUM_COL = 5

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
    "Check System Instructions", value=False
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

st.session_state.api_key = st.text_input(
    "Enter your Gemini API key:",
    type="password",
    value=st.session_state.api_key
)

if st.session_state.api_key:
    os.environ['GEMINI_API_KEY'] = st.session_state.api_key
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])

    st.sidebar.subheader("Distortions")

    # Set some default value for distortion
    # These values are the values where the image stay original.
    overlay_image = None
    overlay_intensity = 1.0
    warp_params = {}
    blur_intensity = 0.0
    brightness_intensity = 0.0
    contrast_intensity = 0.0
    sharpness_intensity = 0.0
    saturation_intensity = 1.0
    hue_shift = 1.0

    with st.sidebar.expander("Rain Effect"):
        rain_intensity = st.slider(
            "Rain Intensity", 0.0, 1.0, 0.0
        )

    with st.sidebar.expander("Wrapping Effect"):
        warp_params["wave_amplitude"] = st.slider(
            "Wave Amplitude", 0.0, 1.0, 0.0
        )
        warp_params["wave_frequency"] = st.slider(
            "Wave Frequency", 0.0, 1.0, 0.0
        )
        warp_params["bulge_factor"] = st.slider(
            "Bulge Factor", 0.0, 1.0, 0.0
        )

    with st.sidebar.expander("Overlay Effect"):
        overlay_image = st.file_uploader(
            "Upload overlay image",
            type=["png", "jpg", "jpeg"]
        )

        if overlay_image:
            overlay_intensity = st.slider(
                "Overlay Transparency", 0.0, 1.0, 1.0
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

    # Analysis part
    folder_path = st.text_input(
        "Enter the absolute folder path containing images:"
    )

    processed_images = []

    if folder_path and os.path.isdir(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        uploaded_files = [os.path.join(folder_path, f) for f in image_files]
        st.success(f"Found {len(uploaded_files)} images in the specified folder.")

        # Display images
        if uploaded_files:
            for i in range(0, len(uploaded_files), NUM_COL):
                cols = st.columns(NUM_COL)
                for j in range(min(NUM_COL, len(uploaded_files) - i)):
                    image_path = uploaded_files[i + j]

                    # Apply possible distortion
                    image = Image.open(image_path)
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

                    # Add processed images to a list for later uploading
                    processed_images.append((processed_image, os.path.basename(image_path)))
                    with cols[j]:
                        st.image(
                            processed_image,
                            caption=os.path.basename(image_path),
                            use_column_width=True
                        )
        else:
            st.error("Invalid folder path. Please check and try again.")
            uploaded_files = []
    else:
        uploaded_files = []

    # Start analysis
    if st.button("Run Analysis") and uploaded_files:
        results = []
        progress_bar = st.progress(0)

        for i in range(len(processed_images)):
            image, image_name = processed_images[i]
            try:
                response = get_gemini_response(
                    input_text,
                    image,
                    st.session_state.model_choice,
                    st.session_state.system_instructions if st.session_state.use_system_instructions else None
                )

                # Add result to list
                results.append({
                    "Image": image_name,
                    "Input Text": input_text,
                    "Rain Intensity": rain_intensity,
                    "Overlay Image": overlay_image.name if overlay_image else "",
                    "Overlay Intensity": overlay_intensity,
                    "Wave Amplitude": warp_params["wave_amplitude"],
                    "Wave Frequency": warp_params["wave_frequency"],
                    "Bulge Factor": warp_params["bulge_factor"],
                    "Blur Intensity": blur_intensity,
                    "Brightness Intensity": brightness_intensity,
                    "Contrast Intensity": contrast_intensity,
                    "Sharpness Intensity": sharpness_intensity,
                    "Saturation Intensity": saturation_intensity,
                    "Hue Shift": hue_shift,
                    "AI Response": response
                })

                # Show AI response
                st.write(f"AI Response for {image_name}:")
                st.write(response)
                st.markdown("---")  # Add a separator between images

            except Exception as e:
                st.error(f"Error processing {image_name}: {str(e)}")

            progress_bar.progress((i + 1) / len(processed_images))

        if results:
            results_df = pd.DataFrame(results)
            st.subheader("Analysis Results")
            st.dataframe(results_df)

            # Convert DataFrame to CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="analysis_results.csv",
                mime="text/csv",
            )
            processed_images = []
        else:
            st.warning("No results were generated. Please check your inputs and try again.")
    elif not uploaded_files:
        st.warning("Please upload at least one image or specify a valid folder path to proceed with bulk analysis.")

else:
    st.warning("Please enter your API key to proceed.")
