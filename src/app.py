import streamlit as st
import os
import google.generativeai as genai

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
        st.write("Single image analysis mode selected.")
    # If 'Bulk' mode is selected, display the corresponding message
    else:  # Bulk Analysis
        st.write("Bulk analysis mode selected.")
else:
    # Display a warning if the API key is not provided
    st.warning("Please enter your API key to proceed.")
