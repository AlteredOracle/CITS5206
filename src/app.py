import streamlit as st
import os
import google.generativeai as genai

# Set page configuration
st.set_page_config(page_title="Multimodal LLM Road Safety Platform", layout="wide")

st.title("Multimodal LLM Road Safety Platform")

api_key = st.text_input("Enter your Gemini API key:", type="password")

if api_key:
    os.environ['GEMINI_API_KEY'] = api_key
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    st.success("API key set successfully!")

    # Sidebar for settings
    st.sidebar.title("Settings")

    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["gemini-1.5-flash-latest", "gemini-1.5-pro"]
    )

    # Add a new option in the sidebar for analysis mode
    analysis_mode = st.sidebar.radio("Analysis Mode", ["Single", "Bulk"])

    if analysis_mode == "Single":
        st.write("Single image analysis mode selected.")
    else:  # Bulk Analysis
        st.write("Bulk analysis mode selected.")
else:
    st.warning("Please enter your API key to proceed.")
