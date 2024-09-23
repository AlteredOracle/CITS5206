from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import os
from PIL import Image
from utils import file_does_exist
from utils import is_valid_image
from utils import timeout
from utils import TimeoutException
import streamlit as st


@timeout(timeout=20)
def generate_content_timeout(model, prompt, image=None):
    """
    generate content using AI model with a timeout setting.
    """
    if image:
        return model.generate_content([prompt, image])
    return model.generate_content(prompt)


def get_analysis(prompt, image_path=None):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    image = Image.open(image_path) if image_path else None

    try:
        resp = generate_content_timeout(model, prompt, image)
        print("AI Analysis Result:")
        print(resp.text)
    except google_exceptions.InvalidArgument as e:
        print(f"Invalid argument error: {str(e)}")
    except google_exceptions.ResourceExhausted:
        print("Error: API quota exceeded. Please try again later or upgrade your plan.")
    except TimeoutException:
        print(
            "Error: Request timed out. Please try again or check your internet connection.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def main():
    load_dotenv()

    # Set page configuration
    st.set_page_config(
        page_title="Multimodal LLM Road Safety Platform", layout="wide")

    st.title("Multimodal LLM Road Safety Platform")

    api_key = st.text_input("Enter your Gemini API key:", type="password")
    prompt = input("Enter your prompt for the AI: ")
    image_path = input(
        "Enter the path to the image (or press Enter to skip): ").strip()
    
    if api_key:
        os.environ['GEMINI_API_KEY'] = api_key
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        st.success("API key set successfully!")
    else:
        st.warning("Please enter your API key to proceed.")

    if image_path:
        if not file_does_exist(image_path):
            print("Error: invalid file path or the file does not exist.")
            return
        elif not is_valid_image(image_path):
            print("Error: not an image file, please check again.")
            return
        else:
            get_analysis(prompt, image_path)
    else:
        get_analysis(prompt)


if __name__ == "__main__":
    main()
