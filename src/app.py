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
    # Configure the Gemini API using the API key from the environment variable
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Select the generative model 'gemini-1.5-flash-latest' from the Gemini API
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Open the image from the provided image path, or set to None if no image path is provided
    image = Image.open(image_path) if image_path else None

    try:
        # Call the generate_content_timeout function with the selected model, prompt, and image
        resp = generate_content_timeout(model, prompt, image)

        # Print the response text to the console
        print("AI Analysis Result:")
        print(resp.text)
    except google_exceptions.InvalidArgument as e:
        # Catch invalid argument errors and print an error message
        print(f"Invalid argument error: {str(e)}")
    except google_exceptions.ResourceExhausted:
        # Catch resource exhaustion error and print an error message
        print("Error: API quota exceeded. Please try again later or upgrade your plan.")
    except TimeoutException:
        # Catch timeout errors and print an error message
        print(
            "Error: Request timed out. Please try again or check your internet connection.")
    except Exception as e:
        # Catch any other unexpected errors and print an error message
        print(f"Unexpected error: {str(e)}")


def main():
    # Load environment variables from a .env file
    load_dotenv()

    # Set the page configuration to customize the appearance of the Streamlit app
    st.set_page_config(
        page_title="Multimodal LLM Road Safety Platform", 
        layout="wide"
    )

    # Display the title of the app
    st.title("Multimodal LLM Road Safety Platform")

    # Create a text input field to enter the Gemini API key
    api_key = st.text_input("Enter your Gemini API key:", type="password")

    # Check if an API key has been entered
    if api_key:
        # Set the API key as an environment variable
        os.environ['GEMINI_API_KEY'] = api_key
        
        # Configure the genai library with the provided API key
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        
        # Display a success message to indicate that the API key has been set successfully
        st.success("API key set successfully!")
    else:
        # Display a warning message to prompt the user to enter their API key
        st.warning("Please enter your API key to proceed.")

    # Check if GEMINI_API_KEY environment variable is set
    if os.getenv("GEMINI_API_KEY") is None:
        # If not set, print an error message and exit
        print("Error: GEMINI_API_KEY env variable not set.")
        return

    # Prompt the user to enter a prompt for the AI
    prompt = input("Enter your prompt for the AI: ")

    # Prompt the user to enter the path to an image file
    image_path = input(
        "Enter the path to the image (or press Enter to skip): ").strip()

    # If an image path is provided
    if image_path:
        # Check if the file exists at the specified path
        if not file_does_exist(image_path):
            # If the file doesn't exist, print an error message and exit
            print("Error: invalid file path or the file does not exist.")
            return
        # Check if the file is a valid image
        elif not is_valid_image(image_path):
            # If the file is not a valid image, print an error message and exit
            print("Error: not an image file, please check again.")
            return
        else:
            # Call get_analysis() function with both prompt and image_path if image path is valid
            get_analysis(prompt, image_path)
    else:
        # Call get_analysis() with only the prompt if image path is not provided
        get_analysis(prompt)


if __name__ == "__main__":
    main()
