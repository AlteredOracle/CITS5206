from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image

def get_analysis(prompt, image_path=None):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    try:
        if image_path:
            image = Image.open(image_path)
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        print("AI Analysis Result:")
        print(response.text)
    except Exception as e:
        print(f"Error generating content: {str(e)}")

def main():
    load_dotenv()
    if os.getenv("GEMINI_API_KEY") is None:
        print("Error: GEMINI_API_KEY env variable not set.")
        return
    
    prompt = input("Enter your prompt for the AI: ")
    image_path = input("Enter the path to the image (or press Enter to skip): ")
    
    if image_path.strip():
        get_analysis(prompt, image_path)
    else:
        get_analysis(prompt)

if __name__ == "__main__":
    main()