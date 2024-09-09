from dotenv import load_dotenv
import os

def get_analysis(key, image_path="path/to/iamge"):
    print(f"{key}\n{image_path}")

def main():
    load_dotenv()
    key = os.getenv("gemini_api_key")
    if key is None:
        print("Error: gemini_api_key env variable not set.")
        return
    get_analysis(key)


main()