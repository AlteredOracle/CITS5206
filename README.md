# CITS5206 Group 8 Project: Multimodal LLM Road Safety Platform

## Project Overview

This project implements a Multimodal Large Language Model (LLM) Road Safety Platform using Streamlit and Google's Gemini 1.5 Flash AI model. The platform is designed to analyze road safety scenarios using advanced AI, providing insights and recommendations based on both textual and visual inputs.

## Project Scope

The scope of this project includes:

1. Development of a user-friendly web interface using Streamlit.
2. Integration with Google's Gemini 1.5 Flash AI model.
3. Implementation of image processing capabilities to simulate various environmental conditions.
4. Analysis of road safety scenarios through text and image inputs.
5. Generation of AI-powered responses and recommendations for road safety improvements.
6. Bulk analysis of multiple images with customizable settings.

## Group Members

| UWA ID   | Name                 | GitHub Username   |
|----------|----------------------|-------------------|
| 23832048 | Gnaneshwar Reddy Bana| gnaneshwarbana    |
| 23959947 | Kanishk Kanishk      | kanishk-uwa       |
| 23870387 | Pedro Wang           | CoderPdr          |
| 22941307 | Sarath Pathari       | AlteredOracle     |
| 23743373 | Yuxin Gu             | SoleilGU          |
| 23633858 | Yuanfu Cao           | Cyf1160819266     |

## Features

- Text and image input for analysis
- Integration with Gemini 1.5 Flash and Gemini 1.5 Pro models
- Image distortion options:
  - Blur
  - Brightness
  - Contrast
  - Sharpness
  - Color (with saturation and hue shift)
  - Rain effect
  - Overlay (with custom image upload)
  - Warp (with customizable wave and bulge effects)
- Adjustable distortion intensity for each effect
- Batch processing of multiple images
- Bulk analysis with centralized or individual image settings
- Support for folder path input for bulk analysis
- Customizable system instructions for AI
- Predefined and custom prompts for analysis
- AI-generated responses and recommendations for road safety scenarios
- Structured CSV output for analysis results

## Technical Stack

![Python](https://img.shields.io/badge/Python-v3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.28.0+-blue)
![Google Generative AI](https://img.shields.io/badge/Google_Generative_AI-v0.3.1+-blue)
![Pillow](https://img.shields.io/badge/Pillow-v10.0.0+-blue)
![NumPy](https://img.shields.io/badge/NumPy-v1.24.0+-blue)
![SciPy](https://img.shields.io/badge/SciPy-v1.10.0+-blue)
![Pandas](https://img.shields.io/badge/Pandas-v2.0.0+-blue)

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/AlteredOracle/CITS5206.git
   cd CITS5206
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

5. Open a web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. Enter your Gemini API key in the provided field when you start the app.
2. Choose between single image analysis or bulk analysis mode.
3. For single image analysis:
   - Enter your text prompt or select a predefined one.
   - Upload an image related to the scenario.
   - Select and adjust image distortions if desired.
   - Click "Analyse" to get the AI-generated response.
4. For bulk analysis:
   - Choose to upload multiple files or specify a folder path.
   - Set centralized distortion settings or customize for each image.
   - Run the bulk analysis to process all images and generate a CSV report.

## Project Structure

## Application Structure

The application stack for this project is illustrated in the following diagram:

![Application Stack](https://github.com/AlteredOracle/CITS5206/blob/main/Project%20Documents/Application%20Stack.png)

This diagram outlines the key components and technologies used in our Multimodal LLM Road Safety Platform, including:

- Frontend: Streamlit
- Backend: Python
- AI Model: Google Gemini 1.5 Flash
- Image Processing: PIL (Python Imaging Library)
- Data Handling: Pandas, NumPy

The structure showcases how user inputs are processed through our application, leveraging various libraries and the Gemini AI model to analyze road safety scenarios and provide insights.

## Design Mockups

For design mockups and visual representations of the Multimodal LLM Road Safety Platform, please refer to the following link:

[View Design on Figma](https://www.figma.com/design/XaY1Gj4GGDYnQT3a7rHhKs/Multimodal-LLM-Road-Safety-Platform?node-id=0-1&t=m9u1DMpXtYAX23DD-1)

