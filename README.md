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
- Integration with Gemini 1.5 Flash model
- Image distortion options:
  - Blur
  - Brightness
  - Contrast
  - Sharpness
  - Color
  - Rain effect
  - Overlay
  - Warp
- Adjustable distortion intensity
- Batch processing of multiple images
- AI-generated responses and recommendations for road safety scenarios
- Structured CSV output for analysis results

## Technical Stack

![Python](https://img.shields.io/badge/Python-v3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.0.0-blue) 
![Google Generative AI](https://img.shields.io/badge/Google_Generative_AI-Gemini_1.5_Flash-blue)
![Pillow](https://img.shields.io/badge/Pillow-v8.0.0-blue)
![SciPy](https://img.shields.io/badge/SciPy-v1.7.0-blue)

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
   streamlit run app.py
   ```

5. Open a web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. Enter your Gemini API key in the provided field when you start the app.
2. Enter your text prompt describing the road safety scenario or question.
3. Upload an image related to the scenario.
4. Select and adjust image distortions if desired.
5. Click "Analyse" to get the AI-generated response.

## Project Structure

