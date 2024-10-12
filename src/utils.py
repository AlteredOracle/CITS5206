from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
import random
import google.generativeai as genai
import io
import numpy as np
from scipy.ndimage import map_coordinates
import traceback
import json
import re

def apply_distortion(image, type, **params):
    print(f"Applying distortion: {type}")  # Debug print
    if type == "Color":
        if "saturation" in params:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(params["saturation"])
        
        if "hue_shift" in params:
            image = shift_hue(image, params["hue_shift"])
        
        print(f"Color distortion applied. Original size: {image.size}, Distorted size: {image.size}")  # Debug print
        return image
    elif type == "Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=params.get("intensity", 0) * 10))
    elif type == "Brightness":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1 + params.get("intensity", 0))
    elif type == "Contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1 + params.get("intensity", 0))
    elif type == "Sharpness":
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1 + (params.get("intensity", 0) * 4))
    elif type == "Rain":
        return apply_rain_effect(image, params.get("intensity", 0))
    elif type == "Overlay":
        return apply_overlay(image, params.get("intensity", 0), params.get("overlay_image", None))
    elif type == "Warp":
        return apply_warp_effect(image, params.get("intensity", 0), params.get("warp_params", None))
    return image

def shift_hue(image, amount):
    img_hsv = image.convert('HSV')
    h, s, v = img_hsv.split()
    h = h.point(lambda x: (x + amount * 255) % 255)
    return Image.merge('HSV', (h, s, v)).convert('RGB')

def apply_rain_effect(image, intensity):
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = image.size
    for _ in range(int(intensity * 1000)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        length = random.randint(10, 20)
        draw.line((x, y, x + random.randint(-2, 2), y + length), fill=(255, 255, 255, random.randint(50, 150)), width=1)
    
    rain_overlay = overlay.filter(ImageFilter.GaussianBlur(1))
    return Image.alpha_composite(image.convert("RGBA"), rain_overlay).convert("RGB")

def apply_overlay(image, intensity, overlay_image):
    if overlay_image is None:
        return image
    
    try:
        if isinstance(overlay_image, (bytes, io.BytesIO)):
            overlay = Image.open(io.BytesIO(overlay_image)).convert("RGBA")
        elif isinstance(overlay_image, Image.Image):
            overlay = overlay_image.convert("RGBA")
        elif isinstance(overlay_image, str):
            overlay = Image.open(overlay_image).convert("RGBA")
        else:
            print(f"Unsupported overlay_image type: {type(overlay_image)}")
            return image
        
        overlay = overlay.resize(image.size)
        
        # Create a new image with the same size as the original
        result = Image.new("RGBA", image.size)
        
        # Paste the original image
        result.paste(image.convert("RGBA"), (0, 0))
        
        # Apply the overlay with the given intensity
        overlay = Image.blend(Image.new("RGBA", image.size, (0, 0, 0, 0)), overlay, intensity)
        result = Image.alpha_composite(result, overlay)
        
        return result.convert("RGB")
    except Exception as e:
        print(f"Error applying overlay: {str(e)}")
        traceback.print_exc()
        return image  # Return the original image if there's an error

def apply_warp_effect(image, intensity, warp_params):
    try:
        img = np.array(image)
        rows, cols = img.shape[0], img.shape[1]
        
        # Create meshgrid
        src_cols, src_rows = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))
        
        # Wave effect
        wave_amplitude = warp_params.get('wave_amplitude', 20) * intensity
        wave_frequency = warp_params.get('wave_frequency', 0.05) * 10  # Increase frequency impact
        dst_rows = src_rows + np.sin(src_cols * wave_frequency) * wave_amplitude
        dst_cols = src_cols + np.sin(src_rows * wave_frequency) * wave_amplitude
        
        # Bulge/Pinch effect
        center_row, center_col = rows // 2, cols // 2
        dist_from_center = np.sqrt((src_rows - center_row)**2 + (src_cols - center_col)**2)
        
        bulge_factor = warp_params.get('bulge_factor', 30) * intensity * 2  # Increase bulge impact
        max_dist = np.sqrt(center_row**2 + center_col**2)
        
        # Normalize distances
        dist_from_center = dist_from_center / max_dist
        
        # Apply bulge/pinch
        factor = (1 - dist_from_center**2) * bulge_factor
        dst_rows += (src_rows - center_row) * factor / (rows / 4)  # Increase effect
        dst_cols += (src_cols - center_col) * factor / (cols / 4)  # Increase effect
        
        # Map coordinates
        warped = np.zeros_like(img)
        for i in range(min(3, img.shape[2])):  # Handle both RGB and RGBA
            warped[:,:,i] = map_coordinates(img[:,:,i], [dst_rows, dst_cols], order=1, mode='reflect')
        
        if img.shape[2] == 4:  # If RGBA, copy the alpha channel
            warped[:,:,3] = img[:,:,3]
        
        result = Image.fromarray(warped.astype(np.uint8))
        print(f"Warp effect applied successfully. Max pixel diff: {np.max(np.abs(img - warped))}")
        return result
    except Exception as e:
        print(f"Error in apply_warp_effect: {str(e)}")
        print(traceback.format_exc())
        return image  # Return the original image if there's an error

def apply_distortions(image, distortions):
    for distortion in distortions:
        image = apply_distortion(image, **distortion)
    return image

def get_gemini_response(input_text, image, model_name, system_instructions, expected_fields):
    model = genai.GenerativeModel(model_name)
    response = None
    
    # Add the JSON request to the system instructions internally
    json_request = f"""
    After your natural language response, please provide a JSON representation of your analysis.
    The JSON structure should include the following fields (only include non-empty fields):
    {', '.join(expected_fields)}
    Ensure that the content in the JSON matches your natural language response exactly.
    Enclose the JSON structure within ===JSON=== tags.
    """
    
    full_instructions = f"{system_instructions}\n\n{json_request}" if system_instructions else json_request

    # Ensure the image is in the correct format
    if image:
        if isinstance(image, Image.Image):
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
        elif isinstance(image, bytes):
            img_byte_arr = image
        else:
            raise ValueError("Unsupported image type. Expected PIL Image or bytes.")
    else:
        img_byte_arr = None
    
    try:
        content = []
        if full_instructions:
            content.append(full_instructions)
        if input_text:
            content.append(input_text)
        if img_byte_arr:
            content.append({"mime_type": "image/png", "data": img_byte_arr})
        
        if content:
            response = model.generate_content(content)
            text_response = response.text if response else "No response from the model."
            
            # Extract JSON from the response
            json_match = re.search(r'===JSON===\s*(.*?)\s*===JSON===', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    json_response = json.loads(json_str)
                    # Remove empty fields from the JSON response
                    json_response = {k: v for k, v in json_response.items() if v}
                    # Remove the JSON part from the text response
                    text_response = re.sub(r'===JSON===.*===JSON===', '', text_response, flags=re.DOTALL).strip()
                except json.JSONDecodeError:
                    json_response = {"error": "Failed to parse JSON from AI response"}
            else:
                json_response = {"error": "No JSON found in AI response"}
            
            return text_response, json_response  # Return JSON as a Python dictionary
        else:
            return "No input provided to the model.", {}
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        return error_message, {"error": error_message}
