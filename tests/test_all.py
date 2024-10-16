import pytest
from PIL import Image
import io
import numpy as np
from src.utils import (
    apply_distortion,
    shift_hue,
    apply_rain_effect,
    apply_overlay,
    apply_warp_effect,
    apply_distortions,
    get_gemini_response
)
import google.generativeai as genai

def create_test_image(size=(100, 100), color='red'):
    return Image.new('RGB', size, color=color)

def test_apply_distortion_color():
    image = create_test_image()
    result = apply_distortion(image, "Color", saturation=1.5, hue_shift=0.2)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_distortion_blur():
    image = create_test_image()
    result = apply_distortion(image, "Blur", intensity=0.5)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_distortion_brightness():
    image = create_test_image()
    result = apply_distortion(image, "Brightness", intensity=0.5)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_distortion_contrast():
    image = create_test_image()
    result = apply_distortion(image, "Contrast", intensity=0.5)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_distortion_sharpness():
    image = create_test_image()
    result = apply_distortion(image, "Sharpness", intensity=0.5)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_distortion_rain():
    image = create_test_image()
    result = apply_distortion(image, "Rain", intensity=0.5)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_distortion_overlay():
    image = create_test_image()
    overlay = create_test_image(color='blue')
    result = apply_distortion(image, "Overlay", intensity=0.5, overlay_image=overlay)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_distortion_warp():
    image = create_test_image()
    warp_params = {'wave_amplitude': 20, 'wave_frequency': 0.05, 'bulge_factor': 30}
    result = apply_distortion(image, "Warp", intensity=0.5, warp_params=warp_params)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_shift_hue():
    image = create_test_image()
    result = shift_hue(image, 0.5)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_rain_effect():
    image = create_test_image()
    result = apply_rain_effect(image, 0.5)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_overlay():
    image = create_test_image()
    overlay = create_test_image(color='blue')
    result = apply_overlay(image, 0.5, overlay)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_warp_effect():
    image = create_test_image()
    warp_params = {'wave_amplitude': 20, 'wave_frequency': 0.05, 'bulge_factor': 30}
    result = apply_warp_effect(image, 0.5, warp_params)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_distortions():
    image = create_test_image()
    distortions = [
        {'type': 'Blur', 'intensity': 0.5},
        {'type': 'Brightness', 'intensity': 0.2},
        {'type': 'Color', 'saturation': 1.5, 'hue_shift': 0.1},
    ]
    result = apply_distortions(image, distortions)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_get_gemini_response(mocker):
    # Mock the GenerativeModel
    mock_model = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.text = "Test response ===JSON==={}===JSON==="
    mock_model.generate_content.return_value = mock_response
    mocker.patch('google.generativeai.GenerativeModel', return_value=mock_model)

    input_text = "Test input"
    image = create_test_image()
    model_name = "test-model"
    system_instructions = "Test instructions"
    expected_fields = ["field1", "field2"]

    text_response, json_response = get_gemini_response(input_text, image, model_name, system_instructions, expected_fields)

    assert isinstance(text_response, str)
    assert isinstance(json_response, dict)
    mock_model.generate_content.assert_called_once()

# Additional tests for edge cases and error handling

def test_apply_distortion_invalid_type():
    image = create_test_image()
    result = apply_distortion(image, "InvalidType")
    assert result == image  # Should return the original image for invalid type

def test_apply_distortion_extreme_values():
    image = create_test_image()
    result = apply_distortion(image, "Brightness", intensity=100)
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_shift_hue_extreme_values():
    image = create_test_image()
    result = shift_hue(image, 10)  # Extreme hue shift
    assert isinstance(result, Image.Image)
    assert result.size == image.size
    assert result.mode == 'RGB'

def test_apply_rain_effect_zero_intensity():
    image = create_test_image()
    result = apply_rain_effect(image, 0)
    assert np.array_equal(np.array(result), np.array(image))  # Should be identical to original

def test_apply_overlay_null_overlay():
    image = create_test_image()
    result = apply_overlay(image, 0.5, None)
    assert np.array_equal(np.array(result), np.array(image))  # Should be identical to original

def test_apply_warp_effect_zero_intensity():
    image = create_test_image()
    warp_params = {'wave_amplitude': 0, 'wave_frequency': 0, 'bulge_factor': 0}
    result = apply_warp_effect(image, 0, warp_params)
    assert np.array_equal(np.array(result), np.array(image))  # Should be identical to original

def test_get_gemini_response_error_handling(mocker):
    mock_model = mocker.Mock()
    mock_model.generate_content.side_effect = Exception("API Error")
    mocker.patch('google.generativeai.GenerativeModel', return_value=mock_model)

    input_text = "Test input"
    image = create_test_image()
    model_name = "test-model"
    system_instructions = "Test instructions"
    expected_fields = ["field1", "field2"]

    text_response, json_response = get_gemini_response(input_text, image, model_name, system_instructions, expected_fields)

    assert "Error generating response" in text_response
    assert "error" in json_response