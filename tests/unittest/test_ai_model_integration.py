import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from src.app import generate_content_timeout, get_analysis, file_does_exist, is_valid_image, TimeoutException
from google.api_core import exceptions as google_exceptions

class TestGenerateContentTimeout(unittest.TestCase):

    @patch("src.app.genai.GenerativeModel")
    def test_generate_content_success(self, mock_gen_model):
        # Mock the generative model
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance
        mock_model_instance.generate_content.return_value = MagicMock(text="Generated content")

        # Call the function and verify the return result
        result = generate_content_timeout(mock_model_instance, "Test prompt")
        mock_model_instance.generate_content.assert_called_once_with("Test prompt")
        self.assertEqual(result.text, "Generated content")

    @patch("src.app.genai.GenerativeModel")  # Mock AI model generation
    def test_generate_content_no_timeout(self, mock_gen_model):
        # Mock the AI model instance
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance

        # Simulate generate_content throwing a TimeoutException
        mock_model_instance.generate_content.side_effect = TimeoutException

        # Call the function and verify that a TimeoutException is thrown
        with self.assertRaises(TimeoutException):
            generate_content_timeout(mock_model_instance, "Test prompt")



