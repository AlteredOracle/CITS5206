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

class TestGetAnalysis(unittest.TestCase):

    @patch("src.app.generate_content_timeout", side_effect=google_exceptions.InvalidArgument("Invalid"))
    def test_get_analysis_invalid_argument(self, mock_generate_content):
        # Test whether the function correctly handles the InvalidArgument exception
        try:
            get_analysis("Test prompt")
        except google_exceptions.InvalidArgument as e:
            self.fail(f"get_analysis raised {e} unexpectedly!")

    @patch("src.app.generate_content_timeout", side_effect=TimeoutException)
    def test_get_analysis_timeout_exception(self, mock_generate_content):
        # Test whether the function correctly handles the TimeoutException
        try:
            get_analysis("Test prompt")
        except TimeoutException as e:
            self.fail(f"get_analysis raised {e} unexpectedly!")


class TestFileUtils(unittest.TestCase):

    def test_file_does_exist(self):
        # Testing if the file exists
        with patch("src.utils.os.path.exists", return_value=True), \
             patch("src.utils.os.path.isfile", return_value=True):  # The path is correct
            self.assertTrue(file_does_exist("test_path.jpg"))
        
        # Testing if the file not exists
        with patch("src.utils.os.path.exists", return_value=False), \
             patch("src.utils.os.path.isfile", return_value=False):  # The path is correct
            self.assertFalse(file_does_exist("test_path.jpg"))

    def test_is_valid_image(self):
        # Testing the image is valid
        with patch("PIL.Image.open", return_value=MagicMock(spec=Image.Image)):
            self.assertTrue(is_valid_image("valid_image.jpg"))
        # Testing the image is invalid
        with patch("PIL.Image.open", side_effect=OSError):
            self.assertFalse(is_valid_image("invalid_image.txt"))

