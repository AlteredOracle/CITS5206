import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from app import generate_content_timeout, get_analysis, file_does_exist, is_valid_image, TimeoutException, main
from google.api_core import exceptions as google_exceptions

class TestGenerateContentTimeout(unittest.TestCase):

    @patch("app.genai.GenerativeModel")
    def test_generate_content_success(self, mock_gen_model):
        # Mock the generative model
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance
        mock_model_instance.generate_content.return_value = MagicMock(text="Generated content")

        # Call the function and verify the return result
        result = generate_content_timeout(mock_model_instance, "Test prompt")
        mock_model_instance.generate_content.assert_called_once_with("Test prompt")
        self.assertEqual(result.text, "Generated content")

    @patch("app.genai.GenerativeModel")  # Mock AI model generation
    def test_generate_content_no_timeout(self, mock_gen_model):
        # Mock the AI model instance
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance

        # Simulate generate_content throwing a TimeoutException
        mock_model_instance.generate_content.side_effect = TimeoutException

        # Call the function and verify that a TimeoutException is thrown
        with self.assertRaises(TimeoutException):
            generate_content_timeout(mock_model_instance, "Test prompt")

    @patch("app.genai.GenerativeModel")
    def test_generate_content_args_passing(self, mock_gen_model):
        """
        Test whether the generate_content_timeout function correctly passes the prompt and image arguments.
        """
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance

        # Define test prompt and image
        prompt = "Test prompt"
        image = MagicMock()  # Mock image object

        # Call the function and pass in the prompt and image
        generate_content_timeout(mock_model_instance, prompt, image)

        # Verify that the model method received the correct arguments
        mock_model_instance.generate_content.assert_called_once_with([prompt, image])


class TestGetAnalysis(unittest.TestCase):

    @patch("app.generate_content_timeout", side_effect=google_exceptions.InvalidArgument("Invalid"))
    def test_get_analysis_invalid_argument(self, mock_generate_content):
        # Test whether the function correctly handles the InvalidArgument exception
        try:
            get_analysis("Test prompt")
        except google_exceptions.InvalidArgument as e:
            self.fail(f"get_analysis raised {e} unexpectedly!")

    @patch("app.generate_content_timeout", side_effect=TimeoutException)
    def test_get_analysis_timeout_exception(self, mock_generate_content):
        # Test whether the function correctly handles the TimeoutException
        try:
            get_analysis("Test prompt")
        except TimeoutException as e:
            self.fail(f"get_analysis raised {e} unexpectedly!")

    @patch("app.generate_content_timeout")
    @patch("PIL.Image.open")
    @patch("app.genai.GenerativeModel")  # Mock model instance
    def test_get_analysis_args_passing(self, mock_gen_model, mock_image_open, mock_generate_content):
        """
        Test whether get_analysis correctly passes model, prompt, and image arguments to generate_content_timeout.
        """
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image

        # Mock model instance
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance

        prompt = "Test prompt"
        image_path = "test_image.jpg"

        # Call get_analysis function
        get_analysis(prompt, image_path)

        # Verify the parameters received by generate_content_timeout
        mock_generate_content.assert_called_once_with(mock_model_instance, prompt, mock_image)


class TestFileUtils(unittest.TestCase):

    def test_file_does_exist(self):
        # Testing if the file exists
        with patch("utils.os.path.exists", return_value=True), \
             patch("utils.os.path.isfile", return_value=True):  # The path is correct
            self.assertTrue(file_does_exist("test_path.jpg"))
        
        # Testing if the file not exists
        with patch("utils.os.path.exists", return_value=False), \
             patch("utils.os.path.isfile", return_value=False):  # The path is not correct
            self.assertFalse(file_does_exist("test_path.jpg"))

    def test_is_valid_image(self):
        # Testing the image is valid
        with patch("PIL.Image.open", return_value=MagicMock(spec=Image.Image)):
            self.assertTrue(is_valid_image("valid_image.jpg"))
        # Testing the image is invalid
        with patch("PIL.Image.open", side_effect=OSError):
            self.assertFalse(is_valid_image("invalid_image.txt"))

class TestMainFunction(unittest.TestCase):

    @patch("app.genai.configure")  # Mock API key configuration
    @patch("app.os.getenv", return_value="dummy_api_key")  # Simulate setting GEMINI_API_KEY
    @patch("app.file_does_exist", return_value=True)  # Simulate the file exists
    @patch("app.is_valid_image", return_value=True)  # Simulate the image is valid
    @patch("PIL.Image.open", return_value=MagicMock())  # Simulate opening the image
    @patch("app.genai.GenerativeModel")  # Simulate the API model
    @patch("app.generate_content_timeout", return_value=MagicMock(text="Generated content"))  # Ensure content is generated
    @patch("builtins.input", side_effect=["Test prompt", "valid_image.jpg"])  # Simulate input
    def test_main_success(self, mock_input, mock_gen_content_timeout, mock_gen_model, mock_image_open, mock_is_valid_image, mock_file_exist, mock_getenv, mock_genai_configure):
        with patch('builtins.print') as mocked_print:
            main()
            mocked_print.assert_any_call("AI Analysis Result:")
            mocked_print.assert_any_call("Generated content")

    
  


