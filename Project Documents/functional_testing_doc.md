# Functional Testing Document

## 1. **Introduction**

### 1.1 Purpose

The purpose of this functional testing is to ensure that the **Multimodal LLM Road Safety Platform** performs as expected and meets the specified functional requirements. This document outlines the tests performed on key platform functionalities, including image upload, distortion application, and AI response generation.

### 1.2 Scope

The functional testing covers the following features of the platform:

- Image upload functionality.
- Application of different distortion effects (Blur, Brightness, Contrast, etc.).
- Integration with the Gemini AI model for analyzing road safety based on uploaded images and user prompts.
- Displaying results and handling errors.

Out of scope:

- Performance testing (e.g., load handling under high user traffic).
- Security testing (e.g., vulnerability scans).

## 2. **Test Environment**

### 2.1 Hardware

- **Processor** : Intel Core i7
- **RAM** : 16 GB
- **Storage** : 512 GB SSD
- **Operating System** : Windows 10

### 2.2 Software

- **Python Version** : 3.10+
- **Streamlit Version** : 1.28.0+
- **Google Generative AI Version** : 0.3.1+
- **Pillow Version** : 10.0.0+
- **NumPy Version** : 1.24.0+
- **SciPy Version** : 1.10.0+
- **Pandas Version** : 2.0.0+

### 2.3 Test Data

- Test images in JPG and PNG format (100x100 pixels, 300x300 pixels).
- Sample prompts:
  1. "Analyze the road safety features visible in this image."
  2. "Evaluate the lighting conditions and their impact on road safety."
  3. "Identify potential hazards for pedestrians in this scene."

## 3. **Test Objectives**

The main objectives of the testing are:

- Ensure images can be uploaded successfully.
- Verify that distortion effects are applied correctly.
- Confirm that the Gemini AI model generates the expected analysis based on the input prompt and uploaded image.
- Ensure error handling mechanisms work as expected (e.g., handling invalid image types).
- Verify the ability to upload multiple image files or specify a folder path for bulk analysis.
- Ensure centralized distortion settings are correctly applied to all images or allow customization for each image.
- Confirm that bulk analysis processes all images and generates a CSV report with the expected analysis results.

## 4. **Test Cases**

### 4.1 Test Case Summary

| Case ID | Description                                            | Preconditions                     | Test Steps                                                                                                          | Expected Result                                                           | Actual Result     | Status |
| ------- | ------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------- | ------ |
| TC001   | Test API key inputting functionality                   | None                              | 1. Open the app.<br />2. Enter a valid API key.<br />3. Submit.                                                     | API key is accepted, and the application allows further actions.          | API key accepted  | Pass   |
| TC002   | Test image upload functionality                        | API key entered. (TC001)          | <br />1. Upload a valid image.<br />2. Submit the form.                                                             | The image is uploaded and displayed correctly in the app.                 | Image displayed   | Pass   |
| TC003   | Test applying "Blur" distortion                        | Image uploaded (TC002)            | 1. Select "Blur" distortion.<br />2. Set intensity to 0.5.<br />3. Submit and view results.                         | The image is blurred correctly according to the intensity setting.        | Effect correct    | Pass   |
| TC004   | Test applying "Brightness" distortion                  | Image uploaded (TC002)            | 1. Select "Brightness".<br />2. Set intensity to 1.2.<br />3. Submit and view results.                              | The image brightness increases as expected.                               | Effect correct    | Pass   |
| TC005   | Test AI model response                                 | Image uploaded, distortion set    | 1. Enter custom prompt.<br />2. Submit for AI analysis.<br />3. View response in the output section.                | The AI generates an accurate analysis of the image, matching the prompt.  | Response accurate | Pass   |
| TC006   | Test bulk analysis by uploading multiple files         | API key entered(TC001)            | 1. Choose Bulk mode.<br />2. Upload multiple valid image files.<br />3. Set centralized distortion.<br />4. Submit. | All images are processed, distortion applied, and results shown for each. | Images processed  | Pass   |
| TC007   | Test bulk analysis by specifying folder path           | API key entered (TC001)           | 1. Choose Bulk mode.<br />2. Upload multiple images.<br />3. Set custom distortion for each image.<br />4. Submit.  | Each image is processed with its respective custom distortion settings.   | Images processed  | Pass   |
| TC008   | Test custom distortion settings for individual images  | API key entered (TC001)           | 1. Choose Bulk mode.<br />2. Upload multiple images.<br />3. Set custom distortion for each image.<br />4. Submit.  | Each image is processed with its respective custom distortion settings.   | Images processed  | Pass   |
| TC009   | Test CSV generation after bulk analysis                | Bulk analysis completed (TC004/5) | 1. Complete bulk image processing.<br />2. Verify if CSV report is generated.<br />3. Download the CSV file.        | A CSV report is generated and can be downloaded successfully.             | CSV generated     | Pass   |
| TC010   | Test invalid folder path handling during bulk analysis | None                              | 1. Choose Bulk mode.<br />2. Enter an invalid folder path.<br />3. Try to proceed with bulk analysis.               | An error message is shown, and bulk analysis is not run.                  | Error handled     | Pass   |
| TC011   | Test invalid image upload                              | None                              | 1. Upload an invalid file (e.g., text file).<br />2. Try submitting.                                                | Error message displayed, file not accepted.                               | Error handled     | Pass   |
| TC012   | Test AI error handling (missing prompt)                | Image uploaded, no prompt         | 1. Upload image.<br />2. Leave the prompt field empty.<br />3. Submit the form.                                     | Warning message shown: "Please provide input text."                       | Warning shown     | Pass   |

## 5. **Test Results Summary**

### 5.1 Summary

In total, 12 test cases were executed. All test cases passed, indicating that the core functionalities of the platform are working as expected.

- **Total Test Cases** : 12
- **Pass** : 12
- **Fail** : 0

### 5.2 Pass/Fail Criteria

- A test case is marked as **Pass** if the actual result matches the expected result.
- A test case is marked as **Fail** if the actual result does not match the expected result or the function breaks during testing.

## 6. **Issues and Bugs**

During functional testing, a minor issue was observed with the distortion sliders on the Streamlit interface:

- **Issue** : When adjusting distortion intensity via the slider, the slider occasionally bounced back to the previous value, rather than staying at the newly selected value.
- **Steps to Reproduce** :1. Select any distortion effect (e.g., "Blur" or "Brightness").

1. Adjust the intensity using the slider.
2. Sometimes the slider resets to its previous value after being adjusted.

- **Expected Behavior** : The slider should retain the adjusted value after being moved.
- **Impact** : This issue can affect the user experience when setting precise distortion levels during analysis. However, it does not impact the final output of the distortion once the value is correctly set.
- **Possible Cause** : The issue is believed to be related to Streamlit's network running process, which may not be able to keep up with real-time slider adjustments, causing a delay or reset of the slider's state.

**Resolution** : No immediate resolution has been implemented. Further investigation into Streamlit's state management and network latency is recommended.

## 7. **Conclusion**

The functional testing for the **Multimodal LLM Road Safety Platform** was successfully completed. All critical functionalities, including image upload, distortion effects, and AI response generation, worked as expected. Error handling mechanisms for invalid inputs were also confirmed to be functioning correctly. The platform is ready for further testing or deployment.
