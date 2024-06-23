# Checkbox-Detection
The Checkbox Detector model is designed to identify and classify checkboxes in PDF documents. It leverages OCR (Optical Character Recognition) and image processing techniques to detect marked and unmarked checkboxes, providing a robust solution for automating the extraction of checkbox data from scanned documents.

## Requirements
### Initializing the Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Installing Pytesseract
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

## Usage
For a quick demonstration of the model, run the following command:
```bash
python model.py
```
but before running the model, make sure to initialize the pdf path in the model.py file.


## Architecture Components
### Initialization (__init__ Method):
##### Config
The model uses a configuration object (config) to manage parameters such as minimum OCR confidence, gray threshold, checkbox size constraints, and directories for saving images.
##### Draw Images
An optional flag (draw_images) that determines whether detected checkboxes should be drawn and saved as images.

### PDF to Image Conversion (pdf_to_image Method):
Converts each page of the PDF document into an image using the pdf2image library. The resulting images are processed to detect checkboxes.

### OCR Box Extraction (_get_ocr_boxes Method):
Uses Tesseract OCR to extract text boxes from the page images. Each text box is defined by its coordinates and confidence level. Only text boxes with a confidence level above the threshold set in the configuration are considered.

### Binary Image Conversion (_get_binary_image Method):
Converts the page image to grayscale and applies binary thresholding. This process highlights the text and checkbox regions, making it easier to detect checkboxes.

### Interval Trees for Efficient Overlap Checks:
Builds interval trees (build_interval_trees) to manage the OCR boxes. These trees help in efficiently checking if a checkbox is surrounded by text boxes, ensuring that only isolated checkboxes are detected.

### Checkbox Detection (_detect_checkboxes_in_image Method):
Identifies potential checkboxes based on their size and position. Uses connected components analysis to find distinct regions in the binary image. Each region is evaluated to determine if it matches the typical size and aspect ratio of a checkbox.

### Checkbox Validation and Classification:
Validates detected regions to classify them as marked or unmarked checkboxes. This classification is based on the pixel density within the checkbox region.

### Drawing and Saving Detected Checkboxes (_draw_and_save_checkboxes Method):
Optionally draws rectangles around detected checkboxes on the original page image and saves the annotated images. Marked checkboxes are highlighted in red, while unmarked checkboxes are highlighted in blue.

### Prediction Method (predict_images):
Processes the entire PDF document and returns the detected checkboxes for each page. This method can also visualize the detection results if draw_images is enabled.

## Configuration Parameters (CFG)
The CFG object includes:

`MIN_OCR_CONFIDENCE`: Minimum confidence level for OCR text boxes to be considered.

`GRAY_THRESHOLD`: Threshold value for converting the image to binary.

`CHECKBOX_MIN_SIZE` and `CHECKBOX_MAX_SIZE`: Size constraints for detecting valid checkboxes.

`save_dir`: Directory path for saving the drawn images.

