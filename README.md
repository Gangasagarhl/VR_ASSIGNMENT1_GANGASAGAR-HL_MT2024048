# VR_ASSIGNMENT1_GANGASAGAR-HL_MT2024048
Setup 

1. Create output folder
2. Create input_images folder
3. Store the mobile captured images in the input_images folder. 

4. Run the following instructions

```bash
#create virtual environment
python3 -m venv assignment

#Set $PATH variable
source assignment/bin/activate

#Install Dependencies
pip install -r requirements.txt

#Run the scripts
python3 coin_detection_segmentation.py input_images/name_of_image
#for testing with the image what you have enter 1 else you enter any other number

python3 panorama.py  list_of_image_paths seperated by gap
```

# Coin Detection and Segmentation using Canny Edge Detection and Hough Transform

## Overview
This project detects and segments coins from an image using **Canny Edge Detection**, **Hough Circle Transform**, and **Watershed Segmentation**. The detected coins are highlighted, and individual coins are segmented for further analysis.

## Steps in the Pipeline

### 1. **Setup and Initialization**
- The required libraries (`cv2`, `matplotlib`, `numpy`, `os`, `sys`) are imported.
- An `output/` directory is created to store processed images.
- The user provides an image path via command line arguments.
- The user selects either **Testing Mode (Manual Parameter Selection)** or **Default Mode (Predefined Parameters).**

### 2. **Loading and Preprocessing the Image**
- The input image is loaded using OpenCV (`cv2.imread`).
- The image is **downsampled** to reduce computational cost.
- It is converted to **grayscale** for better edge detection.
- A **Laplacian filter** is applied to enhance edges.
- The image is sharpened using weighted addition of the original and Laplacian-filtered images.
- A **Gaussian blur** is applied to smoothen the image before edge detection.

### 3. **Canny Edge Detection (Threshold Selection Aid)**
- The blurred image is passed through **Canny Edge Detection** with varying thresholds.
- A grid of Canny outputs is displayed to help select optimal high and low thresholds.
- The selected thresholds are saved and used for further processing.

### 4. **Hough Circle Transform for Coin Detection**
- The **Hough Circle Transform** is applied to detect circular objects (coins) in the image.
- The user can adjust:
  - `minDist`: Minimum distance between detected circles.
  - `threshold`: Sensitivity of circle detection.
  - `minRadius`, `maxRadius`: Expected coin sizes.
- Detected coins are highlighted with **green circles**, and centers are marked in **red**.
- The total number of detected coins is printed.

### 5. **Coin Segmentation**
- Each detected coin is isolated using a **mask**.
- The segmented coins are displayed individually in a grid.
- The segmented images are saved for further analysis.

### 6. **Output Files**
All processed images are saved in the `output/` directory:
- `original_image.png` – The original grayscale image.
- `canny_thresholds.png` – Canny edge detection results with different thresholds.
- `hough_coins_detected.png` – Coins detected using the Hough Transform.
- `segmented_coins.png` – Individually segmented coins.

##### OVERLAP(Option 1)
###### Image Input
<img src="input_images/3.jpg" alt="Image Input">

###### Canny Threshold
<img src="output/canny_thresholds_overlap(option1).png" alt="Canny Threshold">

###### Hough circled output image
<img src="output/hough_coins_detected_overlap.png" alt="Hough circled output image">

###### Segmented Image 
<img src="output/segmented_coins_overlap.png" alt="Segmented Image">


###### NON OVERLAP(Default)

###### Image Input
<img src="input_images/1.jpg" alt="Image Input">

###### Canny Threshold
<img src="output/canny_thresholds.png" alt="Canny Threshold">

###### Hough circled output image
<img src="output/hough_coins_detected.png" alt="Hough circled output image">

###### Segmented Coins
<img src="output/segmented_coins.png" alt="Segmented Coins">

## Usage
### Running the Script
```bash
python coin_detection.py path/to/image.jpg
```

### Modes of Execution
- **Testing Mode (Option 1):** Prompts the user for manual threshold values.
- **Default Mode:** Uses predefined hyperparameters (works well for non-overlapping coins).



# Panorama Stitching

This project performs image stitching to generate a panorama from multiple images(4 images captured using mobile)

## Functions Overview

### Load the Images
#### `load_image_sequence(image_paths)`
<img src="input_images/pan1.jpeg" alt="pan1">
<img src="input_images/pan2.jpeg" alt="pan2">
<img src="input_images/pan3.jpeg" alt="pan3">
<img src="input_images/pan4.jpeg" alt="pan4">

### Convert Images to Grayscale
#### `grayscale_conversion(image)`
Converts images to grayscale for keypoint detection.

### Extract Keypoints & Descriptors
#### `extract_features(image)`
Uses SIFT (Scale-Invariant Feature Transform) to detect keypoints and extract descriptors.

### Match Keypoints Between Images
#### `match_keypoints(descriptors_first, descriptors_second)`
Uses Brute Force Matcher (BFMatcher) with L2 Norm and sorts matches based on distance.

### Compute Homography
#### `compute_homography(keypoints_first, keypoints_second, matches, reprojection_threshold)`
Uses RANSAC to remove outliers and compute the homography matrix for alignment.

### Warp & Blend Images
#### `merge_images(first_image, second_image, homography_matrix)`
Uses the homography matrix to warp the second image onto the first and blends overlapping regions smoothly.

### Stitch Two Images
#### `stitch_two_images(first_image, second_image)`
Calls all previous functions and generates a stitched panorama of two images.

### Save the Final Panorama
#### `create_panorama(image_paths)`
Calls the stitching process and saves the output as `panorama_result.jpg`.

## RESULTS 
### MATCHING VISUALS
<img src="output/matches_visual.jpg" alt="Matching Visuals">

### PANORAMA OUTPUT
<img src="output/panorama_result.jpg" alt="Panorama Output">

