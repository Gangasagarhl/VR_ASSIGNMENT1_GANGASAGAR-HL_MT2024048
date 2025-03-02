import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure that the "output" directory exists so that all generated images can be saved.
os.makedirs("output", exist_ok=True)

def canny_check(image):
    """
    Display and save the original grayscale image and various edge-detected versions using different Canny thresholds.
    
    What it does:
    - Plots the original image.
    - Applies the Canny edge detector with 12 different threshold ranges to help the tester
      choose the best thresholds for detecting edges.
    - Saves both the original image and the collection of Canny results for further analysis.
    
    Why it does it:
    - Visualizing how different thresholds affect the edge detection helps in tuning
      hyperparameters for subsequent circle detection.
    """
    # Display and save the original image.
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig("output/original_image.png")
    plt.show()

    # Create a grid of subplots to display multiple Canny edge results.
    fig, ax = plt.subplots(2, 6, figsize=(20, 8))
    # Loop through 12 different threshold settings.
    for i in range(12):
        # Compute Canny edges with thresholds ranging from (i*25) to (i*25 + 50)
        edges_img = cv2.Canny(image, i * 25, i * 25 + 50)
        # Determine the subplot row and column for the current iteration.
        row, col = divmod(i, 6)
        ax[row, col].imshow(edges_img, cmap='gray')
        ax[row, col].set_title(f'Canny {i * 25}-{i * 25 + 50}')
        ax[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("output/canny_thresholds.png")
    plt.show()

def detect_with_Hough(gray_image, color_image, canny_high, minDist=None, threshold=30, minRadius=None, maxRadius=None):
    """
    Detect circular objects (coins) in the grayscale image using the Hough Circle Transform.
    
    What it does:
    - Sets default hyperparameters if not provided.
    - Uses cv2.HoughCircles to detect circles based on edges in the grayscale image.
    - Draws the detected circles onto a copy of the color image.
    - Returns the annotated image, the number of circles detected, and a list of circles.
    
    Why it does it:
    - The Hough Circle Transform is a robust method for detecting circles, which is useful 
      for tasks such as coin detection in images.
    """
    # Set default values if not provided.
    if minDist is None:
        minDist = gray_image.shape[0] // 8
    if minRadius is None:
        minRadius = gray_image.shape[0] // 15
    if maxRadius is None:
        maxRadius = gray_image.shape[0] // 8
    if threshold is None:
        threshold = 30

    # Apply the Hough Circle Transform to detect circles.
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, minDist,
                               param1=canny_high, param2=threshold,
                               minRadius=minRadius, maxRadius=maxRadius)
    
    # Make a copy of the original color image to draw circles on.
    output = color_image.copy()
    detected_circles = []
    if circles is not None:
        # Round circle parameters and convert to unsigned integers.
        circles = np.uint16(np.around(circles[0]))
        detected_circles = circles.tolist()
        # Draw each detected circle on the image.
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Draw outer circle in green.
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Draw center of the circle in red.
    # Return the annotated image, count of detected circles, and the list of circles.
    return output, len(detected_circles), detected_circles

def segment_coins(color_image, circles):
    """
    Segment individual coins from the color image based on detected circles.
    
    What it does:
    - For each detected coin (circle), it creates a mask that isolates the coin.
    - Applies the mask to the original image to extract the coin region.
    - Displays and saves the segmented coins in a grid layout.
    
    Why it does it:
    - Segmenting each coin allows further analysis or classification of individual coins.
    - Visualizing segmented coins can verify the accuracy of detection.
    """
    # If no circles were detected, exit the function.
    if not circles:
        return
    
    plt.figure(figsize=(12, 8))
    num_coins = len(circles)
    cols = 4  # Define number of columns in the grid display.
    rows = (num_coins + cols - 1) // cols  # Calculate required number of rows.
    
    # Iterate over each detected circle.
    for i, (x, y, r) in enumerate(circles):
        # Create an empty mask with the same shape as the original color image.
        mask = np.zeros_like(color_image)
        # Draw a filled circle (mask) where the coin is located.
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        
        # Apply the mask to the original image to isolate the coin.
        segmented = cv2.bitwise_and(color_image, mask)
        
        # Convert the segmented coin image to RGB for proper display in matplotlib.
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        
        # Plot each segmented coin.
        plt.subplot(rows, cols, i+1)
        plt.imshow(segmented_rgb)
        plt.axis('off')
        plt.title(f'Coin {i+1}')
    
    plt.tight_layout()
    plt.savefig("output/segmented_coins.png")
    plt.show()


"""
def watershed(path):
    # Read and downsample image
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # Downsample while maintaining aspect ratio
    scale_factor = 529 / max(img.shape[:2])
    color_downsampled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # Convert to grayscale
    gray = cv2.cvtColor(color_downsampled, cv2.COLOR_BGR2GRAY)

    # Improved preprocessing
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Proper sure background/foreground calculation
    sure_bg = cv2.dilate(opening, kernel, iterations=10)  # Fixed from erode to dilate
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)  # Lower threshold

    # Marker refinement
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected components with background handling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # Background becomes 1
    markers[unknown == 255] = 0  # Unknown regions

    # Watershed algorithm
    markers = cv2.watershed(color_downsampled, markers)

    # Count unique regions (excluding background and boundaries)
    unique_markers = np.unique(markers)
    num_coins = len(unique_markers) - 2  # Subtract background (1) and boundaries (-1)

    # Visualization on downsampled image
    color_downsampled[markers == -1] = [0, 255, 0]  # Green boundaries

    # Display results
    print(f"Total coins detected: {num_coins}")
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(color_downsampled, cv2.COLOR_BGR2RGB))
    plt.title('Watershed Segmentation')
    plt.axis('off')
    plt.show()

"""


def coin_detection_and_segmentation(path, option):
    """
    Perform coin detection and segmentation on the input image.
    
    What it does:
    - Reads an image from the provided path.
    - Optionally allows the user to input hyperparameters (in Testing mode) or uses defaults.
    - Downsamples the image to a manageable size.
    - Converts the image to grayscale.
    - Applies preprocessing steps (Laplacian for sharpening, Gaussian blur) to enhance features.
    - Displays Canny edge outputs for threshold selection.
    - Uses the Hough Circle Transform to detect coins.
    - Displays and saves the detection result.
    - If coins are detected, segments each coin and displays the segmented images.
    
    Why it does it:
    - Downsampling helps reduce computation and noise.
    - Preprocessing enhances the edges and features required for accurate circle detection.
    - Interactive hyperparameter tuning (in Testing mode) allows adaptation to different image conditions.
    - Visual feedback is provided at multiple steps to facilitate parameter adjustment and verification of results.
    """
    # Read image and set hyperparameters based on user option.
    if option == 1:
        # In testing mode, allow the user to specify the downsample size.
        downsample_size = int(input("Enter downsample size(default 529): ") or 529)
        img = cv2.imread(path)
    else:
        # In default mode, use preset hyperparameters.
        downsample_size = 529  # This size works well for many images.
        img = cv2.imread(path)
        
    # Downsample the image using pyrDown repeatedly until it reaches the desired size.
    color_downsampled = img.copy()
    while color_downsampled.shape[0] > downsample_size or color_downsampled.shape[1] > downsample_size:
        color_downsampled = cv2.pyrDown(color_downsampled)

    # Convert the downsampled color image to grayscale.
    gray = cv2.cvtColor(color_downsampled, cv2.COLOR_BGR2GRAY)

    # Preprocessing to enhance image features:
    # Apply the Laplacian operator to detect edges.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    # Sharpen the image by combining the original and the Laplacian (edge information).
    sharpened = cv2.addWeighted(gray.astype(np.float64), 1.7, laplacian, -0.7, 0)
    # Clip values to valid range and convert to uint8.
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    # Apply Gaussian blur to reduce noise before edge detection.
    blur = cv2.GaussianBlur(sharpened, (5, 5), 0)

    # Show various Canny edge detection outputs to help select thresholds.
    canny_check(blur)
    
    # Set hyperparameters for circle detection. If option==1, prompt the user for values.
    if option == 1:
        low = int(input("\nCanny low threshold: "))
        high = int(input("\nCanny high threshold: "))
        minDist = int(input("\nMinimum distance between coins(-1 entered will take default measure): ") or -1)
        if minDist == -1: 
            minDist = blur.shape[0] // 8
        
        threshold_val = int(input("\nIf the image has many coins and too much overlapping, increase this threshold (default):") or -1)
        if threshold_val == -1: 
            threshold_val = 30
        
        minRadius = int(input("\nMinimum radius (-1 for default):  ") or -1)
        if minRadius == -1:
            minRadius = blur.shape[0] // 20
        
        maxRadius = int(input("\nMaximum radius (-1 for default):  ") or -1)
        if maxRadius == -1:
            maxRadius = blur.shape[0] // 5
    else:
        # Default hyperparameters that work well for non-overlapping coin images.
        low = 250
        high = 300
        minDist = blur.shape[0] // 8
        threshold_val = 30
        minRadius = blur.shape[0] // 20
        maxRadius = blur.shape[0] // 5

    # Detect coins using the Hough Circle Transform.
    result_img, num_coins, circles = detect_with_Hough(blur, color_downsampled, high, 
                                                         minDist, threshold_val, minRadius, maxRadius)

    # Convert the detection result image from BGR to RGB for proper display in matplotlib.
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Display and save the coin detection results.
    plt.figure(figsize=(10, 6))
    plt.imshow(result_img_rgb)
    plt.title(f'Detected Coins: {num_coins}')
    plt.axis('off')
    plt.savefig("output/hough_coins_detected.png")
    plt.show()
    
    print("Total coins (HOUGH):", num_coins)

    # If coins are detected, segment them and display the segmented results.
    if num_coins > 0:
        segment_coins(color_downsampled, circles)
    
    # Additional processing (e.g., watershed segmentation) could be added here if needed.

# Main entry point for the script.
# This section processes command-line arguments to get the image path and then calls the coin detection and segmentation function.
if __name__ == "__main__":
    # Get the first command-line argument as the image path.
    path = sys.argv[1:][0]

    # Ask the user to choose between Testing mode (manual hyperparameter input) or Default mode.
    option = int(input("Enter 1 for Testing: You must provide all the required parameters for proper functionality.\n\nEnter any other integer for Default mode: Hyperparameters are set for images in the input folder and work well for non-overlapped images: \n"))
    print("\nPath of the image:", path)
    coin_detection_and_segmentation(path, option=option)
