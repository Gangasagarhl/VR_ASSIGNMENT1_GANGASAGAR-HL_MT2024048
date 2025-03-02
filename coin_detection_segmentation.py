import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

def canny_check(image):
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig("output/original_image.png")  # Save original image
    plt.show()

    fig, ax = plt.subplots(2, 6, figsize=(20, 8))
    for i in range(12):
        edges_img = cv2.Canny(image, i * 25, i * 25 + 50)
        row, col = divmod(i, 6)
        ax[row, col].imshow(edges_img, cmap='gray')
        ax[row, col].set_title(f'Canny {i * 25}-{i * 25 + 50}')
        ax[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("output/canny_thresholds.png")  # Save Canny thresholds
    plt.show()

def detect_with_Hough(gray_image, color_image, canny_high, minDist=None, threshold=30, minRadius=None, maxRadius=None):
    if minDist is None:
        minDist = gray_image.shape[0] // 8
    if minRadius is None:
        minRadius = gray_image.shape[0] // 15
    if maxRadius is None:
        maxRadius = gray_image.shape[0] // 8
    if threshold is None:
        threshold = 30

    # Detect circles
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, minDist,
                               param1=canny_high, param2=threshold,
                               minRadius=minRadius, maxRadius=maxRadius)
    
    output = color_image.copy()
    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        detected_circles = circles.tolist()
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
    return output, len(detected_circles), detected_circles

def segment_coins(color_image, circles):
    if not circles:
        return
    
    plt.figure(figsize=(12, 8))
    num_coins = len(circles)
    cols = 4  # Number of columns in the display grid
    rows = (num_coins + cols - 1) // cols
    
    for i, (x, y, r) in enumerate(circles):
        # Create mask for the coin
        mask = np.zeros_like(color_image)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        
        # Apply mask to original image
        segmented = cv2.bitwise_and(color_image, mask)
        
        # Convert to RGB for proper matplotlib display
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        
        # Plot each segmented coin
        plt.subplot(rows, cols, i+1)
        plt.imshow(segmented_rgb)
        plt.axis('off')
        plt.title(f'Coin {i+1}')
    
    plt.tight_layout()
    plt.savefig("output/segmented_coins.png")  # Save segmented coins
    plt.show()


def coin_detection_and_segmentation(path,option):
    # Hyperparameters
    if option==1:
        downsample_size = int(input("Enter downsample size(default 529): ") or 529)  
        img = cv2.imread(path)
    
    else:
        downsample_size = 529  # 23 * 23
        img = cv2.imread(path)
        
        
    # Downsample the color image
    color_downsampled = img.copy()
    while color_downsampled.shape[0] > downsample_size or color_downsampled.shape[1] > downsample_size:
        color_downsampled = cv2.pyrDown(color_downsampled)

    # Convert to grayscale
    gray = cv2.cvtColor(color_downsampled, cv2.COLOR_BGR2GRAY)

    # Preprocessing
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    sharpened = cv2.addWeighted(gray.astype(np.float64), 1.7, laplacian, -0.7, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(sharpened, (5, 5), 0)

    # Show Canny edges for threshold selection
    canny_check(blur)
    
    # Hyperparameters - set by tester
    if option==1:
        low = int(input("\nCanny low threshold: "))
        high = int(input("\nCanny high threshold: "))
        minDist = int(input("\nMinimum distance between coins(-1 entered will take default measure): ") or -1)
        if minDist ==  -1: 
            minDist = blur.shape[0] // 8
        
        
        threshold_val = int(input("\nIf the image has many coins and too muuch overlapping incrase this threshold it works(default):") or -1)
        if threshold_val ==-1: 
            threshold_val = 30
        
            
        minRadius = int(input("\nMinimum radius(-1 for default):  ") or -1)
        if minRadius ==-1:
            minRadius = blur.shape[0] // 20
        
        maxRadius = int(input("\nMaximum radius(-1 for default):  ") or -1)
        if maxRadius==-1:
            maxRadius = blur.shape[0] // 5
        
        
        
    
    else:
            
        low = 250
        high = 300
        minDist = blur.shape[0] // 8
        threshold_val = 30
        minRadius = blur.shape[0] // 20
        maxRadius = blur.shape[0] // 5

    # Hough circle detection
    result_img, num_coins, circles = detect_with_Hough(blur, color_downsampled, high, 
                                                     minDist, threshold_val, minRadius, maxRadius)

    # Convert BGR to RGB for display
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Show detection results
    plt.figure(figsize=(10, 6))
    plt.imshow(result_img_rgb)
    plt.title(f'Detected Coins: {num_coins}')
    plt.axis('off')
    plt.savefig("output/hough_coins_detected.png")  # Save detected coins
    plt.show()
    
    print("Total coins(HOUGH):", num_coins)

    # Show segmented coins
    if num_coins > 0:
        #print("\nSegmented Coins:")
        segment_coins(color_downsampled, circles)

    
    #calling water shed from here,
   


# Replace 'input_images/1.jpg' with your image path

if __name__ == "__main__":
    path=  sys.argv[1:][0]

    option =  int(input("Enter 1 for Testing: You must provide all the required parameters,  for proper functionality.\n\nEnter any other integer for Default mode: Hyperparameters are set for image in input folder and works well for non oerlapped images: \n"))
    print("\npath of the image ", path)
    coin_detection_and_segmentation(path, option=option)