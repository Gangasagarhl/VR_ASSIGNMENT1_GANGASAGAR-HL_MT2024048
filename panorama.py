import numpy as np
import cv2
import sys
import os

class ImageStitcher:
    """A class to stitch multiple images into a panorama."""
    
    def __init__(self, first_image, second_image):
        # Print initialization message
        print("\nInitializing image stitcher\n")
        # Get the width (number of columns) of the first image
        width_first = first_image.shape[1]
        # Get the width of the second image
        width_second = second_image.shape[1]
        # Determine the minimum width to ensure the overlapping region is defined consistently
        min_width = min(width_first, width_second)
        
        # Define a blend ratio for the overlapping region between images.
        # A ratio that is too high could cause a loss of detail, too low might result in abrupt transitions.
        blend_ratio = 0.10  # Blend ratio is 10% of the smaller image's width
        # Calculate the overlapping blend region size:
        # - It is at least 100 pixels for smooth blending.
        # - It is capped at 1000 pixels to prevent heavy loss of information.
        self.blend_overlap_size = max(100, min(int(blend_ratio * min_width), 1000))

    @staticmethod
    def grayscale_conversion(image):
        """Convert the image to grayscale."""
        print("\nConverting image to grayscale\n")
        # Convert the image to grayscale using OpenCV.
        # Grayscale images simplify feature detection while retaining essential structure.
        return image, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def extract_features(image):
        """Detect keypoints and descriptors using SIFT."""
        print("\nExtracting keypoints and descriptors using SIFT\n")
        # Create a SIFT detector object. SIFT identifies scale-invariant keypoints.
        sift = cv2.SIFT_create()
        # Detect keypoints and compute the corresponding descriptors.
        # Keypoints are the distinct points in the image (corners, blobs) and descriptors are their unique fingerprints.
        return sift.detectAndCompute(image, None)

    @staticmethod
    def match_keypoints(descriptors_first, descriptors_second):
        """Match keypoints between two images."""
        print("\nMatching keypoints\n")
        # Create a brute-force matcher that compares descriptors using the Euclidean (L2) distance.
        # crossCheck=True ensures that the match is consistent in both directions.
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match the descriptors from the first image to those in the second image.
        matches = matcher.match(descriptors_first, descriptors_second)
        # Sort the matches based on distance; lower distance indicates a better match.
        sorted_matches = sorted(matches, key=lambda m: m.distance)
        return sorted_matches

    @staticmethod
    def compute_homography(keypoints_first, keypoints_second, matches, reprojection_threshold):
        """Compute the homography matrix."""
        print("\nComputing homography\n")
        # Ensure there are enough matches to compute a homography (minimum 4 points required).
        if len(matches) >= 4:
            # Extract the matching keypoint coordinates from the first image.
            src_points = np.float32([keypoints_first[m.queryIdx].pt for m in matches])
            # Extract the matching keypoint coordinates from the second image.
            dst_points = np.float32([keypoints_second[m.trainIdx].pt for m in matches])
            # Compute the homography matrix using RANSAC to robustly filter out outliers.
            homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, reprojection_threshold)
            return matches, homography_matrix, mask
        
        # If not enough matches are found, print an error message.
        print("Not enough matches to compute homography")
        return None

    def create_blend_mask(self, first_image, second_image, side):
        """Create a blend mask for seamless stitching."""
        print(f"\nCreating blend mask for {side} side\n")
        # Get the height of the image (assumes both images have the same height)
        height = first_image.shape[0]
        # Get the widths of the first and second images
        width_first = first_image.shape[1]
        width_second = second_image.shape[1]
        # Calculate the total width for the panorama by summing the widths of both images
        total_width = width_first + width_second
        # Initialize a blank (zero-valued) mask for the panorama
        mask = np.zeros((height, total_width), dtype=np.float32)
        # Define an offset as half of the blend overlap size
        offset = int(self.blend_overlap_size / 2)
        # Set the boundary where blending will occur in the first image
        blend_boundary = width_first - offset
        
        if side == "left":
            # For the left side:
            # Set all pixels before the blend boundary to full contribution (1)
            mask[:, :blend_boundary] = 1
            # In the overlapping region, create a linear gradient that transitions from 1 to 0 for smooth blending
            mask[:, blend_boundary - offset:blend_boundary + offset] = np.tile(np.linspace(1, 0, 2 * offset), (height, 1))
        else:  # side == "right"
            # For the right side:
            # Create a gradient in the overlapping region that transitions from 0 to 1
            mask[:, blend_boundary - offset:blend_boundary + offset] = np.tile(np.linspace(0, 1, 2 * offset), (height, 1))
            # Set all pixels after the overlapping region to full contribution (1)
            mask[:, blend_boundary + offset:] = 1
        # Merge the single channel mask into a 3-channel mask so it can be applied to an RGB image
        return cv2.merge([mask, mask, mask])

    def merge_images(self, first_image, second_image, homography_matrix):
        """Blend two images using the homography matrix."""
        print("\nMerging images\n")
        # Get the height of the images (assumes both images have the same height)
        height = first_image.shape[0]
        # Get the widths of the first and second images
        width_first = first_image.shape[1]
        width_second = second_image.shape[1]
        # Calculate the total width of the final panorama
        panorama_width = width_first + width_second

        # Convert the first image to float32 to enable smooth blending (avoiding integer overflow)
        first_image_float = first_image.astype(np.float32)
        # Create a blend mask for the left side (corresponding to the first image)
        full_left_mask = self.create_blend_mask(first_image, second_image, "left")
        # Extract the portion of the mask corresponding only to the first image
        left_mask = full_left_mask[:, :width_first, :]
        # Multiply the first image with its mask to get its weighted contribution
        left_contribution = first_image_float * left_mask

        # Initialize an empty panorama image array
        panorama = np.zeros((height, panorama_width, 3), dtype=np.float32)
        # Place the left contribution into the panorama image
        panorama[:, :width_first, :] = left_contribution

        # Warp the second image using the provided homography matrix so that it aligns with the first image
        warped_second = cv2.warpPerspective(second_image, homography_matrix, (panorama_width, height))
        # Convert the warped second image to float32 for blending
        warped_second_float = warped_second.astype(np.float32)
        # Create a blend mask for the right side (corresponding to the second image)
        right_mask = self.create_blend_mask(first_image, second_image, "right")
        # Multiply the warped second image with its mask to get its weighted contribution
        right_contribution = warped_second_float * right_mask

        # Add the contributions from both images to create a blended panorama
        blended_panorama = panorama + right_contribution

        # Identify non-zero regions in the panorama (where there is actual image data)
        rows, cols = np.where(blended_panorama[:, :, 0] != 0)
        # Crop the panorama to the bounding box of the non-zero region to remove any black borders
        if rows.size > 0 and cols.size > 0:
            cropped_panorama = blended_panorama[min(rows):max(rows) + 1, min(cols):max(cols) + 1, :]
        else:
            cropped_panorama = blended_panorama

        # Clip pixel values to the valid range [0, 255] and convert back to unsigned 8-bit integers
        final_panorama = np.clip(cropped_panorama, 0, 255).astype(np.uint8)
        return final_panorama

    @staticmethod
    def stitch_two_images(first_image, second_image):
        """Stitch two images together."""
        print("\nStitching two images\n")
        # Create an instance of the ImageStitcher class using the two images
        stitcher = ImageStitcher(first_image, second_image)
        # Convert both images to grayscale for feature detection
        _, first_gray = stitcher.grayscale_conversion(first_image)
        _, second_gray = stitcher.grayscale_conversion(second_image)
        # Extract keypoints and descriptors from the first grayscale image
        keypoints_first, descriptors_first = stitcher.extract_features(first_gray)
        # Extract keypoints and descriptors from the second grayscale image
        keypoints_second, descriptors_second = stitcher.extract_features(second_gray)
        # Match keypoints between the two images
        matches = stitcher.match_keypoints(descriptors_first, descriptors_second)

        # Visualize the matches (drawing the first 100 matches)
        matches_visualization = cv2.drawMatches(
            first_image, keypoints_first,
            second_image, keypoints_second,
            matches[:100], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Compute the homography to align the two images based on the matched keypoints.
        homography_result = stitcher.compute_homography(keypoints_first, keypoints_second, matches, reprojection_threshold=4)
        if homography_result is None:
            # If homography computation fails, return an error message
            return "Error: Cannot stitch images", None
        # Unpack the computed homography matrix (ignoring the match list and mask)
        _, homography_matrix, _ = homography_result

        # Use the inverse of the computed homography matrix to warp the second image into the first image's frame
        panorama = stitcher.merge_images(first_image, second_image, np.linalg.inv(homography_matrix))

        # Return the final stitched panorama along with the match visualization image
        return panorama, matches_visualization

    @staticmethod
    def stitch_multiple_images(image_list, num_images):
        """Recursively stitch multiple images."""
        print("\nStitching multiple images\n")
        if num_images == 2:
            # Base case: if there are only two images, stitch them directly
            return ImageStitcher.stitch_two_images(image_list[0], image_list[1])
        # Recursively stitch the last two images in the list
        stitched_image, _ = ImageStitcher.stitch_two_images(image_list[num_images-2], image_list[num_images-1])
        # Replace the second-to-last image with the stitched result to reduce the image count
        image_list[num_images-2] = stitched_image
        # Recursively process the list until only one stitched image remains
        return ImageStitcher.stitch_multiple_images(image_list[:-1], num_images-1)

    @staticmethod
    def load_image_sequence(image_paths):
        """Load images from the given paths."""
        print("\nLoading images\n")
        images = []
        # Loop through each provided image file path
        for path in image_paths:
            # Read the image file from disk
            img = cv2.imread(path)
            if img is None:
                # Print a warning if the image cannot be loaded
                print(f"Warning: Could not load image at {path}")
                continue
            # Convert the image from BGR (default in OpenCV) to RGB color space for consistency
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Append the converted image to the list
            images.append(img_rgb)
        # Return the list of images and the total count of images loaded
        return images, len(images)

def create_panorama(image_paths):
    """Main function to create a panorama from a list of images."""
    print("\nStarting panorama creation process\n")
    # Load the sequence of images from the given file paths
    images, num_images = ImageStitcher.load_image_sequence(image_paths)
    if num_images < 2:
        # If fewer than two images are provided, print an error message and exit
        print("Need at least two images to create a panorama.")
        return
    # Stitch the images together recursively to create a single panoramic image
    panorama_result, matches_visual = ImageStitcher.stitch_multiple_images(images, num_images)
    # Define the output directory to save the resulting images
    output_dir = "output"
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Write the final panorama image to disk
    cv2.imwrite(os.path.join(output_dir, "panorama_result.jpg"), panorama_result)
    # Write the visualization of the matched keypoints to disk
    cv2.imwrite(os.path.join(output_dir, "matches_visual.jpg"), matches_visual)
    # Indicate completion of the panorama creation process
    print("\nPanorama creation process completed\n")

if __name__ == "__main__":
    # Execute the panorama creation using image paths provided via command-line arguments
    create_panorama(sys.argv[1:])
