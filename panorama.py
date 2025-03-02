import numpy as np
import cv2
import sys
import os

class ImageStitcher:
    """A class to stitch multiple images into a panorama."""

    def __init__(self, first_image, second_image):
        print("\nInitializing image stitcher\n")
        width_first = first_image.shape[1]
        width_second = second_image.shape[1]
        min_width = min(width_first, width_second)
        
        """
        The blends or transtions to be be made smoothly
        pixel size is more than 1000, there is heavy loss of information
        if values are less than 50px, then  there were very sharp edges, poor transtions, when mwegerd
        """
        blend_ratio = 0.10  # Blend ratio for overlapping region
        self.blend_overlap_size = max(100, min(int(blend_ratio * min_width), 1000))

    @staticmethod
    def grayscale_conversion(image):
        """Convert the image to grayscale."""
        print("\nConverting image to grayscale\n")
        return image, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def extract_features(image):
        """Detect keypoints and descriptors using SIFT."""
        print("\nExtracting keypoints and descriptors using SIFT\n")
        
        sift = cv2.SIFT_create()
        """
        cv2.SIFT_create()
        Used to find the keypoints and  and compute the descriptors
        SIFT is independet of scale and rotation and illumination
        
        
        sift.detectAndCompute(image, None)
        detect the keypoints in an image and create the descriptors for each and every key point
        
        This will be called for every image that is passed through the function
        """
        return sift.detectAndCompute(image, None)

    @staticmethod
    def match_keypoints(descriptors_first, descriptors_second):
        """Match keypoints between two images."""
        print("\nMatching keypoints\n")
        
        
        """
        BFMatcher- 
        The brute force matcher, this is used for chnecking the match between the 2 different decsiptors
        ueses euclidean distance
        cross check from both the ends
        
        
        sorting: 
        1. The difference between the features are kept from lower to higher distance so that
        It helps the RANSAC  to choose the best feature among all. 
        Enhances th epaorama quality
        """
        
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors_first, descriptors_second)
        sorted_matches = sorted(matches, key=lambda m: m.distance)
        return sorted_matches



    @staticmethod
    def compute_homography(keypoints_first, keypoints_second, matches, reprojection_threshold):
        """Compute the homography matrix."""
        print("\nComputing homography\n")
        
        """
        aligns the images properly in panorama sticihing
        so it must need minimum of 4 points to make the match
        Extract the match points and align them properly.'
        
        uses the transformation matrix and make the RANSAC to remove the outliers
        """
        if len(matches) >= 4:
            src_points = np.float32([keypoints_first[m.queryIdx].pt for m in matches])
            dst_points = np.float32([keypoints_second[m.trainIdx].pt for m in matches])
            homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, reprojection_threshold)
            return matches, homography_matrix, mask
        
        
        """
        Mask specifis the inlier vd outlier
        """
        print("Not enough matches to compute homography")
        return None



    def create_blend_mask(self, first_image, second_image, side):
        """Create a blend mask for seamless stitching."""
        print(f"\nCreating blend mask for {side} side\n")
        height = first_image.shape[0]
        width_first = first_image.shape[1]
        width_second = second_image.shape[1]
        total_width = width_first + width_second
        mask = np.zeros((height, total_width), dtype=np.float32)
        offset = int(self.blend_overlap_size / 2)
        blend_boundary = width_first - offset
        if side == "left":
            mask[:, :blend_boundary] = 1
            mask[:, blend_boundary - offset:blend_boundary + offset] = np.tile(np.linspace(1, 0, 2 * offset), (height, 1))
        else:  # side == "right"
            mask[:, blend_boundary - offset:blend_boundary + offset] = np.tile(np.linspace(0, 1, 2 * offset), (height, 1))
            mask[:, blend_boundary + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def merge_images(self, first_image, second_image, homography_matrix):
        """Blend two images using the homography matrix."""
        print("\nMerging images\n")
        height = first_image.shape[0]
        width_first = first_image.shape[1]
        width_second = second_image.shape[1]
        panorama_width = width_first + width_second

        # Process first image contribution
        first_image_float = first_image.astype(np.float32)
        full_left_mask = self.create_blend_mask(first_image, second_image, "left")
        left_mask = full_left_mask[:, :width_first, :]
        left_contribution = first_image_float * left_mask

        # Prepare panorama: insert first image contribution
        panorama = np.zeros((height, panorama_width, 3), dtype=np.float32)
        panorama[:, :width_first, :] = left_contribution

        # Process second image contribution
        warped_second = cv2.warpPerspective(second_image, homography_matrix, (panorama_width, height))
        warped_second_float = warped_second.astype(np.float32)
        right_mask = self.create_blend_mask(first_image, second_image, "right")
        right_contribution = warped_second_float * right_mask

        blended_panorama = panorama + right_contribution

        # Crop to the non-zero region
        rows, cols = np.where(blended_panorama[:, :, 0] != 0)
        if rows.size > 0 and cols.size > 0:
            cropped_panorama = blended_panorama[min(rows):max(rows) + 1, min(cols):max(cols) + 1, :]
        else:
            cropped_panorama = blended_panorama

        final_panorama = np.clip(cropped_panorama, 0, 255).astype(np.uint8)
        return final_panorama

    @staticmethod
    def stitch_two_images(first_image, second_image):
        """Stitch two images together."""
        print("\nStitching two images\n")
        stitcher = ImageStitcher(first_image, second_image)
        _, first_gray = stitcher.grayscale_conversion(first_image)
        _, second_gray = stitcher.grayscale_conversion(second_image)
        keypoints_first, descriptors_first = stitcher.extract_features(first_gray)
        keypoints_second, descriptors_second = stitcher.extract_features(second_gray)
        matches = stitcher.match_keypoints(descriptors_first, descriptors_second)

        # Visualize matches
        matches_visualization = cv2.drawMatches(
            first_image, keypoints_first,
            second_image, keypoints_second,
            matches[:100], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        homography_result = stitcher.compute_homography(keypoints_first, keypoints_second, matches, reprojection_threshold=4)
        if homography_result is None:
            return "Error: Cannot stitch images", None
        _, homography_matrix, _ = homography_result

        # Use inverse homography for warping
        panorama = stitcher.merge_images(first_image, second_image, np.linalg.inv(homography_matrix))

        return panorama, matches_visualization

    @staticmethod
    def stitch_multiple_images(image_list, num_images):
        """Recursively stitch multiple images."""
        print("\nStitching multiple images\n")
        if num_images == 2:
            return ImageStitcher.stitch_two_images(image_list[0], image_list[1])
        stitched_image, _ = ImageStitcher.stitch_two_images(image_list[num_images-2], image_list[num_images-1])
        image_list[num_images-2] = stitched_image
        return ImageStitcher.stitch_multiple_images(image_list[:-1], num_images-1)

    @staticmethod
    def load_image_sequence(image_paths):
        """Load images from the given paths."""
        print("\nLoading images\n")
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load image at {path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
        return images, len(images)


def create_panorama(image_paths):
    """Main function to create a panorama from a list of images."""
    print("\nStarting panorama creation process\n")
    images, num_images = ImageStitcher.load_image_sequence(image_paths)
    if num_images < 2:
        print("Need at least two images to create a panorama.")
        return
    panorama_result, matches_visual = ImageStitcher.stitch_multiple_images(images, num_images)
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, "panorama_result.jpg"), panorama_result)
    cv2.imwrite(os.path.join(output_dir, "matches_visual.jpg"), matches_visual)
    print("\nPanorama creation process completed\n")


if __name__ == "__main__":
    create_panorama(sys.argv[1:])