


# RESIZER LARGE TIF FILES

import cv2
import rasterio
import numpy as np
from typing import Callable, List, Optional, Tuple, Union
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S') 

class Concate(object):
    def __init__(self, scale_factor: float = 0.5, 
        chunk_size: tuple = (512, 512), 
        dist_edje: float = 0.5, 
        result_path: str = './img/result.tif'):

        self.scale_factor = scale_factor
        self.chunk_size = chunk_size
        self.dist_edje = dist_edje
        self.result_path = result_path


    
    def read_tiff(self, tiff_path: str, chank_flag: bool = False, window: rasterio.windows.Window = None) -> np.ndarray:
        """ Function to read a large TIFF image using Rasterio """
        with rasterio.open(tiff_path) as src:
            if chank_flag is True:
                img = src.read([1, 2, 3, 4], window=window, out_shape=(src.count, window.height, window.width))
            else:
                img = src.read([1, 2, 3, 4])  # Read RGBA bands
            img = np.moveaxis(img, 0, -1)  # Move channels to the last axis for OpenCV compatibility
        return img



    def resize_tiff_image_with_alpha(self, input_path: str, output_path: str) -> int:
        """ Function to resize the image with proper handling of the alpha channel """
        # Step 1: Read the large TIFF image with an alpha channel
        img = self.read_tiff(input_path, chank_flag = False)

        # Separate the RGB and Alpha channels
        rgb_img = img[..., :3]  # Extract RGB
        alpha_channel = img[..., 3]  # Extract Alpha channel

        # Step 2: Calculate new dimensions
        height, width = rgb_img.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)

        # Step 3: Resize the RGB and Alpha channel separately using INTER_AREA interpolation
        resized_rgb = cv2.resize(rgb_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_alpha = cv2.resize(alpha_channel, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Step 4: Merge the resized RGB and Alpha channel back into a 4-channel image (RGBA)
        resized_img = np.dstack((resized_rgb, resized_alpha))

        # Step 5: Save the resized image back as a TIFF with alpha channel
        with rasterio.open(input_path) as src:
            meta = src.meta.copy()
            meta.update({
                'width': new_width,
                'height': new_height,
                'count': 4,  # Ensure the metadata knows it's 4 bands (RGBA)
                'dtype': 'uint8'  # Ensure correct data type
            })

        # Move channels back to match Rasterio's (bands, height, width) format
        resized_img = np.moveaxis(resized_img, -1, 0)

        # Step 6: Write the resized image to a new TIFF file
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(resized_img)

        #print(f"Resized TIFF with alpha saved to {output_path}")
        return 0


    # Function to merge two chunks using SIFT and affine transformation
    def merge_chunks_sift_affine(self, chunk1: np.ndarray, chunk2: np.ndarray) -> np.ndarray:
        # Convert chunks to grayscale
        gray1 = cv2.cvtColor(chunk1, cv2.COLOR_BGRA2GRAY)  # Ensure correct conversion for RGBA
        gray2 = cv2.cvtColor(chunk2, cv2.COLOR_BGRA2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Check if descriptors are None or insufficient
        if descriptors1 is None or descriptors2 is None or len(descriptors1) < 2 or len(descriptors2) < 2:
            logging.error(f"Not enough descriptors for matching. Returning original chunk.")
            return chunk1  # Return original chunk if not enough descriptors are found

        # Convert descriptors to 32-bit floating point (CV_32F)
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)

        # Match descriptors using FLANN matcher
        index_params = dict(algorithm=1, trees=5)  # FLANN parameters
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        except cv2.error as e:
            #print(f"Error in FLANN matching: {e}. Returning original chunk.")
            logging.error(f"Error in FLANN matching: {e}. Returning original chunk.")
            return chunk1  # Return original chunk if matching fails

        # Filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < self.dist_edje * n.distance:
                good_matches.append(m)

        #print(len(good_matches))
        logging.info(f"Good matches quantity: {len(good_matches)}")

        # If not enough good matches, return the first chunk
        if len(good_matches) < 3:  # Minimum points needed for affine transformation is 3
            #print("Not enough good matches. Returning original chunk.")
            logging.error(f"Not enough good matches. Returning original chunk.")
            return chunk1  # Return the original chunk if no good matches are found

        # Extract point locations from the good matches
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute affine transformation
        affine_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Check if affine transformation is valid
        if affine_matrix is None or affine_matrix.shape != (2, 3):
            #print("Affine transformation computation failed. Returning original chunk.")
            logging.error(f"Affine transformation computation failed. Returning original chunk.")
            return chunk1

        # Warp the first chunk to align with the second using affine transformation
        height, width = gray2.shape
        warped_chunk1 = cv2.warpAffine(chunk1, affine_matrix, (width, height))

        # Create a mask for the warped chunk (assuming the 4th channel is alpha)
        mask1 = (warped_chunk1[:, :, 3] > 0).astype(np.uint8) 

        # Combine the two chunks using the mask
        merged_chunk = np.where(mask1[..., np.newaxis], warped_chunk1, chunk2)

        return merged_chunk

    # Function to merge multiple large TIFFs by processing chunks
    def merge_large_tiffs_by_chunks(self, tiff_paths: list, chunk_size: tuple = None) -> int:
        if chunk_size is None:
            chunk_size = self.chunk_size
        # Open the first TIFF to get the general metadata
        with rasterio.open(tiff_paths[0]) as src:
            output_height = src.height
            output_width = src.width
            output_meta = src.meta.copy()

        # Prepare output metadata
        output_meta.update({
            'driver': 'GTiff',
            'height': output_height,
            'width': output_width,
            'count': 4,  # Assuming we are working with RGBA (4 bands)
            'dtype': 'uint8',
        })

        # Create the output TIFF file
        with rasterio.open(self.result_path, 'w', **output_meta) as dest:
            # Process chunks
            for i in range(0, output_height, chunk_size[0]):
                for j in range(0, output_width, chunk_size[1]):
                    # Calculate the actual chunk size, adjusting at the edges
                    win_height = min(chunk_size[0], output_height - i)
                    win_width = min(chunk_size[1], output_width - j)

                    # Start with the first chunk as the base
                    base_chunk = None

                    # Merge chunks from all TIFF files
                    for tiff_path in tiff_paths:
                        with rasterio.open(tiff_path) as src:
                            window = rasterio.windows.Window(j, i, win_width, win_height)
                            chunk = self.read_tiff(tiff_path = tiff_path, window = window, chank_flag = True)

                            if base_chunk is None:
                                base_chunk = chunk
                            else:
                                # Resize the chunk if dimensions are different
                                if base_chunk.shape != chunk.shape:
                                    # Resize chunk to match base_chunk
                                    chunk = cv2.resize(chunk, (base_chunk.shape[1], base_chunk.shape[0]))

                                base_chunk = self.merge_chunks_sift_affine(base_chunk, chunk)
                    
                    # Write the merged chunk to the output TIFF
                    base_chunk = np.moveaxis(base_chunk, -1, 0)  # Move channels back to the first axis (C, H, W)
                    dest.write(base_chunk, window = window)

        #print("Merged TIFF saved as './img/merged_output_second_3_6.tif'")
        logging.info(f"Merged TIFF saved as {self.result_path}")
        return 0   
    

'''

# MERGE 2 IMAGES

import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt



# Function to read large TIFF images using rasterio and convert to OpenCV format
def read_tiff_as_cv2(tiff_path):
    with rasterio.open(tiff_path) as src:
        img = src.read([1, 2, 3, 4])  # Read all bands (channels)
        img = np.moveaxis(img, 0, -1)  # Move channels to last axis for easier processing (H, W, C)
    return img

# Convert TIFF images to grayscale for SIFT
def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # For RGB TIFFs
    else:
        return image  # If already grayscale

# Function to merge two images using SIFT and homography
def merge_images_sift(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints and descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Use FLANN based matcher for matching keypoints
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Number of checks
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image 1 to image 2's perspective
    height, width = img2.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, H, (width, height))

    # Merge the two images by taking the maximum pixel value (to handle overlapping areas)
    merged_image = np.maximum(warped_img1, img2)

    return merged_image

# Example usage
tiff1_path = './img/resized_3.tif'
tiff2_path = './img/resized_3_1.tif'

# Step 1: Load TIFF images
img1 = read_tiff_as_cv2(tiff1_path)
img2 = read_tiff_as_cv2(tiff2_path)

# Step 2: Convert to grayscale for SIFT
#img1_gray = convert_to_grayscale(img1)
#img2_gray = convert_to_grayscale(img2)

# Step 3: Merge the images using SIFT and homography
merged_image = merge_images_sift(img1, img2)

# Step 4: Save or visualize the merged image
plt.figure(figsize=(12, 8))
plt.imshow(merged_image)
plt.title('Merged Image')
plt.show()

# Optionally save the merged image using rasterio
# You can adjust this based on the number of bands and metadata you want to preserve
with rasterio.open('./img/merged_image.tif', 'w', driver='GTiff', height=merged_image.shape[0], width=merged_image.shape[1], count=1, dtype=merged_image.dtype) as dest:
    dest.write(merged_image, 1)

'''

'''
# MERGE SEVERAL TIF

import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Function to read large TIFF images using rasterio and convert to OpenCV format
def read_tiff_as_cv2(tiff_path):
    with rasterio.open(tiff_path) as src:
        img = src.read()  # Read all bands (channels)
        img = np.moveaxis(img, 0, -1)  # Move channels to last axis for easier processing (H, W, C)
    return img

# Convert TIFF images to grayscale for SIFT
def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # For RGB TIFFs
    else:
        return image  # If already grayscale

# Function to merge two images using SIFT and homography
def merge_images_sift(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints and descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Use FLANN based matcher for matching keypoints
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Number of checks
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image 1 to image 2's perspective
    height, width = img2.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, H, (width, height))

    # Merge the two images by taking the maximum pixel value (to handle overlapping areas)
    merged_image = np.maximum(warped_img1, img2)

    return merged_image

# Function to merge multiple TIFF files
def merge_multiple_tiffs(tiff_paths):
    # Load the first image
    base_img = read_tiff_as_cv2(tiff_paths[0])
    base_img_gray = convert_to_grayscale(base_img)

    # Iterate over the rest of the TIFF files and merge them one by one
    for tiff_path in tiff_paths[1:]:
        next_img = read_tiff_as_cv2(tiff_path)
        next_img_gray = convert_to_grayscale(next_img)

        # Merge the current base image with the next image
        base_img = merge_images_sift(base_img_gray, next_img_gray)
        base_img_gray = convert_to_grayscale(base_img)  # Convert the merged image to grayscale for further merging

    return base_img

# Example usage for merging several TIFF files
tiff_paths = [
    './img/resized_3.tif',
    './img/resized_3_1.tif',
    './img/resized_4.tif',
    './img/resized_2.tif'
]

# Step 1: Merge all TIFF images using SIFT and homography
merged_image = merge_multiple_tiffs(tiff_paths)

# Step 2: Visualize the merged image
plt.figure(figsize=(12, 8))
plt.imshow(merged_image, cmap='gray')
plt.title('Merged Image')
plt.show()

# Step 3: Save the final merged image
with rasterio.open('merged_multiple_images.tif', 'w', driver='GTiff', height=merged_image.shape[0], width=merged_image.shape[1], count=1, dtype=merged_image.dtype) as dest:
    dest.write(merged_image, 1)

print("Merged multiple TIFF images saved as 'merged_multiple_images.tif'")

'''

'''

# COMPARE WITH SIFT 2 NOT LARGE TIF

import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Function to read large TIFF images using rasterio and convert to OpenCV format
def read_tiff_as_cv2(tiff_path):
    with rasterio.open(tiff_path) as src:
        img = src.read()  # Read all bands (channels)
        img = np.moveaxis(img, 0, -1)  # Move channels to last axis for easier processing (H, W, C)
    return img

# Convert TIFF images to grayscale for SIFT
def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # For RGB TIFFs
    else:
        return image  # If already grayscale

# Function to compare two images using SIFT and plot matches
def compare_images_sift(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints and descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Use FLANN based matcher for matching keypoints
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Number of checks

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches, good_matches

# Example usage
tiff1_path = './img/resized_3.tif'
tiff2_path = './img/resized_3_1.tif'

# Step 1: Load TIFF images
img1 = read_tiff_as_cv2(tiff1_path)
img2 = read_tiff_as_cv2(tiff2_path)

# Step 2: Convert to grayscale for SIFT
img1_gray = convert_to_grayscale(img1)
img2_gray = convert_to_grayscale(img2)

# Step 3: Compare using SIFT and find matches
img_matches, good_matches = compare_images_sift(img1_gray, img2_gray)

# Step 4: Plot the matches using matplotlib
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title(f"Matched Keypoints: {len(good_matches)}")
plt.show()


'''
'''

# MERGE VIA CHUNKS WITH HOMOGRAPHY

import cv2
import rasterio
import numpy as np

# Function to read a chunk (window) from a TIFF file
def read_tiff_chunk(tiff_path, window):
    with rasterio.open(tiff_path) as src:
        img_chunk = src.read(window=window, out_shape=(src.count, window.height, window.width))
        img_chunk = np.moveaxis(img_chunk, 0, -1)  # Move channels to the last axis (H, W, C)
    return img_chunk

# Function to merge two chunks using SIFT and homography
def merge_chunks_sift(chunk1, chunk2):
    # Convert chunks to grayscale
    gray1 = cv2.cvtColor(chunk1, cv2.COLOR_BGRA2GRAY)  # Ensure correct conversion for RGBA
    gray2 = cv2.cvtColor(chunk2, cv2.COLOR_BGRA2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Check if descriptors are None or insufficient
    if descriptors1 is None or descriptors2 is None or len(descriptors1) < 2 or len(descriptors2) < 2:
        print("Not enough descriptors for matching. Returning original chunk.")
        return chunk1  # Return original chunk if not enough descriptors are found

    # Convert descriptors to 32-bit floating point (CV_32F)
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # Match descriptors using FLANN matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN parameters
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error as e:
        print(f"Error in FLANN matching: {e}. Returning original chunk.")
        return chunk1  # Return original chunk if matching fails

    # Filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    print(len(good_matches))

    # If not enough good matches, return the first chunk
    if len(good_matches) < 4:
        print("Not enough good matches. Returning original chunk.")
        return chunk1  # Return the original chunk if no good matches are found

    # Extract point locations from the good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


    # Check if H is valid
    if H is None or H.shape != (3, 3):
        print("Homography computation failed. Returning original chunk.")
        return chunk1

    # Warp the first chunk to align with the second
    height, width = gray2.shape
    warped_chunk1 = cv2.warpPerspective(chunk1, H, (width, height))

    # Create a mask for the warped chunk (assuming the 4th channel is alpha)
    mask1 = (warped_chunk1[:, :, 3] > 0).astype(np.uint8) 

    # Combine the two chunks using the mask
    merged_chunk = np.where(mask1[..., np.newaxis], warped_chunk1, chunk2)

    return merged_chunk

# Function to merge multiple large TIFFs by processing chunks
def merge_large_tiffs_by_chunks(tiff_paths, chunk_size=(512, 512)):
    # Open the first TIFF to get the general metadata
    with rasterio.open(tiff_paths[0]) as src:
        output_height = src.height
        output_width = src.width
        output_meta = src.meta.copy()

    # Prepare output metadata
    output_meta.update({
        'driver': 'GTiff',
        'height': output_height,
        'width': output_width,
        'count': 4,  # Assuming we are working with RGBA (4 bands)
        'dtype': 'uint8',
    })

    # Create the output TIFF file
    with rasterio.open('./img/merged_output_second.tif', 'w', **output_meta) as dest:
        # Process chunks
        for i in range(0, output_height, chunk_size[0]):
            for j in range(0, output_width, chunk_size[1]):
                # Calculate the actual chunk size, adjusting at the edges
                win_height = min(chunk_size[0], output_height - i)
                win_width = min(chunk_size[1], output_width - j)

                # Start with the first chunk as the base
                base_chunk = None

                # Merge chunks from all TIFF files
                for tiff_path in tiff_paths:
                    with rasterio.open(tiff_path) as src:
                        window = rasterio.windows.Window(j, i, win_width, win_height)
                        chunk = read_tiff_chunk(tiff_path, window)

                        if base_chunk is None:
                            base_chunk = chunk
                        else:
                            # Resize the chunk if dimensions are different
                            if base_chunk.shape != chunk.shape:
                                # Resize chunk to match base_chunk
                                chunk = cv2.resize(chunk, (base_chunk.shape[1], base_chunk.shape[0]))

                            base_chunk = merge_chunks_sift(base_chunk, chunk)

                # Write the merged chunk to the output TIFF
                base_chunk = np.moveaxis(base_chunk, -1, 0)  # Move channels back to the first axis (C, H, W)
                dest.write(base_chunk, window=window)

    print("Merged TIFF saved as './img/merged_output_second.tif'")

# Example usage:
tiff_paths = ['./img/merged_output.tif', './img/resized_3.tif']
merge_large_tiffs_by_chunks(tiff_paths, chunk_size=(6000, 6000)) #4096


'''














