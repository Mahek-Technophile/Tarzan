import cv2
import numpy as np
import os

def preprocess_image(image_path, output_dir="output_images"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Load image and resize to 800x600x3 (color image)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (800, 600))
    cv2.imshow("Step 1: 800x600x3", image)
    cv2.imwrite(os.path.join(output_dir, "step1_800x600x3.jpg"), image)
    
    # Step 2: Convert to grayscale (800x600)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Step 2: 800x600 (Grayscale)", gray_image)
    cv2.imwrite(os.path.join(output_dir, "step2_800x600_grayscale.jpg"), gray_image)
    
    # Step 3: Resize to 128x128
    resized_image = cv2.resize(gray_image, (128, 128))
    cv2.imshow("Step 3: 128x128", resized_image)
    cv2.imwrite(os.path.join(output_dir, "step3_128x128.jpg"), resized_image)
    
    # Step 4: Normalize to range [0,1]
    normalized_image = resized_image / 255.0
    print("Step 4: Normalized [0,1]", normalized_image)

    # Step 5: Optional - Normalize to range [-1,1]
    normalized_minus_one_to_one = (normalized_image * 2) - 1
    print("Step 5: Normalized [-1,1]", normalized_minus_one_to_one)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Run the function with an example image
preprocess_image("input.jpg")  # Replace with your actual image path
