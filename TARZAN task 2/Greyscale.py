import cv2
import numpy as np

# Step 1: Read the  image (following the dimensions: 800x600x3)
image = cv2.imread("photo.jpg")  

 # orignal image display
cv2.imshow("Original ", image) 

cv2.waitKey(0)  # Wait for key press 
cv2.destroyAllWindows()

 # image ko Resize kiya -> 800*600
image = cv2.resize(image, (800, 600)) 
cv2.imshow("Resized", image)

cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()

# Step 2: Convert to grayscale (800x600)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray_image)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()

# Step 3: Reduce resolution (Resizing grayscale image)
low_res_gray_image = cv2.resize(gray_image, (128, 128), interpolation=cv2.INTER_LINEAR)

cv2.imshow("Reduced Resolution Grayscale", low_res_gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Step 4: Normalize the image (not grayscale) to range [0,1]
normalized_image = cv2.cvtColor(low_res_gray_image, cv2.COLOR_GRAY2BGR)
cv2.imshow("Normalized [0-1]", (normalized_image * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Step 5: Optionally normalize to range [-1,1]
normalized_minus1_to_1 = (normalized_image - 0.5) * 2  # Scale to [-1,1]
cv2.imshow("Normalized [-1,1]", normalized_minus1_to_1)

cv2.waitKey(0)
cv2.destroyAllWindows()