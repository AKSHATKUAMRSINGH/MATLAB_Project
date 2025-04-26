import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Image
image = cv2.imread('crack_image.jpg')
if image is None:
    raise ValueError("Image not found. Please check the file path.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Use Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Step 4: Morphological Operations to Connect Cracks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 5: Find Contours (Optional - to detect crack boundaries)
contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the original image
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 1)

# Step 6: Display the Results
titles = ['Original Image', 'Grayscale', 'Canny Edges', 'Dilated Edges', 'Final Crack Detection']
images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), gray, edges, dilated, cv2.cvtColor(output, cv2.COLOR_BGR2RGB)]

plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    if i in [1, 2, 3]:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 7: Print Total Number of Crack Contours Found
print(f"Total number of crack regions detected: {len(contours)}")
