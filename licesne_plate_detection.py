import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import os

# Set tesseract executable path (IMPORTANT)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image
image = cv2.imread('CAR2.jpg')
if image is None:
    print("Image not found. Please check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = gray.shape

# Apply horizontal max filter (dilation in 1D across columns)
dilated = gray.copy()
for i in range(rows):
    for j in range(1, cols-1):
        dilated[i, j] = max(gray[i, j-1], gray[i, j], gray[i, j+1])

# Show original and dilated images
cv2.imshow('Original Image', image)
cv2.imshow('Dilated Image', dilated)

# Horizontal edge detection
horz_diff = np.zeros(cols)
for i in range(1, cols):
    sum_diff = 0
    for j in range(1, rows):
        diff = abs(int(dilated[j, i]) - int(dilated[j-1, i]))
        if diff > 20:
            sum_diff += diff
    horz_diff[i] = sum_diff

# Apply low pass filter
horz_filtered = np.convolve(horz_diff, np.ones(41)/41, mode='same')
horz_avg = np.mean(horz_filtered)

# Zero out columns below average
for i in range(cols):
    if horz_filtered[i] < horz_avg:
        dilated[:, i] = 0

# Vertical edge detection
vert_diff = np.zeros(rows)
for i in range(1, rows):
    sum_diff = 0
    for j in range(1, cols):
        diff = abs(int(dilated[i, j]) - int(dilated[i, j-1]))
        if diff > 20:
            sum_diff += diff
    vert_diff[i] = sum_diff

# Apply low pass filter
vert_filtered = np.convolve(vert_diff, np.ones(41)/41, mode='same')
vert_avg = np.mean(vert_filtered)

# Zero out rows below average
for i in range(rows):
    if vert_filtered[i] < vert_avg:
        dilated[i, :] = 0

# Display filtered result
cv2.imshow('Filtered Image', dilated)

# Find contours
contours, _ = cv2.findContours(cv2.Canny(dilated, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a directory to save extracted plates
if not os.path.exists('plates'):
    os.makedirs('plates')

plate_count = 0

# Draw bounding boxes and extract possible plate areas
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    if 2 < aspect_ratio < 6 and w > 100 and h > 20:
        plate = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow(f'Detected Plate {plate_count+1}', plate)

        # OCR to extract text from the detected plate
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(plate_thresh, config='--psm 7')
            cleaned_text = text.strip()
            print(f"Detected Number Plate {plate_count+1}: {cleaned_text}")

            # Save extracted plate image
            cv2.imwrite(f'plates/plate_{plate_count+1}.png', plate)

            # Optionally, save text to a file
            with open("detected_plates.txt", "a") as f:
                f.write(f"Plate {plate_count+1}: {cleaned_text}\n")

            plate_count += 1

        except Exception as e:
            print(f"OCR error: {e}")

# Show final result
cv2.imshow('Final Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
