import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
I = cv2.imread('CAR2.jpg')
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
rows, cols = Igray.shape

# Dilate Image (using custom 3-pixel horizontal dilation)
Idilate = Igray.copy()
for i in range(rows):
    for j in range(1, cols-1):
        temp = max(Igray[i, j-1], Igray[i, j])
        Idilate[i, j] = max(temp, Igray[i, j+1])

# Show images
cv2.imshow('Original Grayscale', Igray)
cv2.imshow('Dilated Image', Idilate)
cv2.waitKey(0)
cv2.destroyAllWindows()

Iproc = Idilate.copy()

# Horizontal Edge Processing
print("Processing Edges Horizontally...")
horz1 = np.zeros(cols)
max_horz, maximum = 0, 0
total_sum = 0

for i in range(1, cols):
    col_diff_sum = 0
    for j in range(1, rows):
        diff = abs(int(Iproc[j, i]) - int(Iproc[j-1, i]))
        if diff > 20:
            col_diff_sum += diff
    horz1[i] = col_diff_sum
    if col_diff_sum > maximum:
        max_horz = i
        maximum = col_diff_sum
    total_sum += col_diff_sum

average_horz = total_sum / cols

# Smooth horizontal histogram using 41-point moving average
horz = horz1.copy()
for i in range(20, cols-20):
    horz[i] = np.mean(horz1[i-20:i+21])

# Dynamic Threshold filtering on horizontal histogram
for i in range(cols):
    if horz[i] < average_horz:
        horz[i] = 0
        Iproc[:, i] = 0

# Vertical Edge Processing
print("Processing Edges Vertically...")
vert1 = np.zeros(rows)
max_vert, maximum = 0, 0
total_sum = 0

for i in range(1, rows):
    row_diff_sum = 0
    for j in range(1, cols):
        diff = abs(int(Iproc[i, j]) - int(Iproc[i, j-1]))
        if diff > 20:
            row_diff_sum += diff
    vert1[i] = row_diff_sum
    if row_diff_sum > maximum:
        max_vert = i
        maximum = row_diff_sum
    total_sum += row_diff_sum

average_vert = total_sum / rows

# Smooth vertical histogram using 41-point moving average
vert = vert1.copy()
for i in range(20, rows-20):
    vert[i] = np.mean(vert1[i-20:i+21])

# Dynamic Threshold filtering on vertical histogram
for i in range(rows):
    if vert[i] < average_vert:
        vert[i] = 0
        Iproc[i, :] = 0

# Display processed image
cv2.imshow('Processed Image after Thresholding', Iproc)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find probable candidates for Number Plate
column = []
for i in range(1, cols-1):
    if horz[i] != 0 and horz[i-1] == 0 and horz[i+1] == 0:
        column.extend([i, i])
    elif (horz[i] != 0 and horz[i-1] == 0) or (horz[i] != 0 and horz[i+1] == 0):
        column.append(i)

if len(column) % 2 != 0:
    column.append(cols-1)

row = []
for i in range(1, rows-1):
    if vert[i] != 0 and vert[i-1] == 0 and vert[i+1] == 0:
        row.extend([i, i])
    elif (vert[i] != 0 and vert[i-1] == 0) or (vert[i] != 0 and vert[i+1] == 0):
        row.append(i)

if len(row) % 2 != 0:
    row.append(rows-1)

# Region of Interest Extraction
for i in range(0, len(row), 2):
    for j in range(0, len(column), 2):
        if not (max_horz >= column[j] and max_horz <= column[j+1] and
                max_vert >= row[i] and max_vert <= row[i+1]):
            Iproc[row[i]:row[i+1]+1, column[j]:column[j+1]+1] = 0

cv2.imshow('Final Number Plate Detection Output', Iproc)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot histograms
plt.figure(figsize=(12, 10))
plt.subplot(3, 2, 1)
plt.plot(horz1)
plt.title('Horizontal Edge Histogram')

plt.subplot(3, 2, 2)
plt.plot(horz)
plt.title('Smoothed Horizontal Histogram')

plt.subplot(3, 2, 3)
plt.plot(vert1)
plt.title('Vertical Edge Histogram')

plt.subplot(3, 2, 4)
plt.plot(vert)
plt.title('Smoothed Vertical Histogram')

plt.tight_layout()
plt.show()
