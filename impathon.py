import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('img2.jpg', 0)

def nothing(x):
    pass

# Create a window
cv2.namedWindow('Filtered Image')

# Create trackbars for low and high frequency cutoff
cv2.createTrackbar('LowPass', 'Filtered Image', 1, 50, nothing)
cv2.createTrackbar('HighPass', 'Filtered Image', 1, 50, nothing)

while True:
    # Get current trackbar positions
    low = cv2.getTrackbarPos('LowPass', 'Filtered Image')
    high = cv2.getTrackbarPos('HighPass', 'Filtered Image')

    # Fourier Transform
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2

    # Create masks for low and high pass
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-low:crow+low, ccol-low:ccol+low] = 1  # Low Pass area

    mask2 = np.ones((rows, cols, 2), np.uint8)
    mask2[crow-high:crow+high, ccol-high:ccol+high] = 0  # High Pass area

    # Apply masks
    fshift_low = dft_shift * mask
    fshift_high = dft_shift * mask2

    # Combine both to eliminate both frequencies
    fshift_combined = fshift_low + fshift_high

    # Inverse DFT
    f_ishift = np.fft.ifftshift(fshift_combined)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    # Normalize result to display properly
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    # Show result
    cv2.imshow('Filtered Image', img_back)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
