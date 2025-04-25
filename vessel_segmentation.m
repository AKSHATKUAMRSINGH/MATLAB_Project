clc;
clear;
close all;

%% Step 1: Load Fundus Image
img = imread('fundus_image.jpg');
figure, imshow(img), title('Original Fundus Image');

% Extract green channel - vessels show best contrast
green_channel = img(:,:,2);
figure, imshow(green_channel), title('Green Channel');

%% Step 2: Enhance Contrast
% Contrast Limited Adaptive Histogram Equalization (CLAHE)
green_eq = adapthisteq(green_channel);
figure, imshow(green_eq), title('Contrast Enhanced (CLAHE)');

%% Step 3: Apply Median Filtering to Remove Noise
filtered = medfilt2(green_eq, [3 3]);
figure, imshow(filtered), title('After Median Filtering');

%% Step 4: Adaptive Thresholding
% Convert image to binary using adaptive thresholding
bw = imbinarize(filtered, 'adaptive', 'Sensitivity', 0.55);
figure, imshow(bw), title('Adaptive Thresholding');

%% Step 5: Morphological Cleaning
% Remove small objects (noise)
cleaned = bwareaopen(bw, 100); % Remove regions with <100 pixels
% Close gaps within vessel segments
cleaned = imclose(cleaned, strel('disk', 1));
figure, imshow(cleaned), title('After Morphological Cleaning');

%% Step 6: Mask with Region of Interest (ROI)
% Create mask to exclude dark background
mask = imbinarize(green_channel, 0.05);
mask = imfill(mask, 'holes'); % Fill holes within mask
figure, imshow(mask), title('ROI Mask');

% Apply mask to segmented vessels
segmented = cleaned & mask;
figure, imshow(segmented), title('Final Vessel Segmentation');

%% Step 7: Display Results Together
figure;
imshowpair(img, segmented, 'montage');
title('Left: Original Fundus Image | Right: Vessel Segmentation');

%% Step 8: Vessel Pixel Percentage Calculation
vessel_area = sum(segmented(:));
total_area = sum(mask(:));
vessel_percentage = 100 * vessel_area / total_area;

fprintf("🩸 Vessel Area (pixels): %d\n", vessel_area);
fprintf("📊 Vessel Area Percentage: %.2f%%\n", vessel_percentage);
