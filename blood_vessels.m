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
green_eq = adapthisteq(green_channel); % CLAHE
figure, imshow(green_eq), title('CLAHE Enhanced');

%% Step 3: Apply Median Filtering to Remove Noise
filtered = medfilt2(green_eq, [3 3]);

%% Step 4: Binarization with Adaptive Threshold
bw = imbinarize(filtered, 'adaptive', 'Sensitivity', 0.55);
figure, imshow(bw), title('Adaptive Thresholding');

%% Step 5: Morphological Opening to Remove Small Dots
cleaned = bwareaopen(bw, 100); % remove small <100 px
cleaned = imclose(cleaned, strel('disk',1)); % close gaps
figure, imshow(cleaned), title('Post Morphology');

%% Step 6: Mask with Region of Interest (ROI)
mask = im2bw(green_channel, 0.05);
mask = imfill(mask, 'holes');
segmented = cleaned & mask;

%% Step 7: Display Results
figure;
imshowpair(img, segmented, 'montage');
title('Left: Original | Right: Vessel Segmentation');

%% Step 8: Vessel Pixel Percentage
vessel_area = sum(segmented(:));
total_area = sum(mask(:));
vessel_percentage = 100 * vessel_area / total_area;

fprintf("🩸 Vessel Area (pixels): %d\n", vessel_area);
fprintf("📊 Vessel Area %%: %.2f%%\n", vessel_percentage);
