% Crack Detection in Pavements using MATLAB
clc;
clear;
close all;

% Step 1: Load the Image
img = imread('pavement.jpg');
figure, imshow(img), title('Original Pavement Image');

% Step 2: Convert to Grayscale
gray = rgb2gray(img);
figure, imshow(gray), title('Grayscale Image');

% Step 3: Apply Median Filtering to Reduce Noise
gray_filtered = medfilt2(gray, [3 3]);
figure, imshow(gray_filtered), title('Denoised Image');

% Step 4: Contrast Enhancement (Optional)
gray_enhanced = imadjust(gray_filtered);
figure, imshow(gray_enhanced), title('Contrast Enhanced');

% Step 5: Edge Detection using Canny
edges = edge(gray_enhanced, 'canny');
figure, imshow(edges), title('Edge Detected (Canny)');

% Step 6: Morphological Operations
se = strel('line', 3, 90); % Structuring element
dilated = imdilate(edges, se);
figure, imshow(dilated), title('Dilated Edges');

% Step 7: Fill Small Holes
filled = imfill(dilated, 'holes');
figure, imshow(filled), title('Filled Cracks');

% Step 8: Remove Small Objects (Noise)
clean = bwareaopen(filled, 100); % Remove objects smaller than 100 pixels
figure, imshow(clean), title('Cleaned Cracks');

% Step 9: Overlay Cracks on Original Image
overlay = img;
overlay(:,:,1) = uint8(clean) * 255 + overlay(:,:,1); % Highlight cracks in red
figure, imshow(overlay), title('Cracks Overlay on Original Image');

% Step 10: Feature Extraction (Optional)
stats = regionprops(clean, 'Area', 'BoundingBox');
num_cracks = length(stats);
fprintf('Total number of detected cracks: %d\n', num_cracks);

% Draw Bounding Boxes
imshow(img), title('Detected Cracks with Bounding Boxes');
hold on;
for k = 1:num_cracks
    thisBB = stats(k).BoundingBox;
    rectangle('Position', thisBB, 'EdgeColor', 'r', 'LineWidth', 2);
end
hold off;
