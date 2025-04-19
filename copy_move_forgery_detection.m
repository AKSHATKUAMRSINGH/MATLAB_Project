clc;
clear;
close all;

%% Step 1: Load and Preprocess Image
img = imread('original_image.jpg');
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

img_gray = im2double(img_gray);
img_resized = imresize(img_gray, [256 256]); % Normalize size

block_size = 8;
[rows, cols] = size(img_resized);
num_blocks = (rows - block_size + 1) * (cols - block_size + 1);

fprintf("🔍 Total Blocks to Process: %d\n", num_blocks);

%% Step 2: Generate Overlapping Blocks + Extract DCT Features
features = zeros(num_blocks, 64); % Each block's DCT is 8x8 = 64
positions = zeros(num_blocks, 2); % Store top-left position of each block

k = 1;
for i = 1:(rows - block_size + 1)
    for j = 1:(cols - block_size + 1)
        block = img_resized(i:i+block_size-1, j:j+block_size-1);
        dct_block = dct2(block);
        features(k, :) = reshape(dct_block, 1, []); % Flatten to 1x64
        positions(k, :) = [i, j];
        k = k + 1;
    end
end

% Keep only first few coefficients (top-left) — best energy compaction
features = features(:, 1:16); % Top 16 DCT coefficients for each block

%% Step 3: Lexicographic Sorting
[sorted_features, idx] = sortrows(features);
sorted_positions = positions(idx, :);

%% Step 4: Similarity Check — Match Neighboring Feature Vectors
threshold = 0.1;     % DCT similarity threshold (lower = stricter)
distance_thresh = 5; % Minimum Euclidean distance between matching blocks

matched = [];

for i = 1:(num_blocks - 1)
    diff = norm(sorted_features(i,:) - sorted_features(i+1,:));
    if diff < threshold
        pos1 = sorted_positions(i,:);
        pos2 = sorted_positions(i+1,:);
        dist = norm(pos1 - pos2);
        if dist > distance_thresh
            matched = [matched; pos1 pos2];
        end
    end
end

fprintf("✅ Matches Found: %d\n", size(matched, 1));

%% Step 5: Mark Matched Regions
detection_map = zeros(rows, cols);
for i = 1:size(matched, 1)
    x1 = matched(i, 1);
    y1 = matched(i, 2);
    x2 = matched(i, 3);
    y2 = matched(i, 4);

    detection_map(x1:x1+block_size-1, y1:y1+block_size-1) = 1;
    detection_map(x2:x2+block_size-1, y2:y2+block_size-1) = 1;
end

%% Step 6: Display Results
figure;
subplot(1,2,1);
imshow(img_gray);
title('Original Image');

subplot(1,2,2);
imshow(img_gray);
hold on;
h = imshow(cat(3, ones(size(detection_map)), zeros(size(detection_map)), zeros(size(detection_map))));
set(h, 'AlphaData', detection_map * 0.5);
title('Detected Forged Regions');
