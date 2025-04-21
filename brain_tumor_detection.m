clc;
clear;
close all;

%% Step 1: Read and Preprocess the Image
img = imread('brain_mri.jpg');
figure, imshow(img), title('Original MRI Image');

% Convert to grayscale if RGB
if size(img, 3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end

gray = im2double(gray);

%% Step 2: K-Means Clustering for Segmentation
% Reshape image to 1D array
nRows = size(gray, 1);
nCols = size(gray, 2);
img_vector = reshape(gray, nRows * nCols, 1);

% Apply K-means clustering (2 or 3 clusters)
num_clusters = 3;
[cluster_idx, cluster_centers] = kmeans(img_vector, num_clusters, 'MaxIter', 200);

% Reshape back to image
clustered_image = reshape(cluster_idx, nRows, nCols);
figure, imagesc(clustered_image), title('Clustered Image (Label Map)');
colormap(jet); colorbar;

%% Step 3: Extract the Cluster with Tumor (Highest Intensity)
% Use the cluster with the highest intensity as tumor
tumor_cluster = find(cluster_centers == max(cluster_centers));
tumor_mask = clustered_image == tumor_cluster;

figure, imshow(tumor_mask), title('Initial Tumor Mask');

%% Step 4: Morphological Cleaning
tumor_cleaned = imfill(tumor_mask, 'holes');              % Fill holes
tumor_cleaned = bwareaopen(tumor_cleaned, 30);            % Remove small blobs
tumor_cleaned = imdilate(tumor_cleaned, strel('disk',5)); % Optional dilation

figure, imshow(tumor_cleaned), title('Cleaned Tumor Mask');

%% Step 5: Overlay on Original Image
tumor_overlay = imoverlay(gray, tumor_cleaned, [1 0 0]); % red overlay

figure, imshow(tumor_overlay), title('Tumor Detected in MRI');

%% Step 6: Calculate Tumor Area
tumor_area_pixels = sum(tumor_cleaned(:));
total_pixels = nRows * nCols;
tumor_percentage = 100 * tumor_area_pixels / total_pixels;

fprintf('🧠 Tumor Area (Pixels): %d\n', tumor_area_pixels);
fprintf('📊 Tumor Area Percentage: %.2f%%\n', tumor_percentage);
