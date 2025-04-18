clc;
clear;
close all;

%% Step 1: Load Test Image
img = imread('test/test_leaf.jpg');
figure, imshow(img), title('Original Leaf');

%% Step 2: Resize and Preprocess
img = imresize(img, [256 256]);
lab_img = rgb2lab(img); % Convert to L*a*b color space
ab = double(lab_img(:,:,2:3));
ab = reshape(ab, [], 2);

%% Step 3: K-means Segmentation (K=3)
nColors = 3;
[cluster_idx, cluster_center] = kmeans(ab, nColors, 'distance', 'sqEuclidean', ...
    'Replicates', 3);

% Label each pixel
pixel_labels = reshape(cluster_idx, size(img,1), size(img,2));
figure, imshow(label2rgb(pixel_labels)), title('K-means Segmentation');

% Extract region with most "disease" based on darker or irregular color
segmented_images = cell(1, nColors);
rgb_label = repmat(pixel_labels, [1 1 3]);
for k = 1:nColors
    color = img;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

% Choose cluster manually or by average color intensity (here auto)
mean_vals = zeros(1, nColors);
for k = 1:nColors
    cluster_img = segmented_images{k};
    gray_cluster = rgb2gray(cluster_img);
    mean_vals(k) = mean(gray_cluster(:));
end
[~, disease_cluster] = min(mean_vals); % dark region is disease
disease_region = segmented_images{disease_cluster};
figure, imshow(disease_region), title('Detected Disease Region');

%% Step 4: Feature Extraction
gray = rgb2gray(disease_region);
bw = imbinarize(gray);
bw = bwareaopen(bw, 500); % Remove small regions
stats = regionprops(bw, 'BoundingBox', 'Area');

% GLCM Texture Features
glcm = graycomatrix(gray, 'Offset', [0 1]);
features = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

% Color Features
r = mean2(disease_region(:,:,1));
g = mean2(disease_region(:,:,2));
b = mean2(disease_region(:,:,3));

feature_vector = [features.Contrast, features.Correlation, ...
    features.Energy, features.Homogeneity, r, g, b];

%% Step 5: Load Training Data and Train SVM (if not saved)
if exist('trained_model.mat', 'file')
    load('trained_model.mat'); % Load pre-trained model
else
    % Extract features from training set
    folders = {'healthy', 'bacterial', 'fungal', 'viral'};
    feature_matrix = [];
    labels = [];

    for i = 1:length(folders)
        folderPath = fullfile('train', folders{i});
        files = dir(fullfile(folderPath, '*.jpg'));

        for j = 1:length(files)
            filePath = fullfile(folderPath, files(j).name);
            I = imread(filePath);
            I = imresize(I, [256 256]);

            % Repeat same process
            lab = rgb2lab(I);
            ab = double(lab(:,:,2:3));
            ab = reshape(ab, [], 2);
            [c_idx, ~] = kmeans(ab, nColors, 'distance', 'sqEuclidean', ...
                'Replicates', 3);
            p_labels = reshape(c_idx, size(I,1), size(I,2));
            means = zeros(1, nColors);
            for k = 1:nColors
                mask = p_labels == k;
                means(k) = mean(I(repmat(mask, [1, 1, 3])));
            end
            [~, disease_k] = min(means);
            mask = p_labels == disease_k;

            % Extract features
            cluster_img = I;
            cluster_img(repmat(~mask, [1,1,3])) = 0;
            gray = rgb2gray(cluster_img);
            glcm = graycomatrix(gray, 'Offset', [0 1]);
            props = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

            r = mean2(cluster_img(:,:,1));
            g = mean2(cluster_img(:,:,2));
            b = mean2(cluster_img(:,:,3));

            feat = [props.Contrast, props.Correlation, props.Energy, props.Homogeneity, r, g, b];
            feature_matrix = [feature_matrix; feat];
            labels = [labels; i];
        end
    end

    % Train SVM
    classifier = fitcecoc(feature_matrix, labels);
    save('trained_model.mat', 'classifier', 'folders');
end

%% Step 6: Classify Test Image
predicted_label = predict(classifier, feature_vector);
disease_name = folders{predicted_label};

fprintf('\n🔍 Predicted Disease: %s\n', upper(disease_name));
msgbox(['Predicted Disease: ', upper(disease_name)], 'Result');
