# Leaf Classification using CNN

## Overview
This project focuses on classifying different types of leaves using a Convolutional Neural Network (CNN). The dataset includes images of leaves and extracted numerical features stored in `train.csv`. The model is trained to predict leaf species based on image features and extracted characteristics.

## Dataset
The dataset consists of:
- **Leaf Images**: Available in [Google Drive](https://drive.google.com/drive/folders/1VDB8mPq9o42QG4ifoRXGoD9xs9rGnHx0)
- **Features and Labels**: Stored in `train.csv`, located in the repository.

## Model Architecture
The CNN model consists of:
- **Convolutional layers** to extract hierarchical features.
- **Batch Normalization** for stable training.
- **Max Pooling layers** for downsampling.
- **Global Average Pooling** for feature aggregation.
- **Fully connected layers** to classify leaf species.
- **Dropout layers** and **L2 regularization** to prevent overfitting.
- **Feature extraction using PCA** for dimensionality reduction.
- **Fusion of tabular features**: After convolutional layers process the image, the extracted tabular features from `train.csv` are fed into the fully connected layers.

## Training Process
1. **Data Preprocessing**:
   - Load images and features from `train.csv`.
   - Resize and normalize images.
   - Convert categorical labels into numerical form.
   - Apply PCA for feature extraction.

2. **Model Training**:
   - Train the CNN using `train.py`.
   - Use cross-entropy loss for classification.
   - Optimize using Adam optimizer.
   - Implement **ModelCheckpoint** to save the best model.
   - Use early stopping to prevent overfitting.

3. **Evaluation**:
   - Validate the model on a test set.
   - Measure accuracy, precision, and recall.
   - Visualize predictions and misclassifications.

## Results
- The trained CNN achieves high accuracy in classifying leaf species.
- The model can be improved further with data augmentation and fine-tuning.

## Future Improvements
- Experiment with deeper architectures like ResNet.
- Implement transfer learning for better performance.
- Optimize hyperparameters using grid search.


