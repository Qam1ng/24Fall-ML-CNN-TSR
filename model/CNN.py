import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import shutil
import warnings
import cv2
import random
from matplotlib.image import imread
import seaborn as sns
import matplotlib
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_dir = r'C:\Users\hello\OneDrive\Desktop\ML-CNN-TSR\data\gtsrb-german-traffic-sign'

# Paths to CSV files
meta_csv_path = os.path.join(data_dir, 'Meta.csv')
test_csv_path = os.path.join(data_dir, 'Test.csv')
train_csv_path = os.path.join(data_dir, 'Train.csv')

meta_df = pd.read_csv(meta_csv_path)
test_data = pd.read_csv(test_csv_path)
train_data = pd.read_csv(train_csv_path)

# Example for loading an image
im1 = cv2.imread(os.path.join(data_dir, 'Train', '0', '00000_00000_00000.png'))
im2 = cv2.imread(os.path.join(data_dir, 'Train', '0', '00000_00000_00005.png'))

# Verify images are loaded
if im1 is None:
    print("im1 could not be loaded. Please check the path.")
else:
    print("im1 loaded successfully.")

if im2 is None:
    print("im2 could not be loaded. Please check the path.")
else:
    print("im2 loaded successfully.")

# Print image shapes
print('Size of im1:', im1.shape)
print('Size of im2:', im2.shape)

# Convert images to RGB for displaying
im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# Display im1
plt.figure(figsize=(5, 5))
plt.imshow(im1_rgb)
plt.title('Image 1')
plt.axis('off')
plt.show()

# Display im2
plt.figure(figsize=(5, 5))
plt.imshow(im2_rgb)
plt.title('Image 2')
plt.axis('off')
plt.show()