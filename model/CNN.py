# Verify installation of TensorFlow
import tensorflow as tf
print(tf.__version__)

# Import related packages for CNN
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# Import Packages for Plotting and Visualization
import cv2
import random
from matplotlib.image import imread
import seaborn as sns
import matplotlib
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# In Winodws, file path is fulfilled by '\' and r- is added ahead for 'raw-path'
data_dir = r'..\data\gtsrb-german-traffic-sign'

# Paths to CSV files
meta_csv_path = os.path.join(data_dir, 'Meta.csv')
test_csv_path = os.path.join(data_dir, 'Test.csv')
train_csv_path = os.path.join(data_dir, 'Train.csv')

meta_df = pd.read_csv(meta_csv_path)
test_data = pd.read_csv(test_csv_path)
train_data = pd.read_csv(train_csv_path)

print('Length of training data', len(train_data))
print('Length of testing data',len(test_data))
print(f'Total Images = {len(train_data)} + {len(test_data)} = {len(train_data) + len(test_data)}')

# Output classes number
len(train_data.ClassId.value_counts())

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

# Visualizing 25 random images from test data
test = pd.read_csv(test_csv_path)
imgs = test["Path"].values

plt.figure(figsize=(25,25))

for i in range(1,26):
    plt.subplot(5,5,i)
    random_img_path = data_dir + '\\' + random.choice(imgs)
    rand_img = imread(random_img_path)
    plt.imshow(rand_img)
    plt.grid()
    plt.xlabel(rand_img.shape[1], fontsize = 20)#width of image
    plt.ylabel(rand_img.shape[0], fontsize = 20)#height of image

# Overall distribution on images with different classes
df = train_data.ClassId.value_counts() 
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
plt.xlabel('Classes')

plt.title('Overall distribution on images with the classes')
sns.barplot(x=df.index, y=df)
plt.ylabel('Images under each Class')

## Setting up the CNN model

# Step 1: Setting up the ImageDataGenerator

train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

# Step 2: Configuring the TensorFlow generators

train_dir = r'..\data\gtsrb-german-traffic-sign\Train'

train_generator = train_datagen.flow_from_directory(
   train_dir,  # This is the source directory for training images
   subset='training',
   target_size=(30, 30),  # All images will be resized to 150x150
   batch_size=32,
   color_mode='rgb',
   shuffle=True,
   seed=42,
   # Since we use binary_crossentropy loss, we need binary labels
   class_mode='categorical')
# Flow validation images in batches of 20 using test_datagen generator
validation_generator = train_datagen.flow_from_directory(
   train_dir,
   subset='validation',
   target_size=(30, 30),
   batch_size=32,
   color_mode='rgb',
   shuffle=True,
   seed=42,
   class_mode='categorical')

# Step 3: Building the CNN Model !!!!!!
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(30,30,3)),
    # tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    # tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(8, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(43, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


## Training the CNN model

# Step 1: Model training and results analysis
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//train_generator.batch_size,
    epochs=10,
    verbose=1,
    validation_data = validation_generator,
    validation_steps=validation_generator.n//validation_generator.batch_size)

def plot_graphs(history, string):
  plt.plot(history.history[string], 'b-x', linewidth=4, markersize=12, markeredgewidth=4, markeredgecolor='navy')
  plt.plot(history.history['val_'+string],'r--o', linewidth=4, markersize=12,)
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# Step 2: Running the model on test data
test_dir = data_dir

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    classes = ['Test'],
    target_size=(30,30),
    batch_size=32,
    color_mode='rgb',
    shuffle=False,
    seed=42,
    class_mode='categorical')

preds = model.predict(test_generator, steps=len(test_generator))

preds_cls_idx = preds.argmax(axis=-1)
idx_to_cls = {v: k for k, v in train_generator.class_indices.items()}
preds_cls = np.vectorize(idx_to_cls.get)(preds_cls_idx)
filenames_to_cls = list(zip(test_generator.filenames, preds_cls))

filenames_to_cls[:10]

# Step 3: Accuracy Score
labels = test_data["ClassId"].values
print('Test Data accuracy: ',accuracy_score(labels, preds_cls.astype(int))*100)

# Step 4: Confusion Matrix
cf = confusion_matrix(labels, preds_cls.astype(int))
test_data.ClassId.values
df_cm = pd.DataFrame(cf, index = df.index,  columns = df.index)
plt.figure(figsize = (20,20))
sns.heatmap(df_cm, annot=True)

# Step 5: Classification report
print(classification_report(labels, preds_cls.astype(int)))


## Testing with individual images
imgs = test_data["Path"].values
data =[]

for img in imgs:
    try:
        image = cv2.imread(data_dir + '\\' + img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((30, 30))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test/255

# Visualize the effect of our CNN Model

plt.figure(figsize = (25, 25))
labels = test_data.ClassId.values
start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = preds_cls[start_index + i]
    actual = labels[start_index + i]
    col = 'g' # Correct Prediction
    if prediction.astype(int) != actual:
        col = 'r' # Wrong Prediction
    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color = col)
    plt.imshow(X_test[start_index + i])
plt.show()