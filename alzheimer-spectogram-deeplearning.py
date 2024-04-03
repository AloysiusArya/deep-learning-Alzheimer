from google.colab import drive
drive.mount('/content/drive')

import os, re, glob, cv2, numpy as np
from os import listdir
from os.path import isfile, join
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image

drive_path = '/content/drive/MyDrive/tubes/tubes_spectrogram'
print(os.listdir(drive_path))
training_path = os.path.join(drive_path, 'train_data')
print(os.listdir(training_path))


import os
import tensorflow as tf

# Replace this with the path to your dataset directory
drive_path = '/content/drive/MyDrive/tubes/tubes_spectrogram'

BATCH_SIZE = 32
IMG_SIZE = (256, 256)
NUM_CLASSES = 2

# Path to the 'Training' directory
training_path = os.path.join(drive_path, 'train_data')
# Create the training dataset using image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    training_path,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    labels='inferred',  # Automatically infer labels from subdirectory names
    label_mode='categorical'  # Use categorical labels
)

def prepare_image(img,target_size=(256, 256)):
    img_resized = cv2.resize(img, target_size)
    img_array = image.img_to_array(img_resized)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img_array_expanded_dims)

def preprocess_batch(batch_images, batch_labels):
    preprocessed_images = []
    for img in batch_images:
        preprocessed_images.append(prepare_image(img.numpy()))  # Convert to numpy array
    return np.array(preprocessed_images), batch_labels

for batch_images, batch_labels in train_dataset:
    processed_images, labels = preprocess_batch(batch_images, batch_labels)

testing_path = os.path.join(drive_path, 'test_data')
testing_dataset = tf.keras.utils.image_dataset_from_directory(
    testing_path,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    labels='inferred',  # Automatically infer labels from subdirectory names
    label_mode='categorical'  # Use categorical labels
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Load a pre-trained model (e.g., MobileNet) with weights from ImageNet
base_model = ResNet50(include_top=False, weights='imagenet',  input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# Function to create a transfer learning model
def create_transfer_model(base_model):
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

res_transfer = create_transfer_model(base_model)
res_transfer.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
res_transfer.summary()

epochs = 25
res_history = res_transfer.fit(train_dataset, epochs=epochs, validation_data=testing_dataset)

from keras.applications.resnet50 import preprocess_input
# Specify the path to your test or test data directory
test_data_dir = '/content/drive/MyDrive/tubes/tubes_spectrogram/test_data'

# Use ImageDataGenerator for test data with preprocessing function
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create a flow from the directory generator for test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important: Set shuffle to False for test data
)

# Evaluate the model on the test data
test_results = res_transfer.evaluate_generator(generator=test_generator)

# Print the accuracy of the model on the test data
print(f"Test Accuracy: {test_results[1]*100:.2f}%")

# Evaluate the models
def evaluate_model(model, dataset, name):
    print(f"Evaluating {name} model:")
    results = model.evaluate(dataset)
    print(f"Loss: {results[0]}, Accuracy: {results[1]}")

evaluate_model(res_transfer, testing_dataset, 'ResNet')

import matplotlib.pyplot as plt
# Plot training history
def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(res_history, 'ResNet Transfer Learning')

# Predictions
def make_predictions(model, dataset):
    predictions = model.predict(dataset)
    return tf.argmax(predictions, axis=1)

# Collect true labels and predicted labels for classification report
true_labels = []
res_predictions = make_predictions(res_transfer, testing_dataset)

for images, labels in testing_dataset:
    true_labels.extend(tf.argmax(labels, axis=1))

# Generate classification reports
res_report = classification_report(true_labels, res_predictions.numpy())

# Print classification reports
print("ResNet Transfer Learning Classification Report:")
print(res_report)