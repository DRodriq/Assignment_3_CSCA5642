
import numpy as np
import pandas as pd
import os
from PIL import Image

import tf_cnn

# Set paths
train_data_path = 'data/train'
test_data_path = 'data/test'
labels_path = 'data/train_labels.csv'

# Load training labels
labels_df = pd.read_csv(labels_path)

def load_images(data_path):
    images = []
    for filename in os.listdir(data_path):
        if filename.endswith('.tif'):
            img_path = os.path.join(data_path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)

def process_in_batches(images, batch_size=32):
    for i in range(0, len(images), batch_size):
        yield images[i:i+batch_size]

# Load training data
train_images = load_images(train_data_path)
train_images = train_images[:round(len(train_images) *.20)]
train_labels = labels_df['label'].values

# Load test data
test_images = load_images(test_data_path)

train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

print("Data Characteristics: ")
print(f"Train images shape: {train_images.shape}  |  Number of Samples: {len(train_images)}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}  |  Number of Samples: {len(test_images)}")

# Create and compile the model
model = tf_cnn.CNN()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_split=0.25, batch_size=32)

# After training the model
model.save('cancer_detection_model')

# Generate predictions for test data
test_predictions = model.predict(test_images)
test_predictions = (test_predictions > 0.5).astype(int).flatten()

# Create a DataFrame with the predictions
test_filenames = [f.split('.')[0] for f in os.listdir(test_data_path) if f.endswith('.tif')]
results_df = pd.DataFrame({
    'id': test_filenames,
    'label': test_predictions
})

# Save the results to a CSV file
results_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")