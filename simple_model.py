import numpy as np
import pandas as pd
import os
from PIL import Image


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
train_images = train_images[:round(len(train_images) *.10)]
train_labels = labels_df['label'].values

# Load test data
#test_images = load_images(test_data_path)

print("Data Characteristics: ")
print(f"Train images shape: {train_images.shape}  |  Number of Samples: {len(train_images)}")
print(f"Train labels shape: {train_labels.shape}")
#print(f"Test images shape: {test_images.shape}  |  Number of Samples: {len(test_images)}")

# Shuffle and split the data
indices = np.arange(len(train_images))
np.random.shuffle(indices)
split = int(0.75 * len(train_images))

train_indices = indices[:split]
val_indices = indices[split:]

X_train = train_images[train_indices]
y_train = train_labels[train_indices]
X_val = train_images[val_indices]
y_val = train_labels[val_indices]

# Create CNN instance
import simple_cnn
cnn = simple_cnn.CNN()

batch_predictions = []
i = 0
for batch in process_in_batches(train_images):
    print(f"Batch {i} of {len(train_images) // 32}")
    batch_pred = cnn.forward(batch)
    batch_predictions.append(batch_pred)
    i += 1

predictions = np.concatenate(batch_predictions, axis=0)

# Evaluate on validation set
val_predictions = cnn.forward(X_val)
val_predictions = (val_predictions > 0.5).astype(int)  # Assuming binary classification

accuracy = np.mean(val_predictions == y_val)
print(f"Validation accuracy: {accuracy * 100:.2f}%")