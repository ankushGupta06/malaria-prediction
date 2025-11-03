#Implement Data Augmentation in Deep Learning using any Medical Image Datasets

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# List all available datasets in TensorFlow Datasets
print(tfds.list_builders())

# Load the Malaria Dataset
(ds_train, ds_test), ds_info = tfds.load(
"malaria",
split=["train[:80%]", "train[80%:]"],  # 80% training, 20% testing
as_supervised=True,
# Includes labels with images
with_info=True
)

# Visualize dataset information
print(ds_info)

# Visualize some sample images
def plot_samples(dataset, title, rows=3, cols=3):
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(dataset.take(rows * cols)):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(image.numpy())
        plt.title("Uninfected" if label.numpy() == 1 else "Infected")
        plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.show()
print("Sample Images from Malaria Dataset:")
plot_samples(ds_train, "Malaria Dataset Samples")

# Data Augmentation function
def augment_image(image, label):
    # Apply random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Apply random vertical flip
    image = tf.image.random_flip_up_down(image)
    # Adjust brightness
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Adjust contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random zoom (scale the image)
    image = tf.image.resize(image, (128, 128))  # Resize to 128x128
    image = tf.image.resize_with_crop_or_pad(image, 140, 140)  # Crop or pad to 140x140
    image = tf.image.random_crop(image, size=[128, 128, 3])  # Crop to 128x128

    return image, label

# Preprocessing function for resizing and normalizing images
def preprocess_image(image, label):
  image = tf.image.resize(image, (128, 128))  # Resize all images to 128x128
  image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
  return image, label

# Augment and preprocess the training dataset
batch_size = 32
augmented_train_ds = (
    ds_train.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)  # Apply augmentation
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)      # Preprocess the image
    .shuffle(1000)                                                  # Shuffle the dataset
    .batch(batch_size)                                              # Create batches
    .prefetch(tf.data.AUTOTUNE)                                     # Prefetch for performan
 )

# Preprocess the testing dataset (without augmentation)
test_ds = (
    ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)  # Preprocess images
    .batch(batch_size)                                                  # Create batches
    .prefetch(tf.data.AUTOTUNE)                                         # Prefetch for performance
)

# Visualize some augmented images
print("Augmented Dataset Samples:")
plot_samples(augmented_train_ds.unbatch(), "Augmented Malaria Dataset Samples")

# Build a Convolutional Neural Network (CNN) model
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classifi
 ])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the augmented dataset
history = model.fit(augmented_train_ds, validation_data=test_ds, epochs=10)

# Plot the training and validation accuracy
plt.figure(figsize=(8, 2))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("malaria_parasite.keras")


# Load the trained model
model = load_model("malaria_parasite.keras")


# Predict on a new image
image_path = "your_image_path_here.jpg"  # Replace with your image path


print(image_path)
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, (128, 128))   # same size used in training
image = tf.cast(image, tf.float32) / 255.0   # normalize
image = tf.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)
print(prediction[0][0])

# Convert sigmoid output to class label with a higher threshold
if prediction[0][0] < 0.5:
    label = "Parasitized"
else:
    label = "Uninfected"

print(f"Predicted Label: {label}")