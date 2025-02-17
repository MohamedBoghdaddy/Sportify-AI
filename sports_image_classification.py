import os
import random
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import shutil
import json
from PIL import Image

# Configuration Parameters
CONFIG = {
    "zip_path": "Dataset.zip",
    "extract_to": "./dataset",
    "train_subdir": "Train",
    "test_subdir": "Test",
    "organized_train_dir": "./dataset/Organized_Train",
    "image_size": (224, 224),
    "batch_size": 32,
    "num_classes": 6,
    "epochs": 50,
    "patience": 5,
    "cnn_model_path": "cnn_model.keras",
    "resnet_model_path": "resnet_model.keras",
    "cnn_history_path": "cnn_history.json",
    "resnet_history_path": "resnet_history.json",
    "output_dir": "./predicted_images"
}

# Utility Functions

def unzip_dataset(zip_path, extract_to):
    """
    Unzips a dataset to a specified directory if it doesn't already exist.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print(f"Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Dataset extracted to {extract_to}")

def organize_dataset(source_dir, destination_dir):
    """
    Organizes images into subdirectories based on class names derived from file names.
    """
    os.makedirs(destination_dir, exist_ok=True)
    for file in os.listdir(source_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            class_name = file.split("_")[0]
            class_dir = os.path.join(destination_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.move(os.path.join(source_dir, file), os.path.join(class_dir, file))
    print("Dataset reorganized into class-based subdirectories.")

def create_generators(destination_dir, image_size, batch_size):
    """
    Creates data generators for training and validation with data augmentation applied to the training data.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        destination_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="training"
    )
    val_generator = val_datagen.flow_from_directory(
        destination_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation"
    )

    return train_generator, val_generator

def create_cnn_model(input_shape, num_classes):
    """
    Builds and compiles a Convolutional Neural Network (CNN) model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),  
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def create_resnet_model(input_shape, num_classes):
    """
    Builds and compiles a ResNet50-based model with transfer learning.
    """
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze the base model initially
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=optimizers.AdamW(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def save_training_history(history, filename):
    """
    Saves the training history of a model to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(history.history, f)

def load_training_history(filename):
    """
    Loads the training history of a model from a JSON file.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def plot_comparison(history_cnn, history_resnet, test_accuracies):
    """
    Plots training, validation, and test accuracies for CNN and ResNet50 models.
    """
    plt.figure(figsize=(14, 8))
    cnn_epochs = range(1, len(history_cnn["accuracy"]) + 1)
    resnet_epochs = range(1, len(history_resnet["accuracy"]) + 1)

    plt.plot(cnn_epochs, history_cnn["accuracy"], label="CNN Training Accuracy")
    plt.plot(cnn_epochs, history_cnn["val_accuracy"], label="CNN Validation Accuracy")
    plt.plot(resnet_epochs, history_resnet["accuracy"], label="ResNet50 Training Accuracy")
    plt.plot(resnet_epochs, history_resnet["val_accuracy"], label="ResNet50 Validation Accuracy")

    plt.axhline(y=test_accuracies["cnn"], color="blue", linestyle="--", label="CNN Test Accuracy")
    plt.axhline(y=test_accuracies["resnet"], color="green", linestyle="--", label="ResNet50 Test Accuracy")

    plt.title("Training, Validation, and Test Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

def preprocess_test_images_flat(test_dir, target_size):
    """
    Preprocesses test images for model evaluation.
    """
    test_images, filenames = [], []
    for file in os.listdir(test_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(test_dir, file)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            test_images.append(img_array)
            filenames.append(file)
    print(f"Processed {len(test_images)} test images from {test_dir}")
    return np.array(test_images), filenames

def save_and_display_random_predictions(test_images, filenames, cnn_predictions, resnet_predictions, class_labels):
    """
    Saves 5 random test images with CNN and ResNet50 predictions to a directory
    and displays them in a plot.
    """
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    random_indices = random.sample(range(len(test_images)), 10)
    cnn_indices = random_indices[:5]
    resnet_indices = random_indices[5:]

    # Save and display CNN predictions
    print("\nSaving and displaying 5 random test images with CNN predictions:")
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(cnn_indices):
        image = (test_images[idx] * 255).astype("uint8")
        pil_image = Image.fromarray(image, 'RGB')
        filename = f"cnn_{filenames[idx]}_pred_{class_labels[cnn_predictions[idx]]}.png"
        pil_image.save(os.path.join(CONFIG["output_dir"], filename))
        
        # Display the saved image
        plt.subplot(2, 5, i + 1)
        plt.imshow(pil_image)
        plt.title(f"CNN: {class_labels[cnn_predictions[idx]]}")
        plt.axis("off")

    # Save and display ResNet50 predictions
    print("\nSaving and displaying 5 random test images with ResNet50 predictions:")
    for i, idx in enumerate(resnet_indices):
        image = (test_images[idx] * 255).astype("uint8")
        pil_image = Image.fromarray(image, 'RGB')
        filename = f"resnet_{filenames[idx]}_pred_{class_labels[resnet_predictions[idx]]}.png"
        pil_image.save(os.path.join(CONFIG["output_dir"], filename))
        
        # Display the saved image
        plt.subplot(2, 5, i + 6)
        plt.imshow(pil_image)
        plt.title(f"ResNet50: {class_labels[resnet_predictions[idx]]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Main Workflow
unzip_dataset(CONFIG["zip_path"], CONFIG["extract_to"])
organize_dataset(os.path.join(CONFIG["extract_to"], CONFIG["train_subdir"]), CONFIG["organized_train_dir"])

train_generator, val_generator = create_generators(CONFIG["organized_train_dir"], CONFIG["image_size"], CONFIG["batch_size"])
class_labels = list(train_generator.class_indices.keys())
input_shape = CONFIG["image_size"] + (3,)

cnn_history = load_training_history(CONFIG["cnn_history_path"])
resnet_history = load_training_history(CONFIG["resnet_history_path"])

if cnn_history is None or resnet_history is None:
    print("Training models...")
    early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=CONFIG["patience"], restore_best_weights=True)

    cnn_model = create_cnn_model(input_shape, CONFIG["num_classes"])
    history_cnn = cnn_model.fit(train_generator, validation_data=val_generator, epochs=CONFIG["epochs"], callbacks=[early_stopping])
    cnn_model.save(CONFIG["cnn_model_path"])
    save_training_history(history_cnn, CONFIG["cnn_history_path"])
    cnn_history = history_cnn.history

    resnet_model = create_resnet_model(input_shape, CONFIG["num_classes"])
    history_resnet = resnet_model.fit(train_generator, validation_data=val_generator, epochs=CONFIG["epochs"], callbacks=[early_stopping])
    resnet_model.save(CONFIG["resnet_model_path"])
    save_training_history(history_resnet, CONFIG["resnet_history_path"])
    resnet_history = history_resnet.history
else:
    print("Loading existing models...")
    cnn_model = tf.keras.models.load_model(CONFIG["cnn_model_path"])
    resnet_model = tf.keras.models.load_model(CONFIG["resnet_model_path"])

test_images, filenames = preprocess_test_images_flat(os.path.join(CONFIG["extract_to"], CONFIG["test_subdir"]), CONFIG["image_size"])
true_labels = [int(os.path.splitext(name)[0]) for name in filenames]

cnn_predictions = np.argmax(cnn_model.predict(test_images), axis=1)
resnet_predictions = np.argmax(resnet_model.predict(test_images), axis=1)

cnn_test_accuracy = accuracy_score(true_labels, cnn_predictions)
resnet_test_accuracy = accuracy_score(true_labels, resnet_predictions)

print(f"\nTest Accuracy (CNN): {cnn_test_accuracy * 100:.2f}%")
print(f"Test Accuracy (ResNet50): {resnet_test_accuracy * 100:.2f}%")

save_and_display_random_predictions(test_images, filenames, cnn_predictions, resnet_predictions, class_labels)

test_accuracies = {"cnn": cnn_test_accuracy, "resnet": resnet_test_accuracy}
plot_comparison(cnn_history, resnet_history, test_accuracies)
