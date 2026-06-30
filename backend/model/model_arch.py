"""
model_arch.py
Defines the CNN architecture used for brain tumor classification
from MRI scans. Classifies images into 4 classes:
    0 - glioma
    1 - meningioma
    2 - no tumor
    3 - pituitary
"""

from tensorflow.keras import layers, models, regularizers

IMG_SIZE = 150
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Builds and returns a compiled CNN for MRI tumor classification."""

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_transfer_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Alternative model using transfer learning (EfficientNetB0).
    Generally gives better accuracy with less training data.
    """
    from tensorflow.keras.applications import EfficientNetB0

    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = False  # freeze base for initial training

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
