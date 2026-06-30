"""
train.py
Trains the brain tumor classification CNN on an MRI dataset.

Expected dataset layout (e.g. the public "Brain Tumor MRI Dataset" on Kaggle):

    dataset/
        Training/
            glioma/
            meningioma/
            no_tumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            no_tumor/
            pituitary/

Usage:
    python train.py --data_dir dataset --epochs 25 --model_type cnn
"""

import argparse
import os

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.model_arch import IMG_SIZE, build_cnn, build_transfer_model


def get_data_generators(data_dir, batch_size=32):
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.15,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def plot_history(history, out_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to dataset root (contains Training/ and Testing/)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="cnn",
                         choices=["cnn", "transfer"])
    parser.add_argument("--out", type=str, default="saved_model/brain_tumor_model.h5")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train_gen, val_gen, test_gen = get_data_generators(args.data_dir, args.batch_size)

    model = build_cnn() if args.model_type == "cnn" else build_transfer_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history)

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    model.save(args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
