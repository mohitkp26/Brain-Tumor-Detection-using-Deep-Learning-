"""
predict.py
Run a single-image prediction from the command line.

Usage:
    python predict.py --image path/to/scan.jpg --model saved_model/brain_tumor_model.h5
"""

import argparse

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

from model.model_arch import CLASS_NAMES, IMG_SIZE


def predict(image_path, model_path):
    model = load_model(model_path)

    img = keras_image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))

    result = {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": float(preds[top_idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
        },
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="saved_model/brain_tumor_model.h5")
    args = parser.parse_args()

    result = predict(args.image, args.model)
    print(f"Prediction: {result['predicted_class']}  "
          f"(confidence: {result['confidence']*100:.2f}%)")
    print("All probabilities:")
    for cls, prob in result["all_probabilities"].items():
        print(f"  {cls:12s}: {prob*100:5.2f}%")
