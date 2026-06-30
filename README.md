# NeuroScan — Brain Tumor Detection using Deep Learning

A full-stack app that classifies brain MRI scans into **glioma**, **meningioma**,
**pituitary tumor**, or **no tumor** using a convolutional neural network,
served through a Flask API to a React frontend.

```
brain-tumor-detection/
├── backend/
│   ├── model/
│   │   └── model_arch.py     # CNN + transfer-learning (EfficientNet) architectures
│   ├── train.py               # trains the model on a labeled MRI dataset
│   ├── predict.py             # CLI single-image prediction
│   ├── app.py                 # Flask API (/api/predict, /api/health)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx             # upload UI + results
    │   ├── main.jsx
    │   └── index.css
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## 1. Dataset

Use a labeled MRI dataset such as the public *Brain Tumor MRI Dataset* (Kaggle),
arranged like:

```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

## 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the model (produces saved_model/brain_tumor_model.h5)
python train.py --data_dir ../dataset --epochs 25 --model_type cnn

# Or try transfer learning (EfficientNetB0) for higher accuracy
python train.py --data_dir ../dataset --epochs 15 --model_type transfer

# Quick CLI test on a single image
python predict.py --image path/to/scan.jpg

# Start the API
python app.py        # runs on http://localhost:5000
```

`app.py` loads the trained model lazily on the first request from
`saved_model/brain_tumor_model.h5` (override with the `MODEL_PATH` env var).

## 3. Frontend setup

```bash
cd frontend
npm install
npm run dev           # runs on http://localhost:5173
```

The Vite dev server proxies `/api/*` requests to the Flask backend on port
5000 (see `vite.config.js`), so just run both servers side by side.

## 4. How it works

1. The user drops/selects an MRI slice (PNG/JPEG) in the React app.
2. On **Run Analysis**, the image is POSTed to `/api/predict` as `multipart/form-data`.
3. Flask resizes the image to 150×150, normalizes it, and runs it through the
   trained Keras CNN.
4. The API returns the predicted class, confidence, and the full probability
   distribution across all four classes, which the UI renders as a result
   panel with per-class bars.

## 5. Model architecture

`model/model_arch.py` ships two options:

- **`build_cnn()`** — a from-scratch 4-block Conv2D/BatchNorm/MaxPool stack
  with global average pooling and dropout, good as a baseline and fast to
  train.
- **`build_transfer_model()`** — EfficientNetB0 with a frozen ImageNet base
  and a small classification head, typically converges faster and reaches
  higher accuracy with the same dataset.

## Disclaimer

This project is for educational/research purposes only. Predictions are not
a medical diagnosis — always defer to a licensed radiologist.

Small README tweak 1 - 2026-06-30T18:06:13+05:30
Small README tweak 2 - 2026-06-30T18:06:13+05:30
Small README tweak 3 - 2026-06-30T18:06:13+05:30