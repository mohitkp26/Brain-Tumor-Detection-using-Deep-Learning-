# Brain Tumor Detection using Deep Learning

## Overview

This project implements a deep learning-based solution for automated brain tumor detection and classification from medical imaging data. The system combines a robust backend API with an intuitive frontend interface to provide a comprehensive platform for medical image analysis.

## Features

- Deep learning model for brain tumor detection and classification
- RESTful API backend for image processing and predictions
- Interactive web-based user interface
- Model training pipeline with configurable parameters
- Support for multiple image formats
- Real-time prediction capabilities

## Project Structure

```
brain-tumor-detection/
├── backend/
│   ├── app.py                 # Flask application entry point
│   ├── predict.py            # Prediction logic
│   ├── train.py              # Model training script
│   ├── requirements.txt       # Python dependencies
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model_arch.py     # Neural network architecture
│   ├── saved_model/          # Pre-trained model storage
│   └── uploads/              # Temporary upload directory
├── frontend/
│   ├── index.html            # Main HTML template
│   ├── package.json          # Node dependencies
│   ├── vite.config.js        # Vite build configuration
│   ├── src/
│   │   ├── App.jsx           # React main component
│   │   ├── main.jsx          # Application entry point
│   │   └── index.css         # Styling
│   └── public/               # Static assets
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- pip package manager
- npm or yarn package manager

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the saved_model directory exists:
```bash
mkdir -p saved_model
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

## Usage

### Running the Backend

To start the Flask API server:
```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000` by default.

### Training the Model

To train the model with your dataset:
```bash
cd backend
python train.py
```

### Running the Frontend

To start the development server:
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` by default.

### Making Predictions

Send an image to the prediction endpoint:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

## API Endpoints

- `POST /predict` - Submit an image for tumor detection
- `GET /health` - Check API status

## Model Architecture

The neural network architecture is defined in `backend/model/model_arch.py`. The model is trained on brain MRI images to detect and classify tumor regions.

## Dependencies

### Backend
- Flask
- TensorFlow/PyTorch
- NumPy
- Pillow
- scikit-learn

### Frontend
- React
- Vite
- Axios

See `backend/requirements.txt` and `frontend/package.json` for complete dependency lists.

## Data Format

The model expects MRI brain scan images in standard formats:
- JPEG
- PNG
- DICOM (with appropriate preprocessing)

Images should be preprocessed to:
- Standard resolution (as per model requirements)
- Normalized pixel values
- Proper orientation

## Contributing

Contributions are welcome. Please ensure:
- Code follows PEP 8 style guidelines
- All new features include appropriate tests
- Documentation is updated accordingly

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue in the repository.

## Disclaimer

This tool is intended for research and educational purposes. It should not be used for clinical diagnosis without proper validation and approval from medical professionals.