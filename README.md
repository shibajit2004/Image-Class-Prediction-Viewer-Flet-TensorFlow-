# Image Class Prediction & Viewer (Flet + TensorFlow)

This is a desktop/web application built with **Flet**, **TensorFlow**, and **MobileNetV2** that allows you to:
- Upload an image
- Predict its class using a pre-trained classifier
- Display similar images from a local or database-backed dataset
- Retrain the model directly from the interface

---

## 🚀 Features

- ✅ Upload and view images instantly
- ✅ Predict class using a Keras model
- ✅ Feature extraction using **MobileNetV2**
- ✅ Show related images based on prediction
- ✅ Retrain model on-the-fly with a button click
- ✅ SQLite or PostgreSQL-compatible backend (via `db_utils_CNN.py`)

---

## 🖼️ UI Overview

- **Image Preview**: Displays the uploaded image.
- **Prediction Text**: Shows predicted class and confidence.
- **Slider**: Select how many related images to show.
- **Related Images Grid**: Shows thumbnails from dataset.
- **Retrain Button**: Trigger training script (`retrain.py`) from UI.

---

## 🛠️ Requirements

- Python 3.8+
- `flet`
- `tensorflow`
- `pillow`
- `numpy`
- `psycopg2-binary` (if using PostgreSQL)
- Pre-trained classifier saved (e.g., `model.pkl` via `joblib`)
- Optional: PostgreSQL with a `photo_dataset` table (see `db_utils_CNN.py`)

### Install dependencies:

```bash
pip install flet tensorflow pillow numpy psycopg2-binary
