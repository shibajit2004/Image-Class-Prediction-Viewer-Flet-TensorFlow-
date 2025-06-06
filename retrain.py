# retrain.py
from db_utils_CNN import load_dataset_from_db  # You can define this function similarly
from model_utils import train_model

X, y = load_dataset_from_db()
train_model(X, y)
