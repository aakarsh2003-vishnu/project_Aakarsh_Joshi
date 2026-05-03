# interface.py
# Standard names exported for grading.

from config import batchsize as the_batch_size
from config import epochs as total_epochs
from dataset import VehicleDataset as TheDataset
from dataset import vehicle_dataloader as the_dataloader
from model import VehicleClassifier as TheModel
from predict import predict_vehicles as the_predictor
from train import train_model as the_trainer
