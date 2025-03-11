import itertools
import json
import os
import pandas as pd
import numpy as np
import ast
import re

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import Dice, SquaredHinge, Hinge
from tensorflow.keras.metrics import BinaryCrossentropy as bce_metric, BinaryAccuracy, Precision, Recall, F1Score, AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from utilities.loaders import concur_load_data, save_meta_data, split_data
from models.llanes_jurado import LSTM_CNN
from models.cueva import LSTM_SVM, LSTM_FE

from argparse import ArgumentParser

