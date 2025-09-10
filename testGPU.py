#!/usr/bin/env python3
"""
Cryptocurrency Short-term Price Prediction System
================================================

A modular system for training and evaluating multiple deep learning and ensemble
models for cryptocurrency price prediction (5-20 minute horizons).

Author: AI Assistant
License: MIT
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

# Data handling and preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# XGBoost for ensemble
import xgboost as xgb

# Data source
# import ccxt

# Suppress warnings
warnings.filterwarnings('ignore')

# Set the environment variable for PyTorch CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")  # Use the GPU
else:
    print("GPU is not available, using CPU.")
    device = torch.device("cpu")  # Use the CPU
