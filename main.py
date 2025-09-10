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


class Config:
    """Configuration class for the prediction system."""

    def __init__(self):
        # Data parameters
        self.csv_file = None
        self.timestamp_col = 'Timestamp'
        self.required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.target_col = 'Close'
        self.prediction_horizons = [5, 10, 15, 20]  # minutes
        self.sequence_length = 60  # lookback window

        # Model parameters
        self.lstm_hidden_size = 64
        self.lstm_num_layers = 2
        self.cnn_filters = 32
        self.transformer_d_model = 64
        self.transformer_nhead = 8
        self.transformer_num_layers = 2

        # Training parameters
        self.batch_size = 16  # 32

        self.epochs = 50
        self.learning_rate = 0.001
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Models to train
        self.models_to_train = {
            'LSTM': True,
            'CNN_LSTM': True,
            'Transformer': True,
            'LSTM_XGBoost': True,
            'RL_Agent': False  # Optional, set to True to enable
        }

        # Output directory
        self.output_dir = 'crypto_prediction_results'


class CryptoDataLoader:
    """Handles data loading, preprocessing, and feature engineering."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.data = None
        self.features = None
        self.targets = {}

    def fetch_crypto_data(self, symbol='BTC/USD', timeframe='1m', limit=500):
        """Fetch real cryptocurrency data using CCXT."""
        print(f"Fetching {symbol} data from Coinbase...")

        try:
            exchange = ccxt.coinbase()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(
                ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

            print(f"Successfully fetched {len(df)} candles")
            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def load_data(self, csv_file: Optional[str] = None) -> pd.DataFrame:
        """Load data from CSV file or fetch from API."""
        if csv_file and os.path.exists(csv_file):
            print(f"Loading data from {csv_file}")
            self.data = pd.read_csv(csv_file)
            self.data[self.config.timestamp_col] = pd.to_datetime(
                self.data[self.config.timestamp_col])
        else:
            print("No CSV file provided or file not found. Fetching live data...")
            self.data = self.fetch_crypto_data()

            if self.data is None:
                raise ValueError("Could not load or fetch data")

        # Ensure required columns exist
        missing_cols = [
            col for col in self.config.required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by timestamp
        self.data = self.data.sort_values(
            self.config.timestamp_col).reset_index(drop=True)

        print(f"Loaded data shape: {self.data.shape}")
        print(
            f"Date range: {self.data[self.config.timestamp_col].min()} to {self.data[self.config.timestamp_col].max()}")

        return self.data

    def engineer_features(self) -> pd.DataFrame:
        """Engineer technical indicators and other features."""
        print("Engineering features...")

        df = self.data.copy()

        # Basic price features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Position'] = (df['Close'] - df['Low']
                                ) / (df['High'] - df['Low'])

        # Technical indicators
        # Simple Moving Averages
        for window in [5, 10, 20]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Price_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}']

        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / \
            (df['BB_Upper'] - df['BB_Lower'])

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()

        # Time-based features
        df['Hour'] = df[self.config.timestamp_col].dt.hour
        df['DayOfWeek'] = df[self.config.timestamp_col].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)

        print(f"Feature engineering complete. Shape: {df.shape}")

        # Select features (exclude timestamp and target)
        feature_cols = [col for col in df.columns
                        if col not in [self.config.timestamp_col, self.config.target_col]]

        self.features = df[feature_cols].copy()
        self.data = df

        return df

    def create_targets(self) -> Dict[int, np.ndarray]:
        """Create target variables for different prediction horizons."""
        print("Creating target variables...")

        targets = {}
        for horizon in self.config.prediction_horizons:
            # Price change after 'horizon' minutes
            future_price = self.data[self.config.target_col].shift(-horizon)
            targets[horizon] = future_price.values

        # Remove last 'max_horizon' rows as they don't have targets
        max_horizon = max(self.config.prediction_horizons)
        for horizon in targets:
            targets[horizon] = targets[horizon][:-max_horizon]

        # Update features to match target length
        self.features = self.features.iloc[:-max_horizon].copy()

        self.targets = targets
        print(f"Target variables created for horizons: {list(targets.keys())}")

        return targets

    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray,
                          sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for time series models."""
        X, y = [], []

        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])

        return np.array(X), np.array(y)

    def split_data(self, horizon: int) -> Dict[str, Any]:
        """Split data into train, validation, and test sets."""
        print(f"Splitting data for {horizon}-minute horizon...")

        # Scale features
        scaled_features = self.scaler.fit_transform(self.features)
        targets = self.targets[horizon]

        # Remove NaN targets
        valid_indices = ~np.isnan(targets)
        scaled_features = scaled_features[valid_indices]
        targets = targets[valid_indices]

        # Calculate split indices
        n_samples = len(scaled_features)
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(
            n_samples * (self.config.train_ratio + self.config.val_ratio))

        # Create sequences
        X_seq, y_seq = self.prepare_sequences(
            scaled_features, targets, self.config.sequence_length)

        # Split sequences
        train_end_seq = max(0, train_end - self.config.sequence_length)
        val_end_seq = max(train_end_seq, val_end - self.config.sequence_length)

        X_train = X_seq[:train_end_seq]
        y_train = y_seq[:train_end_seq]
        X_val = X_seq[train_end_seq:val_end_seq]
        y_val = y_seq[train_end_seq:val_end_seq]
        X_test = X_seq[val_end_seq:]
        y_test = y_seq[val_end_seq:]

        # Also prepare non-sequential data for XGBoost
        X_train_flat = scaled_features[self.config.sequence_length:train_end]
        X_val_flat = scaled_features[train_end:val_end]
        X_test_flat = scaled_features[val_end:]

        y_train_flat = targets[self.config.sequence_length:train_end]
        y_val_flat = targets[train_end:val_end]
        y_test_flat = targets[val_end:]

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'X_train_flat': X_train_flat, 'y_train_flat': y_train_flat,
            'X_val_flat': X_val_flat, 'y_val_flat': y_val_flat,
            'X_test_flat': X_test_flat, 'y_test_flat': y_test_flat
        }


class LSTMModel(nn.Module):
    """Standard LSTM model for time series prediction."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last output
        out = self.fc(out)
        return out


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model."""

    def __init__(self, input_size: int, cnn_filters: int, hidden_size: int,
                 num_layers: int, output_size: int = 1):
        super(CNNLSTMModel, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(input_size, cnn_filters,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters *
                               2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # LSTM layers
        self.lstm = nn.LSTM(cnn_filters*2, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)

        # Output layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, seq_len, features = x.size()

        # CNN processing
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Back to LSTM format
        x = x.transpose(1, 2)  # (batch, seq_len, features)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Output
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    """Transformer-based time series model."""

    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_layers: int, output_size: int = 1):
        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)

        # Transformer processing
        x = self.transformer(x)

        # Take mean of sequence for prediction
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class RLAgent:
    """Simple RL agent for trading signals (placeholder implementation)."""

    def __init__(self, state_size: int, action_size: int = 3):  # Buy, Hold, Sell
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001

        # Simple neural network for Q-learning
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate)

    def get_action(self, state):
        """Get action from current policy."""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state))
            return q_values.argmax().item()

    def train(self, experiences):
        """Train the RL agent (simplified implementation)."""
        # This is a placeholder for a more comprehensive RL implementation
        pass


class ModelTrainer:
    """Handles training and evaluation of all models."""

    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.results = {}

    def train_pytorch_model(self, model, train_loader, val_loader, horizon: int, model_name: str):
        """Train a PyTorch model."""
        print(f"Training {model_name} for {horizon}-minute horizon...")

        model = model.to(self.config.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.config.device)
                    batch_y = batch_y.to(self.config.device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs.squeeze(), batch_y).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(),
                           f'{model_name}_{horizon}min_best.pth')
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            if patience_counter >= 10:  # Early stopping
                print("Early stopping triggered")
                break

        # Load best model
        model.load_state_dict(torch.load(
            f'{model_name}_{horizon}min_best.pth'))
        return model

    def train_xgboost_model(self, X_train, y_train, X_val, y_val, horizon: int):
        """Train XGBoost model."""
        print(f"Training XGBoost for {horizon}-minute horizon...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        model = xgb.train(
            params, dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        return model

    def create_ensemble_features(self, X_data, lstm_model, cnn_lstm_model, transformer_model):
        """Create features for ensemble model using predictions from other models."""
        lstm_model.eval()
        cnn_lstm_model.eval()
        transformer_model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_data).to(self.config.device)

            lstm_preds = lstm_model(X_tensor).cpu().numpy().flatten()
            cnn_lstm_preds = cnn_lstm_model(X_tensor).cpu().numpy().flatten()
            transformer_preds = transformer_model(
                X_tensor).cpu().numpy().flatten()

        # Combine original features with model predictions
        X_flat = X_data.reshape(X_data.shape[0], -1)
        ensemble_features = np.column_stack([
            X_flat, lstm_preds, cnn_lstm_preds, transformer_preds
        ])

        return ensemble_features

    def evaluate_model(self, model, X_test, y_test, model_name: str, horizon: int, is_pytorch: bool = True):
        """Evaluate model performance."""
        if is_pytorch:
            model.eval()
            with torch.no_grad():
                if isinstance(X_test, np.ndarray):
                    X_test_tensor = torch.FloatTensor(
                        X_test).to(self.config.device)
                else:
                    X_test_tensor = X_test.to(self.config.device)
                predictions = model(X_test_tensor).cpu().numpy().flatten()
        else:  # XGBoost
            if isinstance(X_test, np.ndarray):
                dtest = xgb.DMatrix(X_test)
            else:
                dtest = X_test
            predictions = model.predict(dtest)

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'predictions': predictions
        }

    def train_all_models(self, data_loader: CryptoDataLoader):
        """Train all enabled models."""
        print("Starting model training...")

        os.makedirs(self.config.output_dir, exist_ok=True)

        for horizon in self.config.prediction_horizons:
            print(f"\n{'='*50}")
            print(f"Training models for {horizon}-minute horizon")
            print(f"{'='*50}")

            # Get data splits for this horizon
            data_splits = data_loader.split_data(horizon)

            # Prepare data loaders
            X_train_tensor = torch.FloatTensor(data_splits['X_train'])
            y_train_tensor = torch.FloatTensor(data_splits['y_train'])
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True)

            X_val_tensor = torch.FloatTensor(data_splits['X_val'])
            y_val_tensor = torch.FloatTensor(data_splits['y_val'])
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False)

            input_size = data_splits['X_train'].shape[2]
            horizon_results = {}

            # Train LSTM
            if self.config.models_to_train['LSTM']:
                lstm_model = LSTMModel(
                    input_size=input_size,
                    hidden_size=self.config.lstm_hidden_size,
                    num_layers=self.config.lstm_num_layers
                )
                lstm_model = self.train_pytorch_model(
                    lstm_model, train_loader, val_loader, horizon, 'LSTM')

                # Evaluate
                lstm_results = self.evaluate_model(
                    lstm_model, data_splits['X_test'], data_splits['y_test'], 'LSTM', horizon
                )
                horizon_results['LSTM'] = lstm_results

                # Store model
                if horizon not in self.models:
                    self.models[horizon] = {}
                self.models[horizon]['LSTM'] = lstm_model

            # Train CNN-LSTM
            if self.config.models_to_train['CNN_LSTM']:
                cnn_lstm_model = CNNLSTMModel(
                    input_size=input_size,
                    cnn_filters=self.config.cnn_filters,
                    hidden_size=self.config.lstm_hidden_size,
                    num_layers=self.config.lstm_num_layers
                )
                cnn_lstm_model = self.train_pytorch_model(
                    cnn_lstm_model, train_loader, val_loader, horizon, 'CNN_LSTM')

                # Evaluate
                cnn_lstm_results = self.evaluate_model(
                    cnn_lstm_model, data_splits['X_test'], data_splits['y_test'], 'CNN_LSTM', horizon
                )
                horizon_results['CNN_LSTM'] = cnn_lstm_results

                # Store model
                self.models[horizon]['CNN_LSTM'] = cnn_lstm_model

            # Train Transformer
            if self.config.models_to_train['Transformer']:
                transformer_model = TransformerModel(
                    input_size=input_size,
                    d_model=self.config.transformer_d_model,
                    nhead=self.config.transformer_nhead,
                    num_layers=self.config.transformer_num_layers
                )
                transformer_model = self.train_pytorch_model(
                    transformer_model, train_loader, val_loader, horizon, 'Transformer')

                # Evaluate
                transformer_results = self.evaluate_model(
                    transformer_model, data_splits['X_test'], data_splits['y_test'], 'Transformer', horizon
                )
                horizon_results['Transformer'] = transformer_results

                # Store model
                self.models[horizon]['Transformer'] = transformer_model

            # Train LSTM + XGBoost Ensemble
            if self.config.models_to_train['LSTM_XGBoost']:
                if 'LSTM' in self.models[horizon] and 'CNN_LSTM' in self.models[horizon] and 'Transformer' in self.models[horizon]:
                    # Create ensemble features
                    ensemble_X_train = self.create_ensemble_features(
                        data_splits['X_train'],
                        self.models[horizon]['LSTM'],
                        self.models[horizon]['CNN_LSTM'],
                        self.models[horizon]['Transformer']
                    )
                    ensemble_X_val = self.create_ensemble_features(
                        data_splits['X_val'],
                        self.models[horizon]['LSTM'],
                        self.models[horizon]['CNN_LSTM'],
                        self.models[horizon]['Transformer']
                    )
                    ensemble_X_test = self.create_ensemble_features(
                        data_splits['X_test'],
                        self.models[horizon]['LSTM'],
                        self.models[horizon]['CNN_LSTM'],
                        self.models[horizon]['Transformer']
                    )

                    # Train XGBoost ensemble
                    xgb_ensemble = self.train_xgboost_model(
                        ensemble_X_train, data_splits['y_train'],
                        ensemble_X_val, data_splits['y_val'],
                        horizon
                    )

                    # Evaluate
                    ensemble_results = self.evaluate_model(
                        xgb_ensemble, xgb.DMatrix(
                            ensemble_X_test), data_splits['y_test'],
                        'LSTM_XGBoost', horizon, is_pytorch=False
                    )
                    horizon_results['LSTM_XGBoost'] = ensemble_results

                    # Store model
                    self.models[horizon]['LSTM_XGBoost'] = xgb_ensemble

                else:
                    print("Cannot train ensemble - not all base models available")

            # Train RL Agent (placeholder)
            if self.config.models_to_train['RL_Agent']:
                print(
                    "RL Agent training is a placeholder - implement according to your needs")
                # rl_agent = RLAgent(state_size=input_size * self.config.sequence_length)
                # # Implement RL training logic here
                # horizon_results['RL_Agent'] = {'MAE': 0, 'RMSE': 0, 'R2': 0}

            # Store results for this horizon
            self.results[horizon] = horizon_results

            # Print results summary
            print(f"\nResults for {horizon}-minute horizon:")
            print("-" * 40)
            for model_name, results in horizon_results.items():
                print(
                    f"{model_name:15} - MAE: {results['MAE']:.4f}, RMSE: {results['RMSE']:.4f}, R2: {results['R2']:.4f}")

        return self.results


class ResultsAnalyzer:
    """Analyzes and visualizes model results."""

    def __init__(self, config: Config):
        self.config = config

    def create_results_summary(self, results: Dict) -> pd.DataFrame:
        """Create a summary DataFrame of all results."""
        summary_data = []

        for horizon, horizon_results in results.items():
            for model_name, metrics in horizon_results.items():
                summary_data.append({
                    'Horizon (min)': horizon,
                    'Model': model_name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2']
                })

        return pd.DataFrame(summary_data)

    def plot_results_comparison(self, results: Dict, save_path: str = None):
        """Create comparison plots for all models and horizons."""
        summary_df = self.create_results_summary(results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            'Model Performance Comparison Across Prediction Horizons', fontsize=16)

        # MAE comparison
        pivot_mae = summary_df.pivot(
            index='Horizon (min)', columns='Model', values='MAE')
        pivot_mae.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # RMSE comparison
        pivot_rmse = summary_df.pivot(
            index='Horizon (min)', columns='Model', values='RMSE')
        pivot_rmse.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Root Mean Square Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # R2 comparison
        pivot_r2 = summary_df.pivot(
            index='Horizon (min)', columns='Model', values='R2')
        pivot_r2.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('R-squared (R²)')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Best model per horizon (based on R2)
        best_models = summary_df.loc[summary_df.groupby('Horizon (min)')[
            'R2'].idxmax()]
        axes[1, 1].bar(best_models['Horizon (min)'], best_models['R2'],
                       color=['red', 'blue', 'green', 'orange'][:len(best_models)])
        axes[1, 1].set_title('Best Model R² by Horizon')
        axes[1, 1].set_xlabel('Horizon (minutes)')
        axes[1, 1].set_ylabel('R²')

        # Add model names as labels
        for i, (horizon, r2) in enumerate(zip(best_models['Horizon (min)'], best_models['R2'])):
            axes[1, 1].text(horizon, r2 + 0.01, best_models.iloc[i]['Model'],
                            ha='center', rotation=45, fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return summary_df

    def plot_predictions_vs_actual(self, results: Dict, horizon: int, save_path: str = None):
        """Plot predictions vs actual values for a specific horizon."""
        if horizon not in results:
            print(f"No results available for {horizon}-minute horizon")
            return

        horizon_results = results[horizon]
        n_models = len(horizon_results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, (model_name, model_results) in enumerate(horizon_results.items()):
            if i >= 4:  # Only plot first 4 models
                break

            # Plot first 100 predictions
            predictions = model_results['predictions'][:100]
            actual = range(len(predictions))

            axes[i].scatter(range(len(predictions)), predictions,
                            alpha=0.6, label='Predictions')
            axes[i].plot(range(len(predictions)), predictions, alpha=0.8)
            axes[i].set_title(f'{model_name} - {horizon}min horizon')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Price')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_models, 4):
            axes[i].set_visible(False)

        plt.suptitle(
            f'Predictions for {horizon}-minute Horizon (First 100 predictions)', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_results(self, results: Dict, models: Dict, save_dir: str):
        """Save all results and models to disk."""
        os.makedirs(save_dir, exist_ok=True)

        # Save results summary
        summary_df = self.create_results_summary(results)
        summary_df.to_csv(os.path.join(
            save_dir, 'results_summary.csv'), index=False)

        # Save detailed results
        with open(os.path.join(save_dir, 'detailed_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for horizon, horizon_results in results.items():
                json_results[str(horizon)] = {}
                for model_name, metrics in horizon_results.items():
                    json_results[str(horizon)][model_name] = {
                        'MAE': float(metrics['MAE']),
                        'RMSE': float(metrics['RMSE']),
                        'R2': float(metrics['R2']),
                        'predictions': metrics['predictions'].tolist() if 'predictions' in metrics else []
                    }
            json.dump(json_results, f, indent=2)

        # Save PyTorch models
        for horizon, horizon_models in models.items():
            for model_name, model in horizon_models.items():
                if hasattr(model, 'state_dict'):  # PyTorch model
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f'{model_name}_{horizon}min.pth'))
                elif hasattr(model, 'save_model'):  # XGBoost model
                    model.save_model(os.path.join(
                        save_dir, f'{model_name}_{horizon}min.json'))

        print(f"Results and models saved to {save_dir}")


class CryptoPredictionSystem:
    """Main system class that orchestrates the entire prediction pipeline."""

    def __init__(self, csv_file: str = None):
        self.config = Config()
        if csv_file:
            self.config.csv_file = csv_file

        self.data_loader = CryptoDataLoader(self.config)
        self.trainer = ModelTrainer(self.config)
        self.analyzer = ResultsAnalyzer(self.config)

    def add_custom_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom features to the dataset.

        Override this method to add domain-specific features like:
        - Sentiment scores from news/social media
        - Market microstructure features
        - Cross-asset correlations
        - Event-based indicators

        Args:
            feature_df: DataFrame with current features

        Returns:
            DataFrame with additional features
        """
        # Example: Add some example custom features
        # In practice, you would load external data here

        # Example sentiment score (random for demonstration)
        feature_df['Sentiment_Score'] = np.random.normal(0, 1, len(feature_df))

        # Example market regime indicator
        feature_df['Market_Regime'] = (
            feature_df['Volatility'] > feature_df['Volatility'].rolling(50).mean()).astype(int)

        # Example funding rate (for crypto-specific analysis)
        feature_df['Funding_Rate'] = np.random.normal(
            0.01, 0.005, len(feature_df))

        return feature_df

    def run_experiment(self, experiment_name: str = "crypto_prediction_experiment"):
        """Run the complete prediction experiment."""
        print("=" * 60)
        print("CRYPTOCURRENCY PRICE PREDICTION SYSTEM")
        print("=" * 60)

        try:
            # Step 1: Load and prepare data
            print("\n1. Loading and preparing data...")
            self.data_loader.load_data(self.config.csv_file)
            self.data_loader.engineer_features()

            # Add custom features (can be overridden)
            self.data_loader.features = self.add_custom_features(
                self.data_loader.features)

            self.data_loader.create_targets()

            print(f"Final feature shape: {self.data_loader.features.shape}")
            print(
                f"Available features: {list(self.data_loader.features.columns)}")

            # Step 2: Train all models
            print("\n2. Training models...")
            results = self.trainer.train_all_models(self.data_loader)

            # Step 3: Analyze results
            print("\n3. Analyzing results...")

            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{self.config.output_dir}_{experiment_name}_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            # Generate summary
            summary_df = self.analyzer.create_results_summary(results)
            print("\nResults Summary:")
            print("=" * 80)
            print(summary_df.to_string(index=False))

            # Create visualizations
            print("\n4. Creating visualizations...")

            # Overall comparison
            self.analyzer.plot_results_comparison(
                results,
                save_path=os.path.join(output_dir, 'model_comparison.png')
            )

            # Predictions vs actual for each horizon
            for horizon in self.config.prediction_horizons:
                self.analyzer.plot_predictions_vs_actual(
                    results, horizon,
                    save_path=os.path.join(
                        output_dir, f'predictions_{horizon}min.png')
                )

            # Step 5: Save results
            print("\n5. Saving results...")
            self.analyzer.save_results(
                results, self.trainer.models, output_dir)

            # Print best models
            print("\nBest Models by Horizon (based on R²):")
            print("-" * 40)
            best_models = summary_df.loc[summary_df.groupby('Horizon (min)')[
                'R2'].idxmax()]
            for _, row in best_models.iterrows():
                print(
                    f"{row['Horizon (min)']}min: {row['Model']} (R²: {row['R2']:.4f})")

            print(f"\nExperiment completed! Results saved to: {output_dir}")

            return results, output_dir

        except Exception as e:
            print(f"Error during experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict_future(self, horizon: int, model_name: str = None) -> Dict:
        """
        Make predictions for future time steps.

        Args:
            horizon: Prediction horizon in minutes
            model_name: Name of model to use (if None, uses best model)

        Returns:
            Dictionary with prediction results
        """
        if horizon not in self.trainer.models:
            raise ValueError(f"No models trained for {horizon}-minute horizon")

        available_models = list(self.trainer.models[horizon].keys())

        if model_name is None:
            # Use the first available model (or implement logic to select best)
            model_name = available_models[0]
        elif model_name not in available_models:
            raise ValueError(
                f"Model {model_name} not available. Available: {available_models}")

        model = self.trainer.models[horizon][model_name]

        # Get latest data for prediction
        latest_data = self.data_loader.features.iloc[-self.config.sequence_length:].values
        latest_data_scaled = self.data_loader.scaler.transform(latest_data)

        if model_name != 'LSTM_XGBoost':
            # PyTorch model
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(
                    latest_data_scaled).unsqueeze(0).to(self.config.device)
                prediction = model(X_tensor).cpu().numpy().flatten()[0]
        else:
            # XGBoost ensemble - need base model predictions
            base_models = ['LSTM', 'CNN_LSTM', 'Transformer']
            base_predictions = []

            for base_model_name in base_models:
                if base_model_name in self.trainer.models[horizon]:
                    base_model = self.trainer.models[horizon][base_model_name]
                    base_model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(
                            latest_data_scaled).unsqueeze(0).to(self.config.device)
                        base_pred = base_model(
                            X_tensor).cpu().numpy().flatten()[0]
                        base_predictions.append(base_pred)

            # Combine with flattened features
            ensemble_features = np.concatenate([
                latest_data_scaled.flatten(),
                np.array(base_predictions)
            ]).reshape(1, -1)

            dtest = xgb.DMatrix(ensemble_features)
            prediction = model.predict(dtest)[0]

        return {
            'horizon_minutes': horizon,
            'model_used': model_name,
            'predicted_price': float(prediction),
            'timestamp': datetime.now().isoformat()
        }


# Additional utility functions for extending the system

def load_external_features(csv_path: str, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
    """
    Load external features from CSV file.

    This function can be used to load additional features like:
    - Social sentiment data
    - News sentiment scores
    - Economic indicators
    - Options flow data
    - Funding rates

    Args:
        csv_path: Path to CSV file with external features
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with external features
    """
    try:
        df = pd.read_csv(csv_path)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        return df
    except Exception as e:
        print(f"Error loading external features: {e}")
        return pd.DataFrame()


def merge_external_features(main_df: pd.DataFrame, external_df: pd.DataFrame,
                            timestamp_col: str = 'Timestamp',
                            merge_method: str = 'asof') -> pd.DataFrame:
    """
    Merge external features with main dataset.

    Args:
        main_df: Main price data DataFrame
        external_df: External features DataFrame
        timestamp_col: Timestamp column name
        merge_method: Method for merging ('asof' for time-series, 'inner', 'left', etc.)

    Returns:
        Merged DataFrame
    """
    if merge_method == 'asof':
        # Sort both DataFrames by timestamp
        main_df = main_df.sort_values(timestamp_col)
        external_df = external_df.sort_values(timestamp_col)

        # Perform as-of merge (forward-fill external features)
        merged_df = pd.merge_asof(
            main_df, external_df, on=timestamp_col, direction='backward')
    else:
        merged_df = pd.merge(main_df, external_df,
                             on=timestamp_col, how=merge_method)

    return merged_df


def create_ensemble_prediction(models_dict: Dict, X_data: np.ndarray,
                               weights: Optional[List[float]] = None) -> float:
    """
    Create ensemble prediction from multiple models.

    Args:
        models_dict: Dictionary of trained models
        X_data: Input data for prediction
        weights: Optional weights for each model (if None, uses equal weights)

    Returns:
        Ensemble prediction
    """
    predictions = []
    model_names = []

    for name, model in models_dict.items():
        if hasattr(model, 'eval'):  # PyTorch model
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_data).unsqueeze(0)
                pred = model(X_tensor).numpy().flatten()[0]
        else:  # XGBoost or sklearn model
            pred = model.predict(X_data.reshape(1, -1))[0]

        predictions.append(pred)
        model_names.append(name)

    predictions = np.array(predictions)

    if weights is None:
        weights = np.ones(len(predictions)) / len(predictions)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

    ensemble_pred = np.sum(predictions * weights)
    return float(ensemble_pred)


# Example configuration for different use cases

class QuickTestConfig(Config):
    """Configuration for quick testing with reduced parameters."""

    def __init__(self):
        super().__init__()
        self.epochs = 10
        self.sequence_length = 30
        self.prediction_horizons = [5, 10]
        self.models_to_train = {
            'LSTM': True,
            'CNN_LSTM': False,
            'Transformer': False,
            'LSTM_XGBoost': False,
            'RL_Agent': False
        }


class ProductionConfig(Config):
    """Configuration for production use with optimized parameters."""

    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.sequence_length = 120
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 3
        self.transformer_d_model = 128
        self.batch_size = 16  # 32 #64
        self.models_to_train = {
            'LSTM': True,
            'CNN_LSTM': True,
            'Transformer': True,
            'LSTM_XGBoost': True,
            'RL_Agent': False
        }


# Instructions for extending the system:
"""
EXTENDING THE SYSTEM:

1. Adding New Models:
   - Create a new model class inheriting from nn.Module (for PyTorch) or similar
   - Add training logic in ModelTrainer.train_all_models()
   - Add evaluation logic in ModelTrainer.evaluate_model()
   - Enable in Config.models_to_train

2. Adding New Features:
   - Override CryptoPredictionSystem.add_custom_features()
   - Use load_external_features() and merge_external_features() for external data
   - Add feature engineering logic in DataLoader.engineer_features()

3. Adding New Prediction Horizons:
   - Modify Config.prediction_horizons list
   - The system automatically handles multiple horizons

4. Customizing Data Sources:
   - Modify DataLoader.fetch_crypto_data() for different exchanges/symbols
   - Override DataLoader.load_data() for different data formats

5. Adding New Evaluation Metrics:
   - Extend ModelTrainer.evaluate_model()
   - Update ResultsAnalyzer for new visualizations

6. Configuration Management:
   - Create new Config subclasses for different use cases
   - Save/load configurations using pickle or JSON

Example Usage for Different Scenarios:

# Quick test
system = CryptoPredictionSystem()
system.config = QuickTestConfig()
results, output_dir = system.run_experiment("quick_test")

# With custom CSV data
system = CryptoPredictionSystem("my_data.csv")
results, output_dir = system.run_experiment("custom_data_experiment")

# Production setup
system = CryptoPredictionSystem()
system.config = ProductionConfig()
results, output_dir = system.run_experiment("production_run")
"""


# Use your own CSV file
system = CryptoPredictionSystem("./3_months_of_days_of_crypto_(1m).csv")

system.config.models_to_train = {
    'LSTM': True,
    'CNN_LSTM': True,
    'Transformer': True,
    'LSTM_XGBoost': True,
    'RL_Agent': False  # Keep disabled for now
}

# Optionally modify other configuration
system.config.epochs = 20  # 30 # Reduce for faster testing
system.config.prediction_horizons = [5, 10, 15]  # Test with fewer horizons

results, output_dir = system.run_experiment(
    "custom_experiment_for_365_days_of_1_minute_data")

if results:
    print("\nExperiment completed successfully!")

    # Example of making future predictions
    print("\nMaking sample predictions...")
    try:
        for horizon in [5, 10]:
            prediction = system.predict_future(horizon)
            print(
                f"Prediction for {horizon} minutes ahead: ${prediction['predicted_price']:.2f}")
            print(f"Model used: {prediction['model_used']}")
            print("-" * 30)
    except Exception as e:
        print(f"Prediction error: {e}")

else:
    print("Experiment failed. Check error messages above.")
