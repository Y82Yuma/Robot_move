#!/usr/bin/env python3
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(''))
from affetto_nn_ctrl.data_handling import train_test_split_files
from affetto_nn_ctrl.model_utility import load_trained_model, load_datasets, load_train_datasets
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

model_path = 'data/optimization/esn_quick_motion/trained_model.joblib'
if not os.path.exists(model_path):
    print('MODEL_NOT_FOUND', model_path)
    sys.exit(2)
trained = load_trained_model(model_path)
print('Loaded trained model:', type(trained).__name__)
adapter = trained.adapter
# reproduce same split as used previously
train_files, test_files = train_test_split_files(['tests/data'], 0.2, 0.8, 'motion_data_*.csv', 42, shuffle=True, split_in_each_directory=False)
print('Test files:', test_files)
all_y = []
all_yp = []
for tf in test_files:
    datasets = load_datasets(tf)
    x, y = load_train_datasets(datasets, adapter)
    print(f'Loaded test dataset {tf}: x={x.shape}, y={y.shape}')
    yp = trained.predict(x)
    mse = mean_squared_error(y, yp)
    mae = mean_absolute_error(y, yp)
    print('  MSE:', mse, 'MAE:', mae)
    all_y.append(y)
    all_yp.append(yp)

if len(all_y) == 0:
    print('No test data found')
    sys.exit(3)
all_y = np.vstack(all_y)
all_yp = np.vstack(all_yp)
mse_tot = mean_squared_error(all_y, all_yp)
rmse_tot = np.sqrt(mse_tot)
mae_tot = mean_absolute_error(all_y, all_yp)
per_dim_rmse = np.sqrt(np.mean((all_y - all_yp)**2, axis=0))
print('\n=== Aggregate ===')
print('Overall MSE:', mse_tot)
print('Overall RMSE:', rmse_tot)
print('Overall MAE:', mae_tot)
print('Per-dim RMSE:', per_dim_rmse)
print('Target stats: mean std min max')
print(np.mean(all_y,axis=0), np.std(all_y,axis=0), np.min(all_y,axis=0), np.max(all_y,axis=0))
