#!/usr/bin/env python3
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(''))
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from affetto_nn_ctrl.data_handling import train_test_split_files
from affetto_nn_ctrl.model_utility import load_trained_model, load_datasets, load_train_datasets

OUT_DIR = Path('data/optimization/esn_quick_motion/plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_path = Path('data/optimization/esn_quick_motion/trained_model.joblib')
if not model_path.exists():
    print('MODEL_NOT_FOUND', model_path)
    sys.exit(2)
trained = load_trained_model(model_path)
adapter = trained.adapter

# reproduce same split
train_files, test_files = train_test_split_files(['tests/data'], 0.2, 0.8, 'motion_data_*.csv', 42, shuffle=True, split_in_each_directory=False)
print('Test files:', test_files)

for tf in test_files:
    datasets = load_datasets(tf)
    x, y = load_train_datasets(datasets, adapter)
    y_pred = trained.predict(x)
    # ensure shapes
    n = y.shape[0]
    t = np.arange(n)
    # plot per-dimension
    dims = y.shape[1]
    fig, axes = plt.subplots(dims, 1, figsize=(10, 3*dims), sharex=True)
    if dims == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, y[:, i], label='y_true')
        ax.plot(t, y_pred[:, i], label='y_pred')
        ax.set_ylabel(f'dim{i}')
        ax.legend()
    axes[-1].set_xlabel('sample')
    out_file = OUT_DIR / f'prediction_{Path(tf).stem}.png'
    fig.suptitle(f'Prediction {Path(tf).name}')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file)
    plt.close(fig)
    print('Saved', out_file)

print('Done. Plots saved to', OUT_DIR)
