#!/usr/bin/env python
"""Train ESN from recorded q_des trajectory CSV.（ゆうまのコードを参考）
Usage:
  uv run python desktop/apps/esn_train_from_csv.py --csv data/recorded_trajectory/csv/reference_trajectory_1.csv \
      --out-weight data/esn/esn_weights.npy

python3 desktop/apps/ESN_train.py --csv data/recorded_trajectory/csv/reference_trajectory_6.csv --out-weight data/esn/esn_weights_reference6.npy --epochs 10
Process:
  1. Load CSV (expects column 'q_des' or 'q_des_0','q_des_1', etc.)
  2. Build training pairs u[t]=q_des[t], d[t]=q_des[t+1]
  3. Train ESN (Tikhonov) once (optionally multi-epoch with state reset)
  4. Save Wout
"""
from __future__ import annotations
import argparse
import os
import csv
import numpy as np
from ESN import ESN, Tikhonov  # use desktop/apps/ESN.py


def load_qdes(path: str) -> np.ndarray:
    """Load a training series from CSV.

    Priority order:
      1. 'q_des' column (either scalar or list-string like "[a,b]")
      2. 'q_des_0', 'q_des_1', ... multiple columns
      3. fallback to 'enc_deg' (single float per row) if q_des absent

    Returns NxM numpy array (M=1 for enc_deg fallback).
    """
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError('CSV empty')
    fieldnames = reader.fieldnames or []

    # 1) q_des single column (could be '[a,b]' or 'x')
    if 'q_des' in fieldnames:
        data = []
        for r in rows:
            cell = r.get('q_des', '')
            if cell is None or cell == '':
                continue
            s = cell.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    vals = [float(x) for x in s.strip('[]').split(',')]
                except Exception:
                    continue
            else:
                try:
                    vals = [float(s)]
                except Exception:
                    continue
            data.append(vals)
        if data:
            return np.array(data, dtype=float)

    # 2) q_des_0, q_des_1, ... multi-column
    q_cols = [c for c in fieldnames if c.startswith('q_des_')]
    if q_cols:
        data = []
        for r in rows:
            try:
                vals = [float(r[c]) for c in q_cols]
            except Exception:
                continue
            data.append(vals)
        if data:
            return np.array(data, dtype=float)

    # 3) Fallback: enc_deg column (single float per row)
    if 'enc_deg' in fieldnames:
        data = []
        for r in rows:
            c = r.get('enc_deg', '')
            if c is None or c == '':
                continue
            try:
                data.append([float(c)])
            except Exception:
                continue
        if data:
            return np.array(data, dtype=float)

    raise ValueError('No usable q_des or enc_deg columns found in CSV')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Recorded trajectory CSV (contains q_des*)')
    ap.add_argument('--out-weight', default='data/esn/esn_weights.npy', help='Output weight .npy path')
    ap.add_argument('--n-reservoir', type=int, default=200)
    ap.add_argument('--input-scale', type=float, default=0.7)
    ap.add_argument('--density', type=float, default=0.1)
    ap.add_argument('--rho', type=float, default=0.99)
    ap.add_argument('--leaking-rate', type=float, default=0.7)
    ap.add_argument('--beta', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=1, help='Repeat training passes (reusing optimizer accumulation)')
    ap.add_argument('--transient', type=int, default=0, help='Discard first transient steps from optimizer accumulation')
    args = ap.parse_args()

    q = load_qdes(args.csv)
    if len(q) < 3:
        raise ValueError('Not enough samples')
    # Build train pairs
    U = q[:-1]
    D = q[1:]

    N_u = U.shape[1]
    N_y = D.shape[1]
    esn = ESN(N_u, N_y, args.n_reservoir, density=args.density, input_scale=args.input_scale,
              rho=args.rho, activation_func=np.tanh, leaking_rate=args.leaking_rate)
    opt = Tikhonov(args.n_reservoir, N_y, args.beta)

    for ep in range(args.epochs):
        print(f'Epoch {ep+1}/{args.epochs}')
        esn.reset_states()
        esn.train(U, D, opt, trans_len=args.transient)

    os.makedirs(os.path.dirname(args.out_weight), exist_ok=True)
    np.save(args.out_weight, esn.Output.Wout)
    print('Saved Wout ->', args.out_weight)


if __name__ == '__main__':
    main()
