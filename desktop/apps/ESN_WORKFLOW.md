# ESN Workflow (Recording → Training → Control)

## 1. Record reference trajectory (q_des)

Use existing script (no modification required):

```bash
uv run python desktop/apps/ESN_record_trajectory.py -T 30 -v
```

Output CSV: `data/recorded_trajectory/csv/reference_trajectory_<n>.csv` including column `q_des` (or split columns).

## 2. Train ESN from recorded CSV

```bash
uv run python desktop/apps/esn_train_from_csv.py \
  --csv data/recorded_trajectory/csv/reference_trajectory_1.csv \
  --out-weight data/esn/esn_weights.npy --epochs 1
```

Adjust hyperparameters: `--n-reservoir 200 --rho 0.99 --beta 1e-5` etc.

## 3. Real-time adaptive control (generate next q online)

```bash
uv run python desktop/apps/esn_valve_control.py \
  --weight data/esn/esn_weights.npy --duration 30 --interval-ms 50
```

The script:

- Reads current encoder angle (deg→rad)
- Feeds to ESN (no delay embedding) → predicts next q
- Maps predicted angle to differential valve openings (placeholder linear mapping)
- Logs CSV to `data/esn/control_logs/`

## 4. Files Added

- `esn_train_from_csv.py`: Generic trainer from recorded CSV.
- `esn_valve_control.py`: Online ESN driven reference generator + valve actuation.
- `ESN_train.py`: Original one-circle npz trainer (kept, annotated).
- `ESN_WORKFLOW.md`: This guide.

## 5. Customize

| Aspect | Where | Note |
|--------|-------|------|
| Angle→Valve mapping | `angle_to_valve_pair()` in `esn_valve_control.py` | Replace with calibrated mapping (pressure or model-based). |
| Multi-DoF extension | same file | Expand encoder read + mapping for each joint (q is vector). |
| Safety limits | control loop | Add bounds & emergency stop (pressure / angle rate). |
| Filtering | before ESN input | Low-pass encoder noise if necessary. |
| Online adaptation | use RLS in ESN | Implement `adapt()` with live error if ground truth available. |

## 6. Next Steps / TODO Suggestions

- Implement proper encoder zero + scaling in control script.
- Replace placeholder P-style valve mapping with inverse model (pressure → torque → angle).
- Add watchdog (no encoder update → close valves).
- Add RMS / phase error metrics post-run.
- Provide option to export Wout + metadata (JSON alongside .npy).

## 7. Repro Quick Commands (Example)

```bash
REC=data/recorded_trajectory/csv/reference_trajectory_1.csv
uv run python desktop/apps/esn_train_from_csv.py --csv $REC --out-weight data/esn/esn_weights.npy
uv run python desktop/apps/esn_valve_control.py --weight data/esn/esn_weights.npy --duration 20
```

