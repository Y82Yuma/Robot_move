#!/usr/bin/env python
from __future__ import annotations

# Rewritten to use MyRobot hardware stack (apps/myrobot_lib) instead of affetto controller.
# Core algorithm is preserved: load reference trajectory -> generate qdes/dqdes over time ->
# real-time loop to track with PID -> log data -> repeat per reference file.
#uv run python -u apps/track_trajectory.py data/myrobot_model_MixAll/trained_model.joblib -r desktop/esn_models/tmp_motion_data_000_10s.csv -T 10 -n 1 -v --esn-model desktop/esn_models/esn_wfb0.667_tau1.222_10s.npz

import argparse
import csv
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import sys
import threading
from collections import deque
import shutil

# NOTE (2025-09-18): This runner was simplified to remove the dependency on
# `esn_ref_trajectory` adaptive generator to avoid confusing branching paths.
# It now prefers a simple ESN reference generator constructed from the output
# weights saved by `ESN_train.py` (use the CLI option `--esn-weights <path>.npy`).
# The legacy `--esn-model` / adaptive modes were intentionally removed.

# Optional lightweight ESN import (used when loading Wout .npy from ESN_train)
try:
    from ESN import ESN  # type: ignore
except Exception:
    ESN = None  # type: ignore

# Ensure project root and src are on sys.path so local packages (apps.myrobot_lib) can be imported
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / 'src'
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Also add affetto-nn-ctrl/src if present so affetto_nn_ctrl can be imported without installation
_AFFETTO_SRC = _PROJECT_ROOT / 'affetto-nn-ctrl' / 'src'
if _AFFETTO_SRC.exists() and str(_AFFETTO_SRC) not in sys.path:
    sys.path.insert(0, str(_AFFETTO_SRC))
# Also add common desktop-local source locations that may contain affetto_nn_ctrl
_candidate_srcs = [
    _PROJECT_ROOT / 'desktop' / 'ESN' / 'src',
    _PROJECT_ROOT / 'desktop' / 'src',
    _PROJECT_ROOT / 'desktop' / 'affetto' / 'src',
]
for _p in _candidate_srcs:
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)
_DESKTOP_APPS = _PROJECT_ROOT / 'desktop' / 'apps'
if _DESKTOP_APPS.exists() and str(_DESKTOP_APPS) not in sys.path:
    sys.path.insert(0, str(_DESKTOP_APPS))

# Prefer the desktop/src affetto_nn_ctrl implementation (use absolute workspace path)
_desktop_src_pref = Path('/home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/src')
if _desktop_src_pref.exists():
    _ps = str(_desktop_src_pref)
    if _ps in sys.path:
        try:
            sys.path.remove(_ps)
        except Exception:
            pass
    sys.path.insert(0, _ps)

# NOTE: esn_ref_trajectory (advanced/adaptive generator) was removed from
# this runner to avoid ambiguous branching. This script now always prefers
# a simple ESN generator constructed from ESN_train saved output weights
# (use --esn-weights <path>.npy). The legacy adaptive loader was intentionally
# removed to keep the runtime behavior predictable and aligned with the
# ESN_train -> ESN_track workflow.

# MyRobot helpers
from apps.myrobot_lib.hardware import open_devices, close_devices
from apps.myrobot_lib.controller import create_controller as create_myrobot_controller
from apps.myrobot_lib.logger import DataLogger, make_header
from apps.myrobot_lib.plotter import plot_csv
from apps.myrobot_lib import config as cfg

# --- Affetto dependency stub and model loader ---
import sys
import types
from pathlib import Path as _Path

def _ensure_affctrllib_stub() -> None:
    try:
        import affctrllib  # type: ignore
        _ = affctrllib  # noqa: F401
    except Exception:
        mod = types.ModuleType("affctrllib")
        class Logger:  # minimal stub
            def __init__(self, *a, **k):
                pass
        class Timer:  # minimal stub
            def __init__(self, *a, **k):
                pass
            def start(self):
                pass
        mod.Logger = Logger  # type: ignore[attr-defined]
        mod.Timer = Timer  # type: ignore[attr-defined]
        sys.modules["affctrllib"] = mod


def _load_trained_model(path: str):
    _ensure_affctrllib_stub()
    # If the environment's site-packages (e.g. .venv-fix) isn't on sys.path, try to add it
    try:
        # Inject local project paths so affetto_nn_ctrl is importable
        try:
            proj_root = _Path(__file__).resolve().parent.parent
            candidates = [
                proj_root / "src",
                proj_root / "affetto-nn-ctrl" / "src",
            ]
            for c in candidates:
                cs = str(c)
                if c.exists() and cs not in sys.path:
                    sys.path.insert(0, cs)
        except Exception:
            pass
        from affetto_nn_ctrl.model_utility import load_trained_model  # lazy import after stub
        return load_trained_model(path)
    except Exception as e:
        # attempt to discover .venv-fix site-packages under project root
        try:
            root = _Path.cwd()
            venv_root = root / ".venv-fix" / "lib"
            if venv_root.exists():
                for p in venv_root.glob("**/site-packages"):
                    sp = str(p)
                    if sp not in sys.path:
                        sys.path.insert(0, sp)
                        break
        except Exception:
            pass
        # retry import
        from affetto_nn_ctrl.model_utility import load_trained_model  # lazy import after stub
        return load_trained_model(path)

if TYPE_CHECKING:
    from typing import List, Tuple


DEFAULT_N_REPEAT = 10


# --- Local reference loader/interpolator (avoid pyplotutil/scipy dependency) ---
class Reference:
    def __init__(self, csv_path: Path, active_joints: list[int] | None = None, smoothness: float | None = None) -> None:  # noqa: ARG002
        self.path = Path(csv_path)
        if not self.path.exists():
            msg = f"Reference CSV not found: {csv_path}"
            raise FileNotFoundError(msg)
        # Load columns. Accept common keys:
        #  - time: 'ms' (milliseconds) or 't' (seconds)
        #  - angle: prefer 'qdes'/'q_des', else measured 'q'/'enc_deg'
        ts_ms: list[float] = []
        q_col: list[float] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # time
                    t_ms = row.get("ms")
                    if t_ms is None:
                        t_s = row.get("t")
                        t = float(t_s) if t_s not in (None, "") else 0.0
                        ts_ms.append(1000.0 * t)
                    else:
                        ts_ms.append(float(t_ms))
                    # angle
                    q_val = None
                    for key in ("qdes", "q_des", "q", "enc_deg"):
                        v = row.get(key)
                        if v not in (None, ""):
                            try:
                                q_val = float(v)
                                break
                            except Exception:
                                pass
                    if q_val is None:
                        q_val = float("nan")
                    q_col.append(q_val)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to load reference CSV: {csv_path}: {e}") from e
        # sanitize
        t_arr = np.asarray(ts_ms, dtype=float) / 1000.0
        q_arr = np.asarray(q_col, dtype=float)
        # drop NaNs conservatively
        mask = np.isfinite(t_arr) & np.isfinite(q_arr)
        t_arr = t_arr[mask]
        q_arr = q_arr[mask]
        if t_arr.size < 2:
            raise RuntimeError(f"Reference CSV too short: {csv_path}")
        # Ensure strictly increasing time
        order = np.argsort(t_arr)
        self.t = t_arr[order]
        self.q = q_arr[order]
        # Precompute dq via central differences
        dq = np.gradient(self.q, self.t, edge_order=1)
        self.dq = dq
        self.duration = float(self.t[-1] - self.t[0]) if self.t.size > 0 else 0.0
        # Build simple linear interpolators using numpy.interp
        def _interp1(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, left: float, right: float) -> np.ndarray:
            return np.interp(x, xp, fp, left=left, right=right)

        self._q_at: Callable[[float], float] = lambda tt: float(_interp1(np.array([tt]), self.t, self.q, self.q[0], self.q[-1])[0])
        self._dq_at: Callable[[float], float] = lambda tt: float(_interp1(np.array([tt]), self.t, self.dq, self.dq[0], 0.0)[0])

    def get_qdes_func(self) -> Callable[[float], np.ndarray]:
        return lambda t: np.array([self._q_at(t)], dtype=float)

    def get_dqdes_func(self) -> Callable[[float], np.ndarray]:
        return lambda t: np.array([self._dq_at(t)], dtype=float)


# --- Tracking primitive using MyRobot hardware ---

def track_motion_trajectory(
    dac,
    adc,
    enc,
    ldc_sensors,
    controller,
    reference: Reference,
    duration: float,
    data_logger: DataLogger,
    sweep_logger: DataLogger | None = None,
    header_text: str = "",
    loop_interval_ms: float = cfg.DEFAULT_LOOP_INTERVAL_MS,
    *,
    enc_ppr: int = cfg.ENCODER_PPR,
    enc_invert: bool = True,
    enc_zero_deg: float = 0.0,
    target_joint: int | None = None,
    verbose: int = 0,
    esn_generator=None,
) -> str:
    qdes_func = reference.get_qdes_func()
    dqdes_func = reference.get_dqdes_func()

    # open CSV file for this run
    csv_path = data_logger.open_file()

    # control loop
    t0 = time.perf_counter()
    next_tick = t0
    dt_prev = loop_interval_ms / 1000.0
    # debug prints limited to a few iterations
    _debug_left = 10
    # bookkeeping for ESN-produced qdes (for logging)
    qdes_esn_val = ""

    # Detect trained model (optional)
    trained_model = None
    try:
        trained_model = getattr(controller, "trained_model", None)
    except Exception:
        trained_model = None

    # If model present, prepare dimensionality helpers (match training adapter spec)
    model_active_joints = None
    target_joint_id = 0
    dof = 1
    include_tension =True
    angle_unit = "deg"  # default safety; adapter may not define this
    # NEW: defaults for adapter params used in manual 7D feature build
    include_dqdes = False
    preview_time = 0.0
    joints: list[int] = []
    if trained_model is not None:
        # Extract adapter params used in training
        try:
            params = getattr(trained_model, "adapter", None)
            params = getattr(params, "params", None)
        except Exception:
            params = None
        try:
            model_active_joints = list(getattr(params, "active_joints", [])) if params is not None else []
            if not model_active_joints:
                model_active_joints = [0]
        except Exception:
            model_active_joints = [0]
        try:
            dof = int(getattr(params, "dof", max(model_active_joints) + 1)) if params is not None else (max(model_active_joints) + 1)
        except Exception:
            dof = max(model_active_joints) + 1
        try:
            include_tension = bool(getattr(params, "include_tension", False)) if params is not None else False
        except Exception:
            include_tension = False
        # Force angle unit to degrees for both training and runtime to avoid
        # unit mismatches. Training data (enc_deg) is in degrees and runtime
        # should match that. If adapter.params claims 'rad', we ignore it and
        # print an informational warning.
        try:
            angle_unit_raw = getattr(params, "angle_unit", None) if params is not None else None
            if angle_unit_raw is not None and str(angle_unit_raw).lower().startswith('rad'):
                print(f"[WARN] adapter.params.angle_unit='{angle_unit_raw}' replaced with 'deg' to match training/runtime convention.", flush=True)
        except Exception:
            pass
        angle_unit = "deg"
        # NEW: include_dqdes/preview_time/joints for manual feature vector
        try:
            include_dqdes = bool(getattr(params, "include_dqdes", False)) if params is not None else False
        except Exception:
            include_dqdes = False
        try:
            preview_time = float(getattr(params, "preview_step", 0)) * float(getattr(params, "dt", 0.0)) if params is not None else 0.0
        except Exception:
            preview_time = 0.0
        try:
            joints = list(getattr(params, "active_joints", [])) if params is not None else []
            if not joints:
                joints = [0]
        except Exception:
            joints = [0]
        # choose target joint id (global id expected by adapter)
        try:
            if target_joint is None:
                target_joint_id = int(model_active_joints[0])
            else:
                target_joint_id = int(target_joint)
        except Exception:
            target_joint_id = int(model_active_joints[0])

        # helper: convert degrees->adapter unit (rad if requested)
        # Always treat passed angles/velocities as degrees; do not convert to rad
        def _a_angle(x: float) -> float:
            try:
                return float(x)
            except Exception:
                return float(x)
        def _a_vel(x: float) -> float:
            try:
                return float(x)
            except Exception:
                return float(x)

        # full vector length must be adapter dof
        n_full = int(dof)
        def _to_full_vec_at_index(x: float) -> np.ndarray:
            v = np.zeros((n_full,), dtype=float)
            v[target_joint_id] = float(x)
            return v
        # Reference funcs expanded to full dims with value only at target joint id (converted unit)
        def qdes_vec_func(tt: float) -> np.ndarray:
            return _to_full_vec_at_index(_a_angle(qdes_func(tt)[0]))
        def dqdes_vec_func(tt: float) -> np.ndarray:
            return _to_full_vec_at_index(_a_vel(dqdes_func(tt)[0]))
    else:
        n_full = 1
        qdes_vec_func = None  # type: ignore[assignment]
        dqdes_vec_func = None  # type: ignore[assignment]
        include_dqdes = False
        preview_time = 0.0
        joints = [0]

    # Expected feature dimension learned in training (if known) – used only for validation
    try:
        expected_n_features = getattr(controller, "_expected_n_features", None)
    except Exception:
        expected_n_features = None

    # helper to read encoder degrees using collect_data_myrobot method
    def _read_enc_deg() -> float:
        return _read_angle(enc, enc_invert, enc_zero_deg, enc_ppr)

    last_q_meas = _read_enc_deg()
    # ESN adaptive reference bookkeeping

    # If a separate sweep_logger was provided, run ROM sweep first
    if sweep_logger is not None:
        try:
            sweep_csv_path = sweep_logger.open_file()
        except Exception:
            sweep_csv_path = None
    else:
        sweep_csv_path = None

    # Always perform the sweep for hardware settling/verification, but only record if sweep_logger provided
    try:
        _perform_rom_sweep(dac, adc, enc, ldc_sensors, enc_zero_deg, enc_invert, enc_ppr, (sweep_logger if sweep_logger is not None else None), t0, verbose)
    except Exception:
        pass

    # If a sweep CSV was created, plot it
    if sweep_csv_path:
        try:
            plot_csv(sweep_csv_path)
        except Exception:
            pass

    # Re-center valves -> re-capture zero -> reset timebase
    try:
        if dac is not None:
            center_valve_pct = 60.0
            try:
                dac.set_channels(center_valve_pct, center_valve_pct)
            except Exception:
                pass
            time.sleep(1.0)
            try:
                enc_zero_new = _capture_zero(enc, enc_invert, enc_ppr)
                enc_zero_deg = enc_zero_new
                print(f"[INFO] Re-captured encoder zero after sweep: {enc_zero_deg:.3f} deg", flush=True)
            except Exception:
                pass
        # Reset time base
        t0 = time.perf_counter()
        next_tick = t0
        last_q_meas = _read_angle(enc, enc_invert, enc_zero_deg, enc_ppr)
    except Exception:
        pass
    # Start continuous encoder poller to avoid missing readings
    poller = EncoderPoller(enc, enc_invert, enc_zero_deg, enc_ppr, interval_s=0.00015, verbose=verbose)
    try:
        poller.start()
    except Exception:
        poller = None  # fallback if thread cannot start
    last_poll_ts = time.perf_counter()

    while True:
        now = time.perf_counter()
        # maintain cadence
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
            now = time.perf_counter()
        # Read encoder from continuous poller (no misses)
        if poller is not None:
            stats = poller.get_stats_since(last_poll_ts)
            if stats is not None:
                enc_min, enc_max, enc_last, last_ts = stats
                q_meas = enc_last
                last_poll_ts = last_ts
                if verbose >= 2:
                    print(f"[ENC] t={now - t0:6.3f}s enc(last)={enc_last:.6f} (min={enc_min:.3f}, max={enc_max:.3f})", flush=True)
            else:
                q_meas = _read_angle(enc, enc_invert, enc_zero_deg, enc_ppr)
        else:
            # fallback
            q_meas = _read_angle(enc, enc_invert, enc_zero_deg, enc_ppr)

        t = now - t0
        if duration > 0 and t >= duration:
            break
        next_tick += loop_interval_ms / 1000.0

        # Reference generation (ESN first if provided). CSV読み込みqdesを基礎にESNが適応生成する仕様。
        # Always capture the original CSV reference value so it can be logged separately
        try:
            q_des_csv_val = float(qdes_func(t)[0])
        except Exception:
            q_des_csv_val = float("nan")

        try:
            if esn_generator is not None and ReferenceGenerator is not None:
                # ESNは measured_q のみで適応参照を返す (内部で q->qdes 学習済み想定)。
                qdes_esn = esn_generator.step(measured_q=q_meas)
                qdes = qdes_esn
                # stash for CSV logging
                qdes_esn_val = float(qdes_esn)
                dqdes = (qdes - q_meas) / dt_prev if dt_prev > 0 else 0.0
            else:
                # use original CSV reference
                qdes = q_des_csv_val
                dqdes = float(dqdes_func(t)[0])
                qdes_esn_val = ""
        except Exception:
            # on any ESN error, fall back to original CSV reference
            qdes = q_des_csv_val
            dqdes = float(dqdes_func(t)[0])
            qdes_esn_val = ""

        # measured encoder (degrees) - moved after q_meas is determined above
        if 'q_meas' not in locals():
            q_meas = _read_enc_deg()

        # Verbose encoder confirmation (first few loops)
        if verbose:
            try:
                valve_state = f"a={a_pct:.1f}% b={b_pct:.1f}%" if 'a_pct' in locals() and 'b_pct' in locals() else "valve=unknown"
                print(f"[ENC] t={t:6.3f}s q_meas={q_meas:.6f} deg (Δq={q_meas-last_q_meas:.6f}) {valve_state}", flush=True)
            except Exception:
                pass
        dq_meas = (q_meas - last_q_meas) / dt_prev if dt_prev > 0 else 0.0
        last_q_meas = q_meas

        # ADC first so pa/pb are available to model
        adc_vals: list[float | int] = []
        pa = pb = 0.0
        if adc is not None and hasattr(adc, "read_pair"):
            try:
                raw0, volt0, kpa0, raw1, volt1, kpa1 = adc.read_pair()
                adc_vals = [raw0, volt0, kpa0, raw1, volt1, kpa1]
                pa, pb = float(kpa0), float(kpa1)
            except Exception:
                adc_vals = []
                pa = pb = 0.0

        # LDC (optional) — read before model to provide Ta/Tb
        ldc_vals: list[float] = []
        ta = tb = 0.0
        if ldc_sensors:
            for s in ldc_sensors:
                try:
                    v = float(s.read_ch0_induct_uH())
                except Exception:
                    v = float("nan")
                ldc_vals.append(v)
                try:
                    addr = getattr(s, "addr", 0)
                    if addr == 0x2A:
                        ta = float(v) if np.isfinite(v) else ta
                    elif addr == 0x2B:
                        tb = float(v) if np.isfinite(v) else tb
                except Exception:
                    pass

        # Decide valve command: model-first, PID fallback
        pid_u = None
        if trained_model is not None:
            # Convert measured angle/vel to adapter unit
            if 'angle_unit' in locals():
                q_meas_a = float(np.deg2rad(q_meas)) if angle_unit.lower().startswith('rad') else float(q_meas)
                dq_meas_a = float(np.deg2rad(dq_meas)) if angle_unit.lower().startswith('rad') else float(dq_meas)
            else:
                q_meas_a = float(q_meas)
                dq_meas_a = float(dq_meas)
            # Build states as full-size vectors (only target joint is non-zero)
            q_vec = np.zeros((n_full,), dtype=float); q_vec[target_joint_id] = q_meas_a
            dq_vec = np.zeros((n_full,), dtype=float); dq_vec[target_joint_id] = dq_meas_a
            pa_vec = np.zeros((n_full,), dtype=float); pa_vec[target_joint_id] = float(pa)
            pb_vec = np.zeros((n_full,), dtype=float); pb_vec[target_joint_id] = float(pb)
            ta_vec = np.zeros((n_full,), dtype=float); tb_vec = np.zeros((n_full,), dtype=float)
            if include_tension:
                ta_vec[target_joint_id] = float(ta)
                tb_vec[target_joint_id] = float(tb)

            # ALWAYS build manual feature vector to match training schema
            try:
                # Active joints slice
                jnts = joints if joints else [target_joint_id]
                qj = q_vec[jnts]
                dqj = dq_vec[jnts]
                paj = pa_vec[jnts]
                pbj = pb_vec[jnts]
                feats = [qj, dqj, paj, pbj]
                if include_tension:
                    taj = ta_vec[jnts]
                    tbj = tb_vec[jnts]
                    feats.extend([taj, tbj])
                # previewed references (ここではESN生成をそのまま使用。preview_timeは未使用)
                qdes_prev = np.array([qdes], dtype=float)  # 単関節仮定で直接
                feats.append(qdes_prev)
                if include_dqdes:
                    dqdes_prev = np.array([dqdes], dtype=float)
                    feats.append(dqdes_prev)
                X = np.atleast_2d(np.concatenate(feats))
                # Validate dimension
                if expected_n_features is not None and X.shape[1] != int(expected_n_features):
                    if verbose:
                        print(f"[WARN] Manual feature size mismatch: X.shape={X.shape} != expected {expected_n_features}. Falling back to PID.", flush=True)
                    raise RuntimeError("manual-feature-size-mismatch")
                if _debug_left > 0 and verbose:
                    print(f"[DEBUG] Built manual X (with tension={include_tension}) shape={X.shape}", flush=True)
            except Exception:
                # On any issue, fall back to PID
                a_pct, b_pct, pid_u = controller.update(qdes, q_meas, dt_prev, dq_des=dqdes, dq_meas=dq_meas)
                ms = int(round(t * 1000.0))
                # Preserve original CSV reference in q_des_csv column for diagnosis
                row: list[float | int | str] = [ms, a_pct, b_pct, q_des_csv_val, qdes, qdes_esn_val, (pid_u if pid_u is not None else "")]
                if adc_vals:
                    row.extend(adc_vals)
                if ldc_vals:
                    row.extend(ldc_vals)
                row.append(q_meas)
                data_logger.write_row(row)
                continue

            # Predict
            y = trained_model.predict(X)
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(1, -1)

            a_pct = b_pct = None
            if y_arr.shape[1] == 2:
                try:
                    raw_a = float(y_arr[0, 0])
                except Exception:
                    raw_a = 0.0
                try:
                    raw_b = float(y_arr[0, 1])
                except Exception:
                    raw_b = 0.0
                if _debug_left > 0 and verbose:
                    print(f"[DEBUG] Raw model outputs: raw_a={raw_a:.6f}, raw_b={raw_b:.6f}", flush=True)
                raw_a = max(0.0, min(5.0, raw_a))
                raw_b = max(0.0, min(5.0, raw_b))
                a_pct = (raw_a / 5.0) * 100.0
                b_pct = (raw_b / 5.0) * 100.0
            else:
                ca = np.zeros(n_full, dtype=float)
                cb = np.zeros(n_full, dtype=float)
                try:
                    ca, cb = trained_model.adapter.make_ctrl_input(y, {"ca": ca, "cb": cb})
                    a_pct = float(ca[target_joint_id]); b_pct = float(cb[target_joint_id])
                except Exception:
                    a_pct = 0.0; b_pct = 0.0

            try:
                min_pct = float(getattr(controller, "min_pct", 20.0))
                max_pct = float(getattr(controller, "max_pct", cfg.VALVE_MAX))
            except Exception:
                min_pct = 20.0; max_pct = float(cfg.VALVE_MAX)

            a_pct = max(min_pct, min(max_pct, float(a_pct)))
            b_pct = max(min_pct, min(max_pct, float(b_pct)))

            if _debug_left > 0 and verbose:
                try:
                    act = getattr(trained_model.adapter.params, "active_joints", None)
                except Exception:
                    act = None
                try:
                    x_shape = getattr(X, "shape", None)
                except Exception:
                    x_shape = None
                print(f"[DEBUG] active_joints={act} dof={dof} include_tension={include_tension} angle_unit={angle_unit} X_shape={x_shape} y_shape={getattr(y,'shape',None)}", flush=True)
                try:
                    print(f"[DEBUG] y_sample={np.asarray(y).ravel()[:8]}", flush=True)
                    if y_arr.shape[1] == 2:
                        print(f"[DEBUG] final a_pct={a_pct:.3f} b_pct={b_pct:.3f} (raw scaled)", flush=True)
                    print(f"[DEBUG] q_des={qdes:.3f} q_meas={q_meas:.3f} error={qdes-q_meas:.3f} deg", flush=True)
                except Exception:
                    pass
                _debug_left -= 1
        else:
            a_pct, b_pct, pid_u = controller.update(qdes, q_meas, dt_prev, dq_des=dqdes, dq_meas=dq_meas)

        # send to hardware
        try:
            if dac is not None and hasattr(dac, "set_channels"):
                try:
                    a_pct_send = float(a_pct)
                except Exception:
                    a_pct_send = 0.0
                try:
                    b_pct_send = float(b_pct)
                except Exception:
                    b_pct_send = 0.0
                a_pct_send = max(0.0, min(100.0, a_pct_send))
                b_pct_send = max(0.0, min(100.0, b_pct_send))
                try:
                    pct_to_code = getattr(dac, "_pct_to_code", None)
                    if callable(pct_to_code):
                        a_code = int(pct_to_code(a_pct_send))
                        b_code = int(pct_to_code(b_pct_send))
                        if verbose:
                            print(f"[DEBUG] DAC codes -> a_code={a_code} b_code={b_code} (from a_pct={a_pct_send:.3f} b_pct={b_pct_send:.3f})", flush=True)
                except Exception:
                    pass
                dac.set_channels(a_pct_send, b_pct_send)
        except Exception:
            pass

        # log
        ms = int(round(t * 1000.0))
        # Preserve original CSV reference in q_des_csv column for diagnosis
        row: list[float | int | str] = [ms, a_pct, b_pct, q_des_csv_val, qdes, qdes_esn_val, (pid_u if pid_u is not None else "")] 
        if adc_vals:
            row.extend(adc_vals)
        if ldc_vals:
            row.extend(ldc_vals)
        row.append(q_meas)
        data_logger.write_row(row)

    # Stop poller before returning
    try:
        if poller is not None:
            poller.stop()
    except Exception:
        pass

    return csv_path


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a reference trajectory using MyRobot hardware (model-first; PID fallback).",
    )
    parser.add_argument(
        "model",
        help="Path to trained model (.joblib)",
    )
    parser.add_argument(
        "-r",
        "--reference-files",
        nargs="+",
        required=True,
        help="Path(s) to reference trajectory CSV(s).",
    )
    parser.add_argument(
        "-T",
        "--duration",
        type=float,
        help="Time duration to perform trajectory tracking. If omitted, use reference duration.",
    )
    parser.add_argument(
        "-n",
        "--n-repeat",
        default=DEFAULT_N_REPEAT,
        type=int,
        help="Number of iterations to track each reference trajectory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/myrobot/track",
        help="Output directory to store tracked motion files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tracked_trajectory",
        help="Filename prefix that will be added to tracked motion files.",
    )
    parser.add_argument("--kp", type=float, default=cfg.DEFAULT_KP)
    parser.add_argument("--ki", type=float, default=cfg.DEFAULT_KI)
    parser.add_argument("--kd", type=float, default=cfg.DEFAULT_KD)
    parser.add_argument("--center", type=float, default=cfg.VALVE_CENTER)
    parser.add_argument("--span", type=float, default=cfg.VALVE_SPAN)
    parser.add_argument("--min-valve", type=float, default=20.0, help="Minimum valve percent (default 20)")
    parser.add_argument("--max-valve", type=float, default=cfg.VALVE_MAX)
    parser.add_argument("--esn-model", type=str, default=None, help="ESN参照生成モデル(.npz)。指定時はESNのq->qdes適応参照を使用")
    parser.add_argument(
        "--esn-weights",
        type=str,
        default=None,
        help="Path to ESN output weight .npy saved by ESN_train.py (optional). If provided, a simple ESN-based q->qdes generator will be used.",
    )
    parser.add_argument(
        "--esn-alpha",
        type=float,
        default=0.0,
        help="Optional alpha to add measured_q * alpha to ESN output (default 0.0)",
    )
    parser.add_argument("--ppr", type=int, default=cfg.ENCODER_PPR, help="Encoder PPR per channel")
    parser.add_argument("--zero-at-start", dest="zero_at_start", action="store_true", default=True, help="Capture encoder zero at start")
    parser.add_argument("--no-zero-at-start", dest="zero_at_start", action="store_false")
    parser.add_argument("--zero-deg", type=float, default=None, help="Explicit encoder zero offset (deg)")
    parser.add_argument("--encoder-invert", dest="encoder_invert", action="store_true", default=True, help="Invert encoder sign")
    parser.add_argument("--no-encoder-invert", dest="encoder_invert", action="store_false")
    parser.add_argument("--target-joint", type=int, default=None, help="Joint id in adapter.active_joints that corresponds to this hardware joint")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose console output.",
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=cfg.DEFAULT_LOOP_INTERVAL_MS,
        help="Loop interval in milliseconds (dt = interval_ms/1000). ESNメタdt存在時は自動上書き。",
    )
    return parser.parse_args()


# --- Encoder utility functions (matching collect_data_myrobot.py exactly) ---
def _capture_zero(enc, invert: bool, ppr: int) -> float:
    """Sample encoder briefly and return averaged angle (after optional invert) as zero offset."""
    if enc is None:
        return 0.0
    t0 = time.perf_counter()
    samples: list[float] = []
    while time.perf_counter() - t0 < 0.08:
        try:
            enc.poll()
            a = enc.degrees(ppr)
            if invert:
                a = -a
            samples.append(a)
        except Exception:
            pass
        time.sleep(0.002)
    if not samples:
        try:
            enc.poll()
            a = enc.degrees(ppr)
            if invert:
                a = -a
            return a
        except Exception:
            return 0.0
    return sum(samples) / len(samples)


def _read_angle(enc, invert: bool, zero_deg: float, ppr: int) -> float:
    """Read current encoder angle, apply invert and zero offset."""
    a = 0.0
    if enc is not None:
        try:
            enc.poll()
            a = enc.degrees(ppr)
        except Exception:
            a = 0.0
    if invert:
        a = -a
    return a - (zero_deg if zero_deg is not None else 0.0)


class EncoderPoller:
    """Background encoder poller to avoid missing readings between frames.

    Polls enc at high frequency (~150us), applies invert and zero, and stores
    timestamped samples in a ring buffer. The main loop can then fetch the
    most recent value (and min/max) since the last frame.
    """
    def __init__(self, enc, invert: bool, zero_deg: float | None, ppr: int, interval_s: float = 0.00015, maxlen: int = 20000, verbose: int = 0) -> None:
        self.enc = enc
        self.invert = bool(invert)
        self.zero = float(zero_deg) if zero_deg is not None else 0.0
        self.ppr = int(ppr)
        self.interval_s = float(interval_s)
        self.verbose = int(verbose)
        self._buf: deque[tuple[float, float]] = deque(maxlen=int(maxlen))
        self._last: tuple[float, float] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._th: threading.Thread | None = None

    def update_zero(self, zero_deg: float | None) -> None:
        self.zero = float(zero_deg) if zero_deg is not None else 0.0

    def _read_once(self) -> float | None:
        if self.enc is None:
            return None
        try:
            self.enc.poll()
            a = self.enc.degrees(self.ppr)
        except Exception:
            return None
        if self.invert:
            a = -a
        return a - self.zero

    def _loop(self) -> None:
        while self._running:
            ts = time.perf_counter()
            v = self._read_once()
            if v is not None:
                with self._lock:
                    self._buf.append((ts, v))
                    self._last = (ts, v)
                    if self.verbose >= 3:
                        print(f"[ENC-POLLER] ts={ts:.6f} v={v:.3f}", flush=True)
            # keep cadence
            try:
                time.sleep(self.interval_s)
            except Exception:
                pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._loop, name="EncoderPoller", daemon=True)
        self._th.start()

    def stop(self, timeout: float | None = 1.0) -> None:
        self._running = False
        th = self._th
        if th is not None:
            try:
                th.join(timeout=timeout)
            except Exception:
                pass
            self._th = None

    def get_stats_since(self, since_ts: float) -> tuple[float, float, float, float] | None:
        """Return (min, max, last_value, last_ts) for samples newer than since_ts.
        If no new samples are available, return the latest sample if present.
        """
        with self._lock:
            if not self._buf:
                return None
            # collect samples newer than since_ts
            vals: list[tuple[float, float]] = [item for item in self._buf if item[0] > since_ts]
            if not vals:
                # no fresh sample, return the most recent
                last_ts, last_v = self._buf[-1]
                return (last_v, last_v, last_v, last_ts)
            ts_list = [ts for ts, _ in vals]
            v_list = [v for _, v in vals]
            vmin = float(min(v_list))
            vmax = float(max(v_list))
            last_ts = float(ts_list[-1])
            last_v = float(v_list[-1])
            return (vmin, vmax, last_v, last_ts)


def _perform_rom_sweep(dac, adc, enc, ldc_sensors, enc_zero: float, enc_invert: bool, enc_ppr: int, data_logger: DataLogger | None, t0: float, verbose: int = 0) -> None:
    """Perform range-of-motion sweep and record every step into the provided DataLogger.

    This function polls the encoder at high frequency (matches ROM sweep semantics)
    and writes one CSV row per step into the provided DataLogger if provided.
    If data_logger is None the sweep still executes (valve commands & polling)
    but no CSV rows are written or plotted.
    """
    if dac is None or enc is None:
        return

    print("[INFO] Performing range-of-motion sweep: 60/60 -> 20/100 -> 100/20 -> 60/60", flush=True)

    # Sweep sequence (unchanged)
    ramps = [((60.0, 60.0), (20.0, 100.0)), ((20.0, 100.0), (100.0, 20.0)), ((100.0, 20.0), (60.0, 60.0))]

    # Timing: 5s total, 50ms steps
    total_sweep_time = 5.0
    step_dt = 0.05
    per_ramp_time = max(0.5, total_sweep_time / len(ramps))

    for ramp_idx, ((start_a, start_b), (end_a, end_b)) in enumerate(ramps):
        steps = max(1, int(per_ramp_time / step_dt))
        for s in range(steps + 1):
            frac = float(s) / float(steps)
            a_pct = start_a + (end_a - start_a) * frac
            b_pct = start_b + (end_b - start_b) * frac

            # Set valve command
            try:
                dac.set_channels(a_pct, b_pct)
            except Exception:
                pass

            # High-frequency polling during this step — keep encoder polled but DO NOT
            # accumulate min/max. Capture the last reading for logging.
            step_start = time.perf_counter()
            poll_until = step_start + step_dt
            last_v = None
            while time.perf_counter() < poll_until:
                try:
                    enc.poll()
                    a = enc.degrees(enc_ppr)
                    if enc_invert:
                        a = -a
                    a = a - (enc_zero if enc_zero is not None else 0.0)
                    last_v = a
                    if verbose >= 2 and last_v is not None:
                        print(f"[SWEEP-POLL] a={a_pct:.1f}% b={b_pct:.1f}% enc_deg={last_v:.3f}", flush=True)
                except Exception:
                    pass
                time.sleep(0.00015)  # ~150µs polling interval

            # Read ADC once for this step (if available)
            adc_vals: list[float | int] = []
            pa = pb = 0.0
            if adc is not None and hasattr(adc, "read_pair"):
                try:
                    raw0, volt0, kpa0, raw1, volt1, kpa1 = adc.read_pair()
                    adc_vals = [raw0, volt0, kpa0, raw1, volt1, kpa1]
                    pa, pb = float(kpa0), float(kpa1)
                except Exception:
                    adc_vals = []

            # Read LDC sensors once for this step (if available)
            ldc_vals: list[float] = []
            if ldc_sensors:
                for sdev in ldc_sensors:
                    try:
                        v = float(sdev.read_ch0_induct_uH())
                    except Exception:
                        v = float("nan")
                    ldc_vals.append(v)

            # Build CSV row consistent with main loop format only if a data_logger was provided:
            # [ms, a_pct, b_pct, q_des_csv, qdes, qdes_esn, pid_u, <adc vals...>, <ldc vals...>, enc_deg]
            ms = int(round((time.perf_counter() - t0) * 1000.0))
            # include placeholders for q_des_csv, qdes, qdes_esn and pid
            row: list[float | int | str] = [ms, a_pct, b_pct, "", "", "", ""]
            if adc_vals:
                row.extend(adc_vals)
            if ldc_vals:
                row.extend(ldc_vals)
            row.append(last_v if last_v is not None else "")

            # Write only if a DataLogger was provided (user requested recording)
            if data_logger is not None:
                try:
                    data_logger.write_row(row)
                except Exception:
                    pass

            # Step summary (no min/max)
            if verbose >= 1:
                if last_v is not None:
                    print(f"[SWEEP] ramp{ramp_idx+1} step{s}/{steps} a={a_pct:.1f}% b={b_pct:.1f}% enc={last_v:.3f}", flush=True)
                else:
                    print(f"[SWEEP] ramp{ramp_idx+1} step{s}/{steps} a={a_pct:.1f}% b={b_pct:.1f}% enc=none", flush=True)

            # Ensure step timing
            remaining = poll_until - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    # Intentionally DO NOT return valves to center here. Leave the last
    # commanded valve percentages in place so the main loop continues from
    # the sweep endpoint without changing air pressure.
    return


def main() -> None:
    args = parse()

    # Ensure project root and src are on sys.path so local packages can be imported
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _SRC_PATH = _PROJECT_ROOT / 'src'
    if str(_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(_SRC_PATH))
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    # Try adding affetto-nn-ctrl/src so affetto_nn_ctrl is importable without installation
    _AFFETTO_SRC = _PROJECT_ROOT / 'affetto-nn-ctrl' / 'src'
    if _AFFETTO_SRC.exists() and str(_AFFETTO_SRC) not in sys.path:
        sys.path.insert(0, str(_AFFETTO_SRC))
    # Also add common desktop-local source locations that may contain affetto_nn_ctrl
    _candidate_srcs = [
        _PROJECT_ROOT / 'desktop' / 'ESN' / 'src',
        _PROJECT_ROOT / 'desktop' / 'src',
        _PROJECT_ROOT / 'desktop' / 'affetto' / 'src',
    ]
    for _p in _candidate_srcs:
        _ps = str(_p)
        if _p.exists() and _ps not in sys.path:
            sys.path.insert(0, _ps)
    # Ensure desktop/apps is on sys.path so esn_ref_trajectory can be found there
    _DESKTOP_APPS = _PROJECT_ROOT / 'desktop' / 'apps'
    if _DESKTOP_APPS.exists() and str(_DESKTOP_APPS) not in sys.path:
        sys.path.insert(0, str(_DESKTOP_APPS))

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open hardware
    dac, adc, enc, ldc_sensors = open_devices(i2c_bus=cfg.I2C_BUS, ldc_addrs=cfg.LDC_ADDRS)
    print("[INFO] DAC opened", flush=True)
    if adc is not None:
        print("[INFO] ADC opened", flush=True)
    if enc is not None:
        print("[INFO] Encoder enabled", flush=True)
    if ldc_sensors:
        addrs = ",".join(f"0x{getattr(s, 'addr', 0):02X}" for s in ldc_sensors)
        print(f"[INFO] LDC sensors: {len(ldc_sensors)} ({addrs})", flush=True)

    # Build controller
    ctrl = create_myrobot_controller(
        dac,
        enc,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        center=args.center,
        span_pct=args.span,
        min_pct=args.min_valve,
        max_pct=args.max_valve,
    )

    # --- Startup sequence following collect_data_myrobot pattern ---
    enc_zero = args.zero_deg
    try:
        # Set initial valves to center (60%) and settle
        if dac is not None:
            center_valve_pct = 60.0
            dac.set_channels(center_valve_pct, center_valve_pct)
            print(f"[INFO] Initial valves set to {center_valve_pct:.1f}% for encoder zero capture", flush=True)
        time.sleep(1.0)
        
        # Capture encoder zero using collect_data_myrobot method
        if args.zero_at_start and enc is not None and enc_zero is None:
            try:
                enc_zero = _capture_zero(enc, args.encoder_invert, args.ppr)
                print(f"[INFO] Captured encoder zero: {enc_zero:.3f} deg", flush=True)
            except Exception:
                enc_zero = 0.0
        if enc_zero is None:
            enc_zero = 0.0
        
        # NOTE: ROM sweep execution is performed per-reference inside track_motion_trajectory.
        # Do not run the sweep here at startup to avoid duplicate sweeps and to keep
        # sweep-recording optional.
        
        # Small settle after potential startup actions
        time.sleep(0.5)
    
    except Exception:
        if enc_zero is None:
            enc_zero = 0.0

    # Load trained model and attach to controller (for use inside loop)
    trained_model = None
    expected_n_features = None
    try:
        trained_model = _load_trained_model(args.model)
        # Reset adapter internal state if available (for delay/preview features)
        try:
            if hasattr(trained_model.adapter, 'reset'):
                trained_model.adapter.reset()
        except Exception:
            pass
        setattr(ctrl, "trained_model", trained_model)
        try:
            aj = getattr(trained_model.adapter.params, "active_joints", None)
            dof = getattr(trained_model.adapter.params, "dof", None)
            inc_t = getattr(trained_model.adapter.params, "include_tension", None)
            ang_u = getattr(trained_model.adapter.params, "angle_unit", None)
        except Exception:
            aj = dof = inc_t = ang_u = None
        # Inspect pipeline expected input feature size (if available)
        try:
            from sklearn.pipeline import Pipeline  # type: ignore
            pipe = getattr(trained_model, "model", None)
            if isinstance(pipe, Pipeline):
                scaler = pipe.steps[0][1]
                expected_n_features = getattr(scaler, "n_features_in_", None)
        except Exception:
            expected_n_features = None
        # Keep expected feature size on controller for use in loop
        try:
            setattr(ctrl, "_expected_n_features", expected_n_features)
        except Exception:
            pass
        print(
            f"[INFO] Model loaded: {getattr(trained_model, 'model_name', type(trained_model).__name__)} (active_joints={aj}, dof={dof}, include_tension={inc_t}, angle_unit={ang_u}, expected_n_features={expected_n_features})",
            flush=True,
        )
    except Exception as e:
        print(f"[WARN] Failed to load model '{args.model}': {e}. Using PID fallback.", flush=True)

    # ESN generator (adaptive). Reset/history不要。CSV参照は基礎データとして読み込むがESNは measured q から直接 qdes を生成。
    esn_generator = None
    interval_ms_effective = float(args.interval_ms)
    # Ensure we refer to the module-level symbols when retrying import
    global load_esn_model, ReferenceGenerator, ReferenceGeneratorAdaptive
    # If user provided an ESN weights file (.npy), create a simple ESN generator from it
    if getattr(args, 'esn_weights', None):
        esn_w_path = Path(args.esn_weights)
        try:
            Wout = np.load(esn_w_path)
            if ESN is None:
                print(f"[WARN] ESN class not importable; cannot use --esn-weights {esn_w_path}", flush=True)
            else:
                # Build a minimal generator that uses ESN(Input->Reservoir->Wout) and optionally adds alpha*measured_q
                class SimpleESNGenerator:
                    def __init__(self, Wout, N_u=1, N_x=None, alpha=0.0, **esn_kwargs):
                        self.Wout = np.asarray(Wout)
                        # infer N_x if not given
                        if N_x is None:
                            self.N_y, self.N_x = self.Wout.shape
                        else:
                            self.N_x = int(N_x)
                            self.N_y = int(self.Wout.shape[0])
                        self.esn = ESN(N_u, self.N_y, self.N_x, **esn_kwargs)
                        # overwrite output weights
                        try:
                            self.esn.Output.Wout = self.Wout
                        except Exception:
                            pass
                        # Reset internal ESN state so runtime sequence starts from a
                        # clean reservoir (matches ESN_train usage where states are
                        # reset before accumulation). The generator will then
                        # ingest measured q sequentially, producing the same
                        # one-step prediction y = Wout @ x used during training.
                        try:
                            self.esn.reset_states()
                        except Exception:
                            try:
                                self.esn.reset()
                            except Exception:
                                pass
                        # alpha is intentionally ignored to maintain exact parity
                        # with ESN_train one-step prediction. Warn if non-zero.
                        self._alpha = float(alpha)
                        if abs(self._alpha) > 1e-12:
                            print(f"[WARN] --esn-alpha provided ({self._alpha}) but ignored: runtime ESN uses pure Wout@x one-step prediction to match training.", flush=True)

                    def step(self, measured_q: float) -> float:
                        # measured_q is assumed scalar; keep units consistent (deg)
                        # Feed the measured q as the ESN input and return the
                        # one-step-ahead prediction identical to training: y = Wout @ x
                        u = np.array([float(measured_q)])
                        x_in = self.esn.Input(u)
                        x = self.esn.Reservoir(x_in)
                        y = np.dot(self.esn.Output.Wout, x)
                        out = float(y[0])
                        return out

                try:
                    # infer N_x from Wout shape
                    ny, nx = Wout.shape
                except Exception:
                    ny = None
                    nx = None

                # Basic validation: Wout should be 2D and have at least one column
                if Wout.ndim != 2 or nx is None or nx < 1:
                    print(f"[ERROR] Invalid ESN weights file shape: {getattr(Wout, 'shape', None)}. Expected (N_y, N_x) with N_x>=1.", flush=True)
                else:
                    # Create generator with inferred reservoir size; if mismatch happens later the ESN class will raise
                    try:
                        esn_generator = SimpleESNGenerator(
                            Wout,
                            N_u=1,
                            N_x=nx,
                            alpha=float(getattr(args, 'esn_alpha', 0.0)),
                            density=0.1,
                            input_scale=0.7,
                            rho=0.99,
                            leaking_rate=0.7,
                        )
                        print(f"[INFO] Simple ESN generator initialized from {esn_w_path} (alpha={getattr(args,'esn_alpha',0.0)})", flush=True)
                    except Exception as ee:
                        print(f"[ERROR] Failed to initialize SimpleESNGenerator from Wout shape {Wout.shape}: {ee}", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to initialize ESN generator from weights {esn_w_path}: {e}", flush=True)
    # NOTE: Legacy adaptive ESN loader and --esn-model handling removed.
    # The runner now only supports a simple ESN instance built from the
    # output weights saved by ESN_train.py via --esn-weights <path>.npy.

    ref_paths = [Path(p) for p in args.reference_files]
    for i, ref_path in enumerate(ref_paths, start=1):
        ref = Reference(ref_path)
        duration = float(args.duration) if args.duration is not None else float(ref.duration)
        ref_dir = output_dir / f"reference_{i:03d}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        base_dest = Path('/home/hosodalab2/Desktop/MyRobot/data/tracked_trajectory')
        csv_dest = base_dest / 'csv'
        graph_dest = base_dest / 'graph'
        csv_dest.mkdir(parents=True, exist_ok=True)
        graph_dest.mkdir(parents=True, exist_ok=True)
        header = make_header(has_adc=adc is not None, ldc_addrs=[getattr(s, 'addr', 0) for s in ldc_sensors], has_enc=enc is not None)
        # Append provenance metadata fields so every tracked CSV records how it was produced
        try:
            meta_fields = [
                f"esn_weights={str(Path(args.esn_weights))}" if getattr(args, 'esn_weights', None) else "esn_weights=",
                f"esn_alpha={float(getattr(args, 'esn_alpha', 0.0))}",
                f"angle_unit=deg",
            ]
            # store metadata as empty header columns (will be placed after existing header)
            for mf in meta_fields:
                if isinstance(header, list):
                    if mf not in header:
                        header.append(mf)
        except Exception:
            pass
        # Ensure CSV always contains columns for the original CSV reference (q_des_csv)
        # and the ESN-generated reference (qdes_esn). This keeps rows and header aligned
        # even when --esn-model is not provided or import was delayed.
        try:
            if isinstance(header, list):
                # If already present, avoid duplicates
                if "qdes_esn" not in header:
                    # Find an existing angle-like column if any
                    angle_keys = ("q_des", "qdes", "q_des_csv")
                    angle_idx = None
                    for k in angle_keys:
                        if k in header:
                            angle_idx = header.index(k)
                            break
                    # Default insertion position after valve columns if none found
                    if angle_idx is None:
                        angle_idx = 3
                        # guard list length
                        if angle_idx > len(header):
                            angle_idx = len(header)
                            header.extend([""] * (angle_idx - len(header)))
                    # Replace the existing angle column name with q_des_csv
                    try:
                        header[angle_idx] = "q_des_csv"
                    except Exception:
                        # extend if necessary
                        if angle_idx >= len(header):
                            header.insert(angle_idx, "q_des_csv")
                    # Ensure 'qdes' follows q_des_csv (represents the runtime qdes column)
                    insert_pos = angle_idx + 1
                    if not (insert_pos < len(header) and header[insert_pos] == "qdes"):
                        header.insert(insert_pos, "qdes")
                    # Ensure qdes_esn is immediately after qdes
                    if not (insert_pos + 1 < len(header) and header[insert_pos + 1] == "qdes_esn"):
                        header.insert(insert_pos + 1, "qdes_esn")
        except Exception:
            pass
        def _next_idx(d: Path) -> int:
            existing = sorted([p for p in d.glob(f"{args.output_prefix}_*.csv")])
            max_idx = 0
            for p in existing:
                try:
                    stem = p.stem
                    suffix = stem.rsplit("_", 1)[-1]
                    max_idx = max(max_idx, int(suffix))
                except Exception:
                    pass
            return max_idx + 1
        base_idx = _next_idx(csv_dest)
        for j in range(args.n_repeat):
            try:
                ctrl.reset()
            except Exception:
                pass
            current_idx = base_idx + j
            output_name = f"{args.output_prefix}_{current_idx}"
            logger = DataLogger(str(csv_dest), output_name, header)
            sweep_logger = None
            header_text = f"[Ref:{i}/{len(ref_paths)}(Cnt:{j + 1}/{args.n_repeat})] Tracking..."
            csv_path = track_motion_trajectory(
                dac,
                adc,
                enc,
                ldc_sensors,
                ctrl,
                ref,
                duration,
                logger,
                sweep_logger,
                header_text=header_text,
                loop_interval_ms=float(interval_ms_effective),
                enc_ppr=int(args.ppr),
                enc_invert=bool(args.encoder_invert),
                enc_zero_deg=float(enc_zero or 0.0),
                target_joint=(None if args.target_joint is None else int(args.target_joint)),
                verbose=int(args.verbose),
                esn_generator=esn_generator,
            )
            print(f"[INFO] Motion file saved: {csv_path}")
            try:
                try:
                    import os as _os
                    model_path = Path(args.model)
                    model_folder = model_path.parent.name if model_path.parent.name else model_path.stem
                    _os.environ['PLOT_TITLE'] = f"Model: {model_folder}/{model_path.name}  Ref: {ref_path.name}"
                except Exception:
                    pass
                plot_csv(csv_path)
            except Exception:
                pass
            try:
                src_csv = Path(csv_path)
                try:
                    png_src = src_csv.with_suffix('.png')
                    if png_src.exists():
                        shutil.copy2(str(png_src), str(graph_dest / png_src.name))
                        print(f"[INFO] Plot saved: {graph_dest / png_src.name}")
                        png_src.unlink()
                        print(f"[INFO] Removed original PNG from CSV folder: {png_src}")
                except Exception as e:
                    print(f"[WARN] Failed to copy/remove PNG: {e}")
            except Exception:
                pass
            # ESN 比較プロット生成
            try:
                plot_esn_vs_csv(csv_path)
            except Exception:
                pass

    # Release/cleanup
    close_devices(dac, adc, enc, ldc_sensors)
    print("[INFO] Finished trajectory tracking", flush=True)

def plot_esn_vs_csv(csv_path: str) -> Path | None:
    """Generate comparison plot: measured q, q_des (CSV), qdes_esn and their difference.
    Uses a non-interactive matplotlib backend so it works in headless environments.
    Saves two PNGs next to CSV:
      - <csv>.png -> main plot where 'qdes' is replaced by qdes_esn when available
      - <csv>.esn_compare.png -> comparison plot with q, q_des_csv, qdes_esn and correction
    Returns the Path to the main PNG or None on failure.
    """
    try:
        from pathlib import Path as _P
        import matplotlib as _mpl
        _mpl.use('Agg')
        import matplotlib.pyplot as _plt
        path = _P(csv_path)
        if not path.exists():
            return None

        times: list[float] = []
        q_meas: list[float] = []
        q_des_csv: list[float] = []
        q_des_esn: list[float] = []

        # Read CSV robustly
        with path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # time
                t_ms = row.get('ms') or row.get('t')
                t = None
                try:
                    if t_ms is None or t_ms == '':
                        t = None
                    else:
                        # if provided in seconds in 't', treat appropriately (assume ms if ms present)
                        if row.get('ms') is not None:
                            t = float(row.get('ms')) / 1000.0
                        else:
                            t = float(row.get('t'))
                except Exception:
                    t = None
                if t is None:
                    # approximate from row count
                    t = float(len(times)) * 0.001
                times.append(float(t))

                # q measured: prefer 'q_meas' or 'q' or 'enc_deg' or last column
                qm = None
                for key in ('q_meas', 'q', 'enc_deg'):
                    v = row.get(key)
                    if v not in (None, ''):
                        try:
                            qm = float(v)
                            break
                        except Exception:
                            pass
                # fallback: look at last column value
                if qm is None:
                    try:
                        last_vals = list(row.values())
                        if last_vals:
                            try:
                                qm = float(last_vals[-1])
                            except Exception:
                                qm = float('nan')
                        else:
                            qm = float('nan')
                    except Exception:
                        qm = float('nan')
                q_meas.append(float(qm))

                # q_des from CSV original (q_des_csv)
                qdc = None
                for key in ('q_des_csv', 'q_des', 'qdes'):
                    v = row.get(key)
                    if v not in (None, ''):
                        try:
                            qdc = float(v)
                            break
                        except Exception:
                            pass
                q_des_csv.append(float(qdc) if qdc is not None else float('nan'))

                # qdes_esn
                qde = None
                for key in ('qdes_esn', 'qdes_esn_val', 'qdes_esn_val'):
                    v = row.get(key)
                    if v not in (None, ''):
                        try:
                            qde = float(v)
                            break
                        except Exception:
                            pass
                q_des_esn.append(float(qde) if qde is not None else float('nan'))

        # Convert to numpy for plotting convenience
        t_arr = np.asarray(times, dtype=float)
        q_arr = np.asarray(q_meas, dtype=float)
        qcsv_arr = np.asarray(q_des_csv, dtype=float)
        qesn_arr = np.asarray(q_des_esn, dtype=float)

        # Main PNG: plot measured q and qdes_esn as 'qdes' (fallback to qcsv if no esn)
        main_png = path.with_suffix('.png')
        try:
            _plt.figure(figsize=(10, 4))
            _plt.plot(t_arr, q_arr, label='q (measured)', color='tab:blue')
            # choose plotted qdes: prefer esn values if available
            if np.isfinite(qesn_arr).any():
                _plt.plot(t_arr, qesn_arr, label='qdes (ESN)', color='tab:orange')
            else:
                _plt.plot(t_arr, qcsv_arr, label='qdes (CSV)', color='tab:orange')
            _plt.xlabel('time (s)')
            _plt.ylabel('angle (deg)')
            _plt.grid(True)
            _plt.legend()
            _plt.tight_layout()
            _plt.savefig(str(main_png), dpi=150)
            _plt.close()
        except Exception:
            try:
                _plt.close()
            except Exception:
                pass

        # Comparison PNG: q, q_des_csv, qdes_esn, and correction (qdes_esn - q_des_csv)
        compare_png = path.with_suffix('.esn_compare.png')
        try:
            _plt.figure(figsize=(10, 6))
            _plt.subplot(2, 1, 1)
            _plt.plot(t_arr, q_arr, label='q (measured)', color='tab:blue')
            if np.isfinite(qcsv_arr).any():
                _plt.plot(t_arr, qcsv_arr, label='q_des_csv', color='tab:green')
            if np.isfinite(qesn_arr).any():
                _plt.plot(t_arr, qesn_arr, label='qdes_esn', color='tab:orange')
            _plt.xlabel('time (s)')
            _plt.ylabel('angle (deg)')
            _plt.grid(True)
            _plt.legend()

            _plt.subplot(2, 1, 2)
            # compute correction (esn - csv)
            corr = np.full_like(t_arr, np.nan)
            if np.isfinite(qesn_arr).any() and np.isfinite(qcsv_arr).any():
                corr = qesn_arr - qcsv_arr
            _plt.plot(t_arr, corr, label='qdes_esn - q_des_csv', color='tab:red')
            _plt.xlabel('time (s)')
            _plt.ylabel('correction (deg)')
            _plt.grid(True)
            _plt.legend()
            _plt.tight_layout()
            _plt.savefig(str(compare_png), dpi=150)
            _plt.close()
        except Exception:
            try:
                _plt.close()
            except Exception:
                pass

        print(f"[INFO] ESN plots saved: {main_png} and {compare_png}", flush=True)
        return main_png
    except Exception as e:
        print(f"[WARN] Failed to generate ESN plot for {csv_path}: {e}", flush=True)
        return None


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "Cnt csv dq dqdes enc kpa ldc pid ppm qdes"
# End:
# End of file
