#!/usr/bin/env python
"""Real-time adaptive reference generation using trained ESN for pneumatic valve robot.
Flow:
    1. Read current joint angle from encoder (deg)
    2. Feed current angle to ESN to predict next desired angle q_des_next
  3. Simple P controller (or open-loop) convert desired change to valve duty
  4. Send to DAC (two channels) and log CSV with pressures etc. (if available)

Assumptions:
  - Encoder provides a single joint angle (extend to multi-joint by adding extra columns)
  - Valve command range 0..100 (%)
  - Linear mapping: center 60%, +/- amplitude for delta angle (placeholder)
  - Trained weights stored as .npy (Wout)

TODO (user specific): refine angle->valve mapping, multi-DoF support, safety interlocks.
"""
import argparse, os, time, csv, math, sys
import numpy as np
from ESN import ESN

# --- Simple PID (copied/adapted from apps/pid_tune.py) ---
class SimplePID:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.i = 0.0
        self.prev_y = None
    def reset(self) -> None:
        self.i = 0.0
        self.prev_y = None
    def step(self, sp: float, y: float, dt: float, umin: float | None = None, umax: float | None = None) -> float:
        e = sp - y
        p = self.kp * e
        d = 0.0
        if self.prev_y is not None and dt > 0:
            d = - self.kd * (y - self.prev_y) / dt
        i_next = self.i + self.ki * e * dt
        u_unclamped = p + i_next + d
        if umin is not None and umax is not None:
            u = _clamp(u_unclamped, umin, umax)
            saturated_high = u >= umax - 1e-9
            saturated_low = u <= umin + 1e-9
            if (saturated_high and e > 0) or (saturated_low and e < 0):
                i_next = self.i
                u = _clamp(p + i_next + d, umin, umax)
        else:
            u = u_unclamped
        self.i = i_next
        self.prev_y = y
        return u


# small helpers (clamping + valve safety) adapted from pid_tune
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _clamp_valve(x: float, lo: float, hi: float) -> float:
    safe_hi = min(hi, 100.0)
    return _clamp(x, lo, safe_hi)

def _u_limits(min_pct: float, max_pct: float, center: float, bias: float = 0.0) -> tuple[float, float]:
    safe_hi = min(max_pct, 100.0)
    cB = center + bias
    lo = min_pct
    hi = safe_hi
    umin = max(2.0 * (lo - cB), -2.0 * (hi - cB))
    umax = min(2.0 * (hi - cB), -2.0 * (lo - cB))
    return (umin, umax)

# --- Minimal hardware abstraction placeholders (user should integrate real classes from integrated_sensor_sin_python) ---
try:
    from integrated_sensor_sin_python import Dac8564, Max11632, EncoderSimple  # type: ignore
except Exception:  # Fallback dummy classes if not on target HW
    Dac8564 = None
    Max11632 = None
    EncoderSimple = None

# Logging directory
LOG_DIR = os.path.join('data','esn','control_logs')


def deg_clip(v):
    return float(np.clip(v, -10.0, 10.0))


def angle_to_valve_pair(q_des: float, center=60.0, gain=15.0):
    """Map desired angle (deg) to two valve percentages (A,B).
    Simple differential mapping: A = center + gain*delta, B = center - gain*delta.
    delta here is deviation from initial reference (captured at start)."""
    a = center + gain * q_des
    b = center - gain * q_des
    return max(0.0, min(100.0, a)), max(0.0, min(100.0, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight', required=True, help='Trained Wout .npy path')
    ap.add_argument('--duration', type=float, default=20.0)
    ap.add_argument('--interval-ms', type=float, default=50.0)
    ap.add_argument('--n-reservoir', type=int, default=200)
    ap.add_argument('--input-scale', type=float, default=0.7)
    ap.add_argument('--density', type=float, default=0.1)
    ap.add_argument('--rho', type=float, default=0.99)
    ap.add_argument('--leaking-rate', type=float, default=0.7)
    ap.add_argument('--center', type=float, default=60.0)
    ap.add_argument('--gain', type=float, default=8.0, help='Valve differential gain for angle mapping')
    ap.add_argument('--kp', type=float, default=0.0, help='PID Kp (0 → disable PID)')
    ap.add_argument('--ki', type=float, default=0.0, help='PID Ki')
    ap.add_argument('--kd', type=float, default=0.0, help='PID Kd')
    ap.add_argument('--min', type=float, default=0.0, help='Minimum valve pct (safety)')
    ap.add_argument('--max', type=float, default=100.0, help='Maximum valve pct (safety)')
    ap.add_argument('--bias', type=float, default=0.0, help='Bias added to both valve channels [%]')
    ap.add_argument('--control-invert', dest='control_invert', action='store_true', default=False, help='Invert control polarity (flip u sign)')
    ap.add_argument('--encoder-scale', type=float, default=1.0, help='Multiplier applied after deg->rad')
    ap.add_argument('--log-prefix', default='esn_control')
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize ESN (dimension inference from weight file)
    Wout = np.load(args.weight)
    N_y, N_x = Wout.shape
    # Assume same in/out dimension
    N_u = N_y
    esn = ESN(N_u, N_y, args.n_reservoir, density=args.density, input_scale=args.input_scale,
              rho=args.rho, activation_func=np.tanh, leaking_rate=args.leaking_rate)
    esn.Output.Wout = Wout
    print('[INFO] Loaded weights shape', Wout.shape)

    # Hardware init (user should replace with real constructors) -----------------
    enc = None
    try:
        if EncoderSimple is not None:
            enc = EncoderSimple('/dev/gpiochip4', 14, 4)
            print('[INFO] Encoder initialized')
    except Exception as e:
        print('[WARN] encoder init failed:', e)

    dac = None
    try:
        if Dac8564 is not None:
            dac = Dac8564(0,0,19)
            dac.open()
            print('[INFO] DAC opened')
            dac.set_channels(args.center, args.center)
    except Exception as e:
        print('[WARN] DAC init failed:', e)

    adc = None
    try:
        if Max11632 is not None:
            adc = Max11632(0,0,24,0,1)
            adc.open()
            print('[INFO] ADC opened')
    except Exception as e:
        print('[WARN] ADC init failed:', e)

    # Logging setup
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f'{args.log_prefix}_{ts}.csv')
    header = ['ms','q_cur','q_pred','valve_a','valve_b','pid_u']
    if adc is not None:
        header += ['adc0_raw','adc0_volt','adc0_kpa','adc1_raw','adc1_volt','adc1_kpa']
    f = open(log_path,'w',buffering=1,newline='')
    writer = csv.writer(f)
    writer.writerow(header)

    # Initial state
    start_t = time.perf_counter()
    next_tick = start_t
    interval = args.interval_ms/1000.0

    # Read initial encoder (degrees, keep units in deg to match integrated_sensor_sin_python.py)
    def read_encoder_deg():
        if enc is None:
            return 0.0
        try:
            enc.poll()
            # integrated_sensor_sin_python.py records encoder in degrees via enc.degrees()
            deg = enc.degrees(2048)  # placeholder PPR
        except Exception:
            deg = 0.0
        return float(deg) * args.encoder_scale

    # keep q_cur in degrees (not radians) to match integrated_sensor_sin_python.py
    q_cur = read_encoder_deg()
    # PID setup
    pid = SimplePID(args.kp, args.ki, args.kd)
    pid.reset()
    pid_enabled = (abs(args.kp) > 1e-12) or (abs(args.ki) > 1e-12) or (abs(args.kd) > 1e-12)
    umin, umax = _u_limits(args.min, args.max, args.center, args.bias)
    last_time = start_t

    try:
        while time.perf_counter() - start_t < args.duration:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(max(0, next_tick - now))
                continue
            next_tick += interval

            # ESN step (input in degrees)
            x_in = esn.Input(np.array([q_cur])) if N_u == 1 else esn.Input(np.repeat(q_cur, N_u))
            x = esn.Reservoir(x_in)
            q_pred_vec = esn.Output(x)
            q_pred = float(q_pred_vec[0]) if N_u == 1 else float(q_pred_vec[0])

            # Simple saturation to avoid jump
            max_step = 0.05
            q_pred = float(np.clip(q_pred, q_cur - max_step, q_cur + max_step))

            # Control: either PID on q_pred (setpoint) or direct mapping
            pid_u = 0.0
            if pid_enabled:
                now_loop = time.perf_counter()
                dt = max(1e-6, now_loop - last_time)
                last_time = now_loop
                pid_u = pid.step(q_pred, q_cur, dt, umin, umax)
                if args.control_invert:
                    pid_u = -pid_u
                a_pct = args.center + pid_u/2.0 + args.bias
                b_pct = args.center - pid_u/2.0 + args.bias
                a_pct = _clamp_valve(a_pct, args.min, args.max)
                b_pct = _clamp_valve(b_pct, args.min, args.max)
                valve_a, valve_b = a_pct, b_pct
                if dac is not None:
                    try:
                        dac.set_channels(valve_a, valve_b)
                    except Exception:
                        pass
            else:
                # legacy direct mapping from angle prediction
                valve_a, valve_b = angle_to_valve_pair(q_pred, center=args.center, gain=args.gain)
                if dac is not None:
                    try:
                        dac.set_channels(valve_a, valve_b)
                    except Exception:
                        pass

            # Acquire ADC
            adc_vals = []
            if adc is not None:
                try:
                    r0,v0,k0,r1,v1,k1 = adc.read_pair()
                    adc_vals = [r0,v0,k0,r1,v1,k1]
                except Exception:
                    adc_vals = []

            ms = (time.perf_counter() - start_t)*1000.0
            writer.writerow([f'{ms:.1f}', q_cur, q_pred, valve_a, valve_b, f'{pid_u:.4f}', *adc_vals])

            # Update current angle
            q_cur = read_encoder_rad()

    except KeyboardInterrupt:
        print('\n[INFO] Interrupted by user')
    finally:
        try: f.close()
        except Exception: pass
        if dac is not None:
            try:
                dac.set_channels(args.center, args.center)
            except Exception:
                pass
        if adc is not None:
            try: adc.close()
            except Exception: pass
        print('[INFO] Log saved to', log_path)

if __name__ == '__main__':
    main()
