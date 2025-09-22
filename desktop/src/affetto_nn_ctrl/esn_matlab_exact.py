"""MATLAB exact-style ESN implementation.

Reproduces the update equations from train_RC_robot.m:
    Xv_current = W * X + WIn * Input + WFb * Out
    Xv = Xv + ((-Xv + Xv_current) ./ tau) * dt
    X = tanh(Xv)
    Out = WOut * X + alpha * q

Also implements an online RLS-like update identical in algebraic form to the MATLAB code:
    error = Out - target
    P_old = P
    P_old_X = P_old * X
    den = 1 + X' * P_old_X
    P = P_old - (P_old_X * P_old_X') / den
    WOut = WOut - error * (P_old_X / den)'

Simplifications:
- We assume 1D input (q) and 1D output (qdes) for now (can generalize later).
- No embedded PD control / arm dynamics (the user can extend externally if needed).
- Training window (start_train_n, end_train_n) and learn_every are supported for parity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import sparse

Array = np.ndarray


def _randn(shape, scale=1.0, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    return scale * rng.standard_normal(shape)


def _make_recurrent(num_units: int, p_connect: float, scale: float, rng: Optional[np.random.Generator] = None) -> sparse.csr_matrix:
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random((num_units, num_units)) <= p_connect
    W = _randn((num_units, num_units), scale=scale, rng=rng)
    W *= mask
    np.fill_diagonal(W, 0.0)
    return sparse.csr_matrix(W)


@dataclass
class MatlabExactESN:
    num_units: int
    num_in: int = 1
    num_out: int = 1
    p_connect: float = 0.1
    scale: float = 1.0
    input_weight_amp: float = 1.0
    feedback_weight_amp: float = 1.0
    alpha: float = 1.0  # Out = WOut*X + alpha * q
    dt: float = 0.01
    tau: float = 0.1
    delta: float = 1.0  # for P initialization (P = I/delta)
    learn_every: int = 1
    start_train_n: int = 0
    end_train_n: Optional[int] = None
    rng_seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)
        self.WIn = _randn((self.num_units, self.num_in), scale=self.input_weight_amp, rng=self.rng)
        self.WFb = _randn((self.num_units, self.num_out), scale=self.feedback_weight_amp, rng=self.rng)
        self.W = _make_recurrent(self.num_units, self.p_connect, self.scale, self.rng)
        self.WOut = np.zeros((self.num_out, self.num_units), dtype=float)
        self.P = np.eye(self.num_units, dtype=float) / self.delta
        # states
        self.Xv = 2 * self.rng.random(self.num_units) - 1.0  # uniform [-1,1]
        self.X = np.tanh(self.Xv)
        self.Out = np.zeros((self.num_out,), dtype=float)
    # optional: default drive array used when train/generate called without explicit input_drive

    def reset_state(self):
        self.Xv = 2 * self.rng.random(self.num_units) - 1.0
        self.X = np.tanh(self.Xv)
        self.Out = np.zeros((self.num_out,), dtype=float)

    def step(self, q: Array, q_external: Optional[Array] = None) -> Array:
        """One forward step (no learning). q is shape (num_in,) = (1,) here.
        q_external can override the q used in alpha*q addition if provided.
        """
        if q_external is None:
            q_external = q
        # Build linear drive
        # Xv_current = W*X + WIn*Input + WFb*Out
        Xv_current = self.W.dot(self.X) + self.WIn.dot(q) + self.WFb.dot(self.Out)
        # Leaky integrator with continuous-time style
        self.Xv = self.Xv + ((-self.Xv + Xv_current) / self.tau) * self.dt
        self.X = np.tanh(self.Xv)
        # Out = WOut * X + alpha * q
        self.Out = self.WOut.dot(self.X) + self.alpha * q_external
        return self.Out.copy()

    def train_sequence(self, q: Array, target: Array, n_learn_loops: int = 1, *, input_drive: Array | None = None) -> dict:
        """Train on a single (time,) sequence using MATLAB-like online RLS update.
        Parameters
        ----------
        q : (T,) array
            Original measured sequence used for the alpha*q 外部項 (Out = WOut*X + alpha*q).
        target : (T,) array
            Desired output (e.g. qdes).
        n_learn_loops : int
            Number of training passes (reservoir state reset each loop).
        input_drive : (T,) array, optional
            If provided, this array is used as the actual reservoir drive (WIn * input_drive[t]) while
            the alpha 項には常に *q[t]* を用いる。これにより MATLAB の input_pattern（パルス入力）
            で内部状態を初期活性化しつつ、出力は測定 q を基準に学習する挙動を再現できる。
            If None, the drive defaults to q (legacy behaviour).
        Returns
        -------
        dict
            {"R2_per_loop": array(num_loops,)}
        """
        assert q.shape == target.shape
        T = q.shape[0]
        if self.end_train_n is None:
            self.end_train_n = T
        R2_per_loop = []
        start = self.start_train_n
        end = self.end_train_n
        # precompute mean/std segment for R^2 if needed
        # allow an instance-level default input drive when caller omits input_drive
        if input_drive is None and getattr(self, 'default_input_drive', None) is not None:
            input_drive = self.default_input_drive

        for loop in range(n_learn_loops):
            self.reset_state()
            for t in range(T):
                # reservoir drive value (パルス等) と alpha*q 用の元の q を分離
                if input_drive is None:
                    drive_val = q[t]
                    q_external = None  # step 内で q をそのまま使用
                else:
                    drive_val = input_drive[t]
                    q_external = np.array([q[t]])
                q_t_drive = np.array([drive_val])
                # forward (if q_external provided, override alpha*q で使う値)
                y = self.step(q_t_drive, q_external=q_external)
                # learning window & interval
                if (t >= start) and (t < end) and (t % self.learn_every == 0):
                    # error = Out - target
                    error = (y - target[t])  # shape (1,)
                    # RLS-like update (vectorized)
                    P_old = self.P
                    X_vec = self.X  # (num_units,)
                    P_old_X = P_old @ X_vec
                    den = 1.0 + X_vec @ P_old_X
                    self.P = P_old - np.outer(P_old_X, P_old_X) / den
                    # WOut update: WOut = WOut - error * (P_old_X/den)'
                    gain_vec = P_old_X / den  # (num_units,)
                    self.WOut = self.WOut - error.reshape(-1, 1) @ gain_vec.reshape(1, -1)
            # compute R^2 on training window (simple): correlation^2 between Out history and target on window
            # 評価時も同じ drive を再現（input_drive があればそれを使いつつ alpha*q を適用）
            # evaluation: prefer explicit input_drive, otherwise use instance default when set
            if input_drive is None and getattr(self, 'default_input_drive', None) is not None:
                out_hist = self.generate_with_drive(q, self.default_input_drive)
            elif input_drive is None:
                out_hist = self.generate(q)
            else:
                out_hist = self.generate_with_drive(q, input_drive)
            seg_o = out_hist[start:end]
            seg_t = target[start:end]
            if seg_o.size > 1:
                c = np.corrcoef(seg_o, seg_t)[0, 1]
                R2_per_loop.append(c * c)
            else:
                R2_per_loop.append(np.nan)
        return {"R2_per_loop": np.array(R2_per_loop)}

    def generate(self, q: Array) -> Array:
        """Run forward without learning, returning outputs (T,)."""
        # If an instance-level default_input_drive is set, use it for the reservoir drive
        if getattr(self, 'default_input_drive', None) is not None:
            return self.generate_with_drive(q, self.default_input_drive)
        T = q.shape[0]
        self.reset_state()
        outs = np.zeros((T,), dtype=float)
        for t in range(T):
            outs[t] = self.step(np.array([q[t]]))
        return outs

    def set_default_input_drive(self, input_drive: Array) -> None:
        """Set an instance-level default input drive array (shape (T,)).
        When set, train_sequence/generate will use this drive when no explicit input_drive is passed.
        """
        self.default_input_drive = input_drive

    def generate_with_drive(self, q: Array, input_drive: Array) -> Array:
        """Forward pass when reservoir drive differs from q used in alpha*q term.
        input_drive: (T,) values fed through WIn
        q:          (T,) values used only for alpha * q (外部項)
        """
        assert q.shape == input_drive.shape
        T = q.shape[0]
        self.reset_state()
        outs = np.zeros((T,), dtype=float)
        for t in range(T):
            outs[t] = self.step(np.array([input_drive[t]]), q_external=np.array([q[t]]))
        return outs

    def to_dict(self) -> dict:
        return {
            "WIn": self.WIn,
            "WFb": self.WFb,
            "WOut": self.WOut,
            "W_data": self.W.data,
            "W_indices": self.W.indices,
            "W_indptr": self.W.indptr,
            "W_shape": np.array(self.W.shape, dtype=int),
            "P": self.P,
            "meta": {
                "num_units": self.num_units,
                "num_in": self.num_in,
                "num_out": self.num_out,
                "p_connect": self.p_connect,
                "scale": self.scale,
                "input_weight_amp": self.input_weight_amp,
                "feedback_weight_amp": self.feedback_weight_amp,
                "alpha": self.alpha,
                "dt": self.dt,
                "tau": self.tau,
                "delta": self.delta,
                "learn_every": self.learn_every,
                "start_train_n": self.start_train_n,
                "end_train_n": self.end_train_n,
                "exact_matlab": True,
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "MatlabExactESN":
        meta = d["meta"]
        obj = MatlabExactESN(
            num_units=meta["num_units"],
            num_in=meta["num_in"],
            num_out=meta["num_out"],
            p_connect=meta["p_connect"],
            scale=meta["scale"],
            input_weight_amp=meta["input_weight_amp"],
            feedback_weight_amp=meta["feedback_weight_amp"],
            alpha=meta["alpha"],
            dt=meta["dt"],
            tau=meta["tau"],
            delta=meta["delta"],
            learn_every=meta["learn_every"],
            start_train_n=meta["start_train_n"],
            end_train_n=meta["end_train_n"],
        )
        # overwrite weights
        from scipy import sparse as _s
        obj.WIn = d["WIn"]
        obj.WFb = d["WFb"]
        obj.WOut = d["WOut"]
        obj.W = _s.csr_matrix((d["W_data"], d["W_indices"], d["W_indptr"]), shape=tuple(d["W_shape"]))
        obj.P = d["P"]
        obj.reset_state()
        return obj

