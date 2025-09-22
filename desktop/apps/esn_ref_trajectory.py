"""ESNによる参照軌道生成スクリプト(河合先生のものを参考)

モード1: q(t)->q(t+1) 予測 (既存)
モード2: q(t)->qdes(t) 適応参照生成 (MATLAB arm feedback + readout 近似)

目的:
    MATLAB スクリプト (construct_network / train_RC_robot / test_RC_robot) が行う
    「内部状態 + 単一入力 → 次ステップ出力 (q(t)→q(t+1))」に揃えた最小構成。

方針 (MATLAB との差異最小化):
    * 明示的な履歴埋め込み / 位相特徴 / 動的 reset 機能を全て排除
    * 入力は常に現在の測定 q(t) 1 次元のみ
    * 予測は q(t+1)
    * 連続自己回帰生成では直前の予測値を次の入力として再利用 (測定が無い場合)

使い方:
    学習:
        python -m apps.esn_ref_trajectory --mode train --data-dir <dir> --output models/esn_refq.npz
    自己回帰デモ (初期値 seed を1回だけ与える):
        python -m apps.esn_ref_trajectory --mode demo --model models/esn_refq.npz --steps 500 --start-q 0.0
    対話 (常に測定値を与える):
        python -m apps.esn_ref_trajectory --mode repl --model models/esn_refq.npz

 学習・評価し、プロットも生成
PYTHONPATH=/mnt/c/Users/HosodaLab2/University/Reserch/reservoir/MyRobotRasPi/desktop/src:/mnt/c/Users/HosodaLab2/University/Reserch/reservoir/MyRobotRasPi/affetto-nn-ctrl/src:/mnt/c/Users/HosodaLab2/University/Reserch/reservoir/myrobot:/mnt/c/Users/HosodaLab2/University/Reserch/reservoir/myrobot/src python3 - << 'PY'
from pathlib import Path
import numpy as np
from desktop.apps.esn_ref_trajectory import load_model, build_dataset
p = Path('desktop/models/esn_rc_compliant.npz')
print('exists', p.exists())
if not p.exists():
    raise SystemExit('model not found')
esn = load_model(p)
X,y = build_dataset(Path('/mnt/c/Users/HosodaLab2/University/Reserch/reservoir/myrobot/hirai/data/myrobot/hirai/csv/20250831/trapezoid/fast'))
print('X,y', X.shape, y.shape)
y_pred = esn.predict(X).ravel()
rmse = np.sqrt(np.mean((y_pred - y)**2))
print('RMSE on train:', rmse)
# save simple plot
import matplotlib.pyplot as plt
idx = slice(0, min(500, len(y)))
plt.figure(figsize=(6,3))
plt.plot(y[idx], label='y')
plt.plot(y_pred[idx], label='y_pred')
plt.legend()
Path('plots').mkdir(parents=True, exist_ok=True)
plt.savefig('plots/esn_rc_compliant_train.png', dpi=150)
print('saved plot')
PY

"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List
from collections import deque

import numpy as np
import pandas as pd

# 既存実装を利用
from affetto_nn_ctrl.esn import ESN, RLS, Tikhonov
from affetto_nn_ctrl.esn_matlab_exact import MatlabExactESN
import matplotlib.pyplot as plt
import time

DEFAULT_DATA_DIR = (
    Path(r"C:/Users/HosodaLab2/University/Reserch/reservoir/myrobot/hirai/data/myrobot/hirai/csv/20250831/trapezoid/fast")
)


"""MATLAB 準拠: 入力 q(t) 1 次元 → 出力 q(t+1) を学習する単純 ESN。"""


def load_q_series(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "q" in df.columns:
        q = df["q"].to_numpy(dtype=float)
    else:
        q_cols = [c for c in df.columns if c.startswith("q")]
        if not q_cols:
            raise ValueError(f"q列が見つかりません: {csv_path}")
        q = df[q_cols[0]].to_numpy(dtype=float)
    return q[~np.isnan(q)]


def build_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_files = sorted(data_dir.glob("motion_data_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSVが見つかりません: {data_dir}")
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for f in csv_files:
        series = load_q_series(f)
        if len(series) < 2:
            continue
        x_part = series[:-1]
        y_part = series[1:]
        X_list.append(x_part[:, None])  # (T-1,1)
        y_list.append(y_part)           # (T-1,)
    if not X_list:
        raise RuntimeError("系列長が不足しています")
    X = np.vstack(X_list).astype(float)  # (N,1)
    y = np.concatenate(y_list).astype(float)  # (N,)
    return X, y


# ----------------------------- 参照生成クラス ----------------------------- #
@dataclass
class ReferenceGenerator:
    """参照生成器 (動的リセット無し)。

    動作:
      * measured_q が与えられた場合: それを入力に 1 ステップ先 q を予測
      * measured_q が None の場合: 直前に予測した値を次入力として自己回帰
    最初の呼び出しで measured_q が None の場合はエラー。
    """
    esn: ESN
    _last_input: float | None = None  # 直近入力 (q(t))
    _last_pred: float | None = None   # 直近予測 (q(t+1))

    def step(self, measured_q: float | None = None) -> float:
        if measured_q is not None:
            x = measured_q
        else:
            # 自己回帰: 直前予測を次の入力にする
            if self._last_pred is None:
                raise RuntimeError("初回は measured_q を与えて初期化してください")
            x = self._last_pred
        feat = np.array([[x]], dtype=float)
        y = float(self.esn.predict(feat)[0])  # q(t+1)
        self._last_input = x
        self._last_pred = y
        return y


# ===================== 適応参照 (q -> qdes)  ===================== #
@dataclass
class ReferenceGeneratorAdaptive:
    """適応参照生成: 測定 q を入力し qdes を生成。

    MATLAB の Out = WOut * X + alpha * q の構造を模倣。
    保存済み readout 重みを WOut, alpha をメタ情報(meta['adaptive_alpha']) として利用。
    1DOF を前提 (拡張は別途)。
    """
    esn: ESN
    alpha: float = 1.0
    _last_q: float | None = None
    _last_qdes_esn: np.ndarray | None = None  # 1ステップ前の最終出力 qdes_esn を保持

    def __post_init__(self) -> None:
        # reset reservoir state when generator is created
        try:
            self.esn.reset_reservoir_state()
        except Exception:
            pass
        # initial last final output (N_y,) zeros
        try:
            self._last_qdes_esn = np.zeros((self.esn.N_y,), dtype=float)
        except Exception:
            self._last_qdes_esn = None

    def step(self, measured_q: float) -> float:
        """Use ESN.oneshot to apply input+feedback so internal feedback path is active.

        The ESN readout is expected to predict (qdes - alpha*q) during training, so
        we call oneshot(u, feedback=last_y) to obtain base, then add alpha*measured_q.
        The feedback `fb` is the *final output* from the previous step (qdes_esn),
        to mimic the MATLAB implementation `WFb * Out`.
        """
        u = np.array([measured_q], dtype=float)
        # use previous final output as feedback if available
        fb = self._last_qdes_esn

        # Special-case: if the loaded model is the MatlabExactESN implementation,
        # use its step()/internal state and WOut to compute base = WOut * X.
        try:
            from affetto_nn_ctrl.esn_matlab_exact import MatlabExactESN as _M
        except Exception:
            _M = None

        base = 0.0
        y_arr = None
        # MatlabExactESN path
        if _M is not None and isinstance(self.esn, _M):
            # set previous final output into model so feedback path WFb*Out uses it
            if fb is not None:
                try:
                    # ensure shape (num_out,)
                    self.esn.Out = np.asarray(fb).ravel()
                except Exception:
                    pass
            # perform one step: this updates X and Out inside the model
            try:
                y_tmp = self.esn.step(np.array([measured_q]))
                # base = WOut * X (readout without alpha*q)
                try:
                    base = float(self.esn.WOut.dot(self.esn.X))
                except Exception:
                    # fallback: compute base from returned value minus alpha*q
                    base = float(np.asarray(y_tmp).ravel()[0] - self.esn.alpha * measured_q)
                y_arr = np.asarray(y_tmp).ravel()
            except Exception:
                base = 0.0

        else:
            # Generic ESN API path (oneshot/readout). Keep previous behavior.
            try:
                # oneshot returns readout vector (N_y,)
                y_out = self.esn.oneshot(u, feedback=fb)
                # ensure 1D
                y_arr = np.asarray(y_out).ravel()
                base = float(y_arr[0])
            except Exception:
                # fallback: use readout(x) without feedback
                try:
                    x_in = self.esn.input(u)
                    x = self.esn.reservoir(x_in)
                    base = float(self.esn.readout(x))
                except Exception:
                    base = 0.0

        # Calculate final output
        qdes_esn = base + self.alpha * measured_q
        
        # Store final output for the next step's feedback
        try:
            self._last_qdes_esn = np.array([qdes_esn], dtype=float)
        except Exception:
            self._last_qdes_esn = None

        self._last_q = measured_q
        return qdes_esn


# ===================== データローダ (q,qdes) ===================== #
def load_q_qdes_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    # q列: 優先 q or q0
    if 'q' in df.columns:
        q = df['q'].to_numpy(dtype=float)
    else:
        q_cols = [c for c in df.columns if c.startswith('q') and not c.startswith('qdes')]
        if not q_cols:
            raise ValueError(f"q列が見つからない: {csv_path}")
        q = df[q_cols[0]].to_numpy(dtype=float)
    # qdes列
    if 'qdes' in df.columns:
        qdes = df['qdes'].to_numpy(dtype=float)
    else:
        qdes_cols = [c for c in df.columns if c.startswith('qdes')]
        if not qdes_cols:
            raise ValueError(f"qdes列が見つからない: {csv_path}")
        qdes = df[qdes_cols[0]].to_numpy(dtype=float)
    mask = ~np.isnan(q) & ~np.isnan(qdes)
    return q[mask], qdes[mask]


def build_dataset_adaptive(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_files = sorted(data_dir.glob('motion_data_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"CSVが見つかりません: {data_dir}")
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for f in csv_files:
        q, qdes = load_q_qdes_series(f)
        if len(q) < 1:
            continue
        # 入力は現在の q(t), 目標は同時刻 qdes(t)
        X_list.append(q[:, None])
        y_list.append(qdes)
    if not X_list:
        raise RuntimeError('系列長が不足しています(adaptive)')
    X = np.vstack(X_list).astype(float)
    y = np.concatenate(y_list).astype(float)
    return X, y


# ----------------------------- 学習/保存/読込 ----------------------------- #
def train_esn_for_q(
    data_dir: Path,
    reservoir_size: int,
    density: float,
    rho: float,
    leaking_rate: float,
    *,
    input_scale: float = 1.0,
    fb_scale: float | None = None,
    optimizer_name: str = "RLS",
    delta: float = 1.0,
    lam: float = 1.0,
    update: int = 1,
) -> tuple[ESN, np.ndarray, np.ndarray]:
    X, y = build_dataset(data_dir)
    # N_y inferred from target shape (1)
    N_y = 1
    esn = ESN(
        N_u=1,
        N_y=N_y,
        N_x=reservoir_size,
        density=density,
        rho=rho,
        leaking_rate=leaking_rate,
        input_scale=input_scale,
        activation_func=np.tanh,
    fb_scale=fb_scale,
        optimizer=None,  # set explicit optimizer below
    )

    # Select optimizer explicitly to match MATLAB (RLS online updates)
    if optimizer_name.lower().startswith("rls"):
        esn.optimizer = RLS(reservoir_size, N_y, delta, lam, update)
    else:
        # default to Tikhonov (batch)
        esn.optimizer = Tikhonov(reservoir_size, N_y, 1e-4)

    # Use teacher forcing during fit to feed true output into feedback (if any)
    esn.fit(X, y[:, None])
    return esn, X, y


def train_esn_adaptive(
    data_dir: Path,
    reservoir_size: int,
    density: float,
    rho: float,
    leaking_rate: float,
    *,
    input_scale: float = 1.0,
    fb_scale: float | None = None,
    optimizer_name: str = 'RLS',
    delta: float = 1.0,
    lam: float = 1.0,
    update: int = 1,
    adaptive_alpha: float = 1.0,
) -> tuple[ESN, np.ndarray, np.ndarray]:
    X, y = build_dataset_adaptive(data_dir)  # X: q, y: qdes
    N_y = 1
    esn = ESN(
        N_u=1,
        N_y=N_y,
        N_x=reservoir_size,
        density=density,
        rho=rho,
        leaking_rate=leaking_rate,
        input_scale=input_scale,
        activation_func=np.tanh,
    fb_scale=fb_scale,
        optimizer=None,
    )
    if optimizer_name.lower().startswith('rls'):
        esn.optimizer = RLS(reservoir_size, N_y, delta, lam, update)
    else:
        esn.optimizer = Tikhonov(reservoir_size, N_y, 1e-4)

    # 目標は (qdes - alpha*q) を readout に学習させる
    target = (y - adaptive_alpha * X.ravel())[:, None]
    esn.fit(X, target)
    # meta に alpha を埋め込むため呼び出し側で save_model を拡張
    return esn, X, y


def save_model(esn: ESN, path: Path, *, extra_meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save weights and useful hyperparameters (meta as dict for full restore)
    meta = {
        "N_u": int(esn.N_u),
        "N_x": int(esn.N_x),
        "N_y": int(esn.N_y),
        "density": float(getattr(esn.reservoir, "density", np.nan)),
        "rho": float(getattr(esn.reservoir, "rho", np.nan)),
        "leaking_rate": float(getattr(esn.reservoir, "alpha", np.nan)),
        "input_scale": float(getattr(esn.input, "weight", np.nan).std() if hasattr(esn.input, "weight") else 1.0),
        # store fb_scale flag/scale and include actual feedback weights when present
        "fb_scale": None if esn.feedback is None else float(np.mean(np.abs(esn.feedback.weight))),
        "optimizer": type(esn.optimizer).__name__ if esn.optimizer is not None else None,
    }
    if extra_meta:
        meta.update(extra_meta)
    # Build save dict
    save_kwargs = dict(
        input_weight=esn.input.weight,
        reservoir_W=esn.reservoir.W.data,
        reservoir_indices=esn.reservoir.W.indices,
        reservoir_indptr=esn.reservoir.W.indptr,
        reservoir_shape=np.array(esn.reservoir.W.shape, dtype=int),
        readout_weight=esn.readout.weight,
        meta=meta,
    )
    # include feedback weights if present
    if esn.feedback is not None and hasattr(esn.feedback, "weight"):
        save_kwargs["feedback_weight"] = esn.feedback.weight

    np.savez(path, **save_kwargs)


def load_model(path: Path) -> ESN:
    data = np.load(path, allow_pickle=True)
    meta = data["meta"].tolist() if hasattr(data["meta"], "tolist") else data["meta"].item()
    # if this is an exact matlab style model, use dedicated loader
    if meta.get("exact_matlab") is True or meta.get("mode") == "matlab_exact":
        from affetto_nn_ctrl.esn_matlab_exact import MatlabExactESN
        d = {k: data[k] for k in data.files if k != "meta"}
        d["meta"] = meta
        return MatlabExactESN.from_dict(d)  # type: ignore[return-value]
    N_u = int(meta.get("N_u", 1))
    N_x = int(meta.get("N_x", 0))
    N_y = int(meta.get("N_y", 1))
    # Reconstruct ESN with saved hyperparameters when possible
    esn = ESN(
        N_u=N_u,
        N_y=N_y,
        N_x=N_x,
        density=float(meta.get("density", 0.05)),
        rho=float(meta.get("rho", 0.95)),
        leaking_rate=float(meta.get("leaking_rate", 1.0)),
        input_scale=float(meta.get("input_scale", 1.0)),
        fb_scale=meta.get("fb_scale", None),
        optimizer=None,
    )
    esn.input.set_weight(data["input_weight"])  # type: ignore[arg-type]
    from scipy import sparse

    W = sparse.csr_matrix(
        (
            data["reservoir_W"],  # data
            data["reservoir_indices"],  # indices
            data["reservoir_indptr"],  # indptr
        ),
        shape=tuple(data["reservoir_shape"]),
    )
    esn.reservoir.W = W  # type: ignore[assignment]
    esn.readout.set_weight(data["readout_weight"])  # type: ignore[arg-type]
    # restore feedback weights if saved
    if "feedback_weight" in data.files and esn.feedback is not None:
        try:
            esn.feedback.set_weight(data["feedback_weight"])  # type: ignore[arg-type]
        except Exception:
            pass
    # Try to reconstruct optimizer object (basic)
    opt_name = meta.get("optimizer", None)
    if opt_name is not None and opt_name.lower().startswith("rls"):
        # use conservative defaults if nothing saved
        esn.optimizer = RLS(N_x, N_y, 1.0, 1.0, 1)
    elif opt_name is not None and opt_name.lower().startswith("tikhonov"):
        esn.optimizer = Tikhonov(N_x, N_y, 1e-4)
    # alpha (適応参照用) は meta から利用者側で参照
    return esn

# ----------------------------- CLI ----------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESN 参照q生成 (history削除版)")
    p.add_argument("--mode", choices=[
        "train", "demo", "repl",
        "train_adaptive", "demo_adaptive",
        "train_matlab_single", "eval_matlab_single",
        "train_matlab_exact",
        "train_matlab_strict", "eval_matlab_strict"
    ], required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output", type=Path, default=Path("models/esn_refq.npz"))
    p.add_argument("--model", type=Path, help="学習済みモデル(npz)パス", default=Path("models/esn_refq.npz"))
    p.add_argument("--reservoir-size", type=int, default=300)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--leaking-rate", type=float, default=0.7)
    p.add_argument("--dt", type=float, default=None, help="time step dt (when using MATLAB-style dt/tau -> leaking_rate)")
    p.add_argument("--tau", type=float, default=None, help="time constant tau (when using MATLAB-style dt/tau -> leaking_rate)")
    p.add_argument("--reg", type=float, default=1e-5, help="Tikhonov regularization (ridge) for batch solver")
    p.add_argument("--fb-scale", type=float, default=0.0, help="feedback weight scale (None disables feedback)")
    p.add_argument("--input-scale", type=float, default=1.0, help="input weight scale")
    p.add_argument("--n-learn-loops", type=int, default=10, help="number of training loops (MATLAB style)")
    p.add_argument("--optimizer", type=str, default="RLS", help="optimizer for matlab_single training: RLS or Tikhonov")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--start-q", type=float, default=0.0)
    p.add_argument("--adaptive-alpha", type=float, default=1.0, help="qdes = alpha*q + readout(x) での alpha")
    p.add_argument("--single-csv", type=Path, help="単一CSVパス (matlab_single 用)")
    # exact matlab style additional args
    p.add_argument("--p-connect", type=float, default=0.1, help="(exact) recurrent connection probability")
    p.add_argument("--scale", type=float, default=1.0, help="(exact) recurrent weight scale")
    p.add_argument("--input-weight-amp", type=float, default=1.0, help="(exact) input weight amplitude")
    p.add_argument("--feedback-weight-amp", type=float, default=1.0, help="(exact) feedback weight amplitude")
    p.add_argument("--alpha", type=float, default=1.0, help="(exact) Out = WOut*X + alpha*q 係数")
    p.add_argument("--delta", type=float, default=1.0, help="(exact) P 初期化係数 (P=I/delta)")
    p.add_argument("--learn-every", type=int, default=1, help="(exact) 何ステップ毎に RLS 更新するか")
    p.add_argument("--start-train-n", type=int, default=0, help="(exact) 学習開始インデックス")
    p.add_argument("--end-train-n", type=int, default=-1, help="(exact) 学習終了インデックス (-1 で末尾)")
    p.add_argument("--rng-seed", type=int, default=0, help="(exact) 乱数シード")
    # pulse input options (MATLAB style input_pattern)
    p.add_argument("--pulse-start", type=float, default=None, help="(exact) パルス開始時刻 (同じ時間単位; dt と同単位)")
    p.add_argument("--pulse-duration", type=float, default=None, help="(exact) パルス継続時間")
    p.add_argument("--pulse-value", type=float, default=1.0, help="(exact) パルス値")
    p.add_argument("--pulse-steps", action="store_true", help="pulse-start と pulse-duration を時刻ではなくサンプル(ステップ)数として解釈する")
    # strict options (reuse exact backend but allow inline configuration)
    p.add_argument("--matlab-strict-forward", action="store_true", help="standard ESN の場合でも forward_external を使い MATLAB 順序にする")
    return p.parse_args(argv)


def cmd_train(args: argparse.Namespace) -> None:
    # allow Matlab-style dt/tau mapping: leaking_rate ~= dt / tau
    leaking_rate = args.leaking_rate
    if args.dt is not None and args.tau is not None:
        leaking_rate = float(args.dt) / float(args.tau)
        print(f"[INFO] using dt/tau -> leaking_rate = dt/tau = {leaking_rate:.6g}")

    esn, X, y = train_esn_for_q(
        data_dir=args.data_dir,
        reservoir_size=args.reservoir_size,
        density=args.density,
    rho=args.rho,
    leaking_rate=leaking_rate,
    fb_scale=args.fb_scale,
    )
    # strict forward option
    if args.matlab_strict_forward:
        esn.reservoir.forward = esn.reservoir.forward_external  # type: ignore[method-assign]
    save_model(esn, args.output)
    print(f"[TRAIN] samples={len(X)} N_u=1 保存: {args.output}")


def cmd_demo(args: argparse.Namespace) -> None:
    esn = load_model(args.model)
    gen = ReferenceGenerator(esn)
    q_ref_list: list[float] = []
    # 初期 seed として start-q を 1 回測定入力として与える
    first = gen.step(measured_q=args.start_q)
    q_ref_list.append(first)
    for _ in range(args.steps - 1):
        q_ref_list.append(gen.step(measured_q=None))
    arr = np.array(q_ref_list)
    print(f"[DEMO] 生成 shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}")


def cmd_repl(args: argparse.Namespace) -> None:
    esn = load_model(args.model)
    gen = ReferenceGenerator(esn)
    print("現在の実測 q を入力してください (Ctrl+C で終了)。空行で直前予測を自己回帰に使用。")
    while True:
        try:
            line = input("q?> ").strip()
            if line == "":
                try:
                    q_ref = gen.step(measured_q=None)
                except RuntimeError:
                    print("(最初の自己回帰前に数値を1度入力してください)")
                    continue
            else:
                try:
                    q_val = float(line)
                except ValueError:
                    print("数値を入力してください")
                    continue
                q_ref = gen.step(measured_q=q_val)
            print(f"q_ref = {q_ref:.6f}")
        except KeyboardInterrupt:
            print("\n終了")
            break


def cmd_train_adaptive(args: argparse.Namespace) -> None:
    leaking_rate = args.leaking_rate
    if args.dt is not None and args.tau is not None:
        leaking_rate = float(args.dt) / float(args.tau)
        print(f"[INFO] using dt/tau -> leaking_rate = dt/tau = {leaking_rate:.6g}")

    esn, X, y = train_esn_adaptive(
        data_dir=args.data_dir,
        reservoir_size=args.reservoir_size,
        density=args.density,
    rho=args.rho,
    leaking_rate=leaking_rate,
    fb_scale=args.fb_scale,
        adaptive_alpha=args.adaptive_alpha,
    )
    if args.matlab_strict_forward:
        esn.reservoir.forward = esn.reservoir.forward_external  # type: ignore[method-assign]
    save_model(esn, args.output, extra_meta={"adaptive_alpha": args.adaptive_alpha, "mode": "adaptive"})
    print(f"[TRAIN_ADAPTIVE] samples={len(X)} 保存: {args.output}")


def cmd_demo_adaptive(args: argparse.Namespace) -> None:
    esn = load_model(args.model)
    # adaptive_alpha を meta から読みたいが現行 load_model は meta を返さないので再ロード
    data = np.load(args.model, allow_pickle=True)
    meta = data["meta"].tolist() if hasattr(data["meta"], "tolist") else data["meta"].item()
    alpha = float(meta.get("adaptive_alpha", 1.0))
    gen = ReferenceGeneratorAdaptive(esn, alpha=alpha)
    # デモ: ランダムウォーク q を入力し qdes を生成
    rng = np.random.default_rng(0)
    q = 0.0
    outs = []
    for i in range(args.steps):
        q += 0.01 * rng.normal()
        qdes_esn = gen.step(q)
        outs.append((q, qdes_esn))
    arr = np.array(outs)
    print(f"[DEMO_ADAPTIVE] shape={arr.shape} q_mean={arr[:,0].mean():.4f} qdes_mean={arr[:,1].mean():.4f}")
    # 簡易プロット
    try:
        plt.figure(figsize=(6,3))
        plt.plot(arr[:,0], label='q')
        plt.plot(arr[:,1], label='qdes_esn')
        plt.legend(); Path('plots').mkdir(exist_ok=True, parents=True)
        # save with timestamp to avoid overwriting
        out_png = Path('plots') / f"demo_adaptive_{int(time.time())}.png"
        plt.savefig(out_png, dpi=120)
        print(f"[DEMO_ADAPTIVE] プロット保存: {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] plot failed: {e}")


# -------- MATLAB style: 単一CSV q->qdes 学習 (teacher forcing) -------- #
def train_matlab_style_single(
    csv_path: Path,
    reservoir_size: int,
    density: float,
    rho: float,
    leaking_rate: float,
    adaptive_alpha: float,
    input_scale: float = 1.0,
    fb_scale: float | None = None,
    reg: float = 1e-4,
    *,
    n_learn_loops: int = 10,
    optimizer_name: str = "RLS",
) -> tuple[ESN, np.ndarray, np.ndarray]:
    q, qdes = load_q_qdes_series(csv_path)
    X = q[:, None]
    # residual target for readout (d = qdes - alpha*q)
    residual = (qdes - adaptive_alpha * q)
    target = residual[:, None]
    esn = ESN(
        N_u=1,
        N_y=1,
        N_x=reservoir_size,
        density=density,
        rho=rho,
        leaking_rate=leaking_rate,
    input_scale=input_scale,
        activation_func=np.tanh,
        fb_scale=fb_scale,
        optimizer=None,
    )
    # choose optimizer
    if optimizer_name.lower().startswith("rls"):
        opt = RLS(reservoir_size, 1, 1.0, 1.0, 1)
    else:
        opt = Tikhonov(reservoir_size, 1, reg)
    esn.optimizer = opt

    # Build final output sequence for feedback if needed
    final_out = residual + adaptive_alpha * q  # qdes_esn ideal sequence
    F = None
    if fb_scale is not None and fb_scale != 0.0 and esn.feedback is not None:
        F = np.vstack([np.zeros((1,)), final_out[:, None]])  # shape (T+1,1)

    # Perform n_learn_loops loops, reinitializing reservoir each loop (MATLAB style)
    for j in range(n_learn_loops):
        esn.reset_reservoir_state()
        if F is not None:
            esn._fit(X, target, F=F, enable_teacher_force=False, optimizer=esn.optimizer)  # type: ignore[arg-defined]
        else:
            # run fit with online updates (no teacher forcing) so optimizer updates Wout progressively
            esn._fit(X, target, enable_teacher_force=False, optimizer=esn.optimizer)  # type: ignore[arg-defined]

    # After loops, ensure readout weight is set from optimizer
    try:
        esn.readout.set_weight(esn.optimizer.get_Wout())
    except Exception:
        pass
    return esn, q, qdes


def cmd_train_matlab_single(args: argparse.Namespace) -> None:
    if args.single_csv is None:
        raise SystemExit("--single-csv を指定してください")
    leaking_rate = args.leaking_rate
    if args.dt is not None and args.tau is not None:
        leaking_rate = float(args.dt) / float(args.tau)
        print(f"[INFO] using dt/tau -> leaking_rate = dt/tau = {leaking_rate:.6g}")

    esn, q, qdes = train_matlab_style_single(
        args.single_csv,
        args.reservoir_size,
        args.density,
        args.rho,
        leaking_rate,
        args.adaptive_alpha,
    fb_scale=args.fb_scale,
    reg=args.reg,
    n_learn_loops=args.n_learn_loops,
    optimizer_name=args.optimizer,
    )
    if args.matlab_strict_forward:
        esn.reservoir.forward = esn.reservoir.forward_external  # type: ignore[method-assign]
    out_path = args.output
    save_model(esn, out_path, extra_meta={
        "adaptive_alpha": args.adaptive_alpha,
        "mode": "matlab_single",
        "fb_feedback_mode": "final_output" if args.fb_scale not in (None, 0.0) else "none",
        "target_type": "residual",
        "forward_mode": "external" if args.matlab_strict_forward else "internal",
    })
    print(f"[TRAIN_MATLAB_SINGLE] samples={len(q)} 保存: {out_path}")


def cmd_eval_matlab_single(args: argparse.Namespace) -> None:
    if args.single_csv is None:
        raise SystemExit("--single-csv を指定してください")
    esn = load_model(args.model)
    data = np.load(args.model, allow_pickle=True)
    meta = data['meta'].tolist() if hasattr(data['meta'], 'tolist') else data['meta'].item()
    alpha = float(meta.get('adaptive_alpha', 1.0))
    q, qdes = load_q_qdes_series(args.single_csv)
    # 再現生成
    gen = ReferenceGeneratorAdaptive(esn, alpha=alpha)
    qdes_esn = []
    for val in q:
        qdes_esn.append(gen.step(val))
    qdes_esn = np.array(qdes_esn)
    rmse = float(np.sqrt(np.mean((qdes_esn - qdes)**2)))
    print(f"[EVAL_MATLAB_SINGLE] N={len(q)} RMSE(qdes_esn vs qdes)={rmse:.6f}")
    # プロット
    try:
        plt.figure(figsize=(7,3))
        plt.plot(q, label='q(measured)')
        plt.plot(qdes, label='qdes(orig)')
        plt.plot(qdes_esn, label='qdes_esn(ESN)')
        plt.legend(); Path('plots').mkdir(exist_ok=True, parents=True)
        # include model stem and timestamp to avoid overwriting
        model_stem = Path(args.model).stem if args.model is not None else 'model'
        png = Path('plots') / f"matlab_single_eval_{model_stem}_{int(time.time())}.png"
        plt.savefig(png, dpi=130)
        print(f"[EVAL_MATLAB_SINGLE] plot: {png}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] plot failed: {e}")


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "demo":
        cmd_demo(args)
    elif args.mode == "repl":
        cmd_repl(args)
    elif args.mode == "train_adaptive":
        cmd_train_adaptive(args)
    elif args.mode == "demo_adaptive":
        cmd_demo_adaptive(args)
    elif args.mode == "train_matlab_single":
        cmd_train_matlab_single(args)
    elif args.mode == "eval_matlab_single":
        cmd_eval_matlab_single(args)
    elif args.mode == "train_matlab_exact":
        if args.single_csv is None:
            raise SystemExit("--single-csv を指定してください")
        # load q,qdes
        q, qdes = load_q_qdes_series(args.single_csv)
        end_train_n = None if args.end_train_n < 0 else args.end_train_n
        esn_exact = MatlabExactESN(
            num_units=args.reservoir_size,
            num_in=1,
            num_out=1,
            p_connect=args.p_connect,
            scale=args.scale,
            input_weight_amp=args.input_weight_amp,
            feedback_weight_amp=args.feedback_weight_amp,
            alpha=args.alpha,
            dt=args.dt if args.dt is not None else 0.01,
            tau=args.tau if args.tau is not None else 0.1,
            delta=args.delta,
            learn_every=args.learn_every,
            start_train_n=args.start_train_n,
            end_train_n=end_train_n,
            rng_seed=args.rng_seed,
        )
        # optional pulse drive construction
        input_drive = None
        if args.pulse_start is not None and args.pulse_duration is not None:
            T = len(q)
            if getattr(args, 'pulse_steps', False):
                # interpret provided values as step counts directly
                start_n = int(args.pulse_start)
                dur_n = int(args.pulse_duration)
            else:
                dt_eff = args.dt if args.dt is not None else 0.01
                start_n = int(round(args.pulse_start / dt_eff))
                dur_n = int(round(args.pulse_duration / dt_eff))
            input_drive = np.zeros((T,), dtype=float)
            end_n = min(start_n + dur_n, T)
            if start_n < T:
                input_drive[start_n:end_n] = args.pulse_value
            # 学習開始インデックスを pulse 終了後に自動調整（指定が 0 のままなら）
            if args.start_train_n == 0:
                args.start_train_n = end_n
                esn_exact.start_train_n = args.start_train_n
        metrics = esn_exact.train_sequence(q, qdes, n_learn_loops=args.n_learn_loops, input_drive=input_drive)
        # save
        d = esn_exact.to_dict()
        meta = d.pop("meta")
        meta.update({
            "mode": "matlab_exact",
            "R2_per_loop": metrics["R2_per_loop"],
        })
        import numpy as _np
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _np.savez(out_path, **d, meta=meta)
        print(f"[TRAIN_MATLAB_EXACT] 保存: {out_path} R2_per_loop={metrics['R2_per_loop']}")
    elif args.mode == "train_matlab_strict":
        if args.single_csv is None:
            raise SystemExit("--single-csv を指定してください")
        # strict は matlab_exact 実装をそのまま使い、保存メタに strict フラグ
        q, qdes = load_q_qdes_series(args.single_csv)
        end_train_n = None if args.end_train_n < 0 else args.end_train_n
        esn_exact = MatlabExactESN(
            num_units=args.reservoir_size,
            num_in=1,
            num_out=1,
            p_connect=args.p_connect,
            scale=args.scale,
            input_weight_amp=args.input_weight_amp,
            feedback_weight_amp=args.feedback_weight_amp,
            alpha=args.alpha,
            dt=args.dt if args.dt is not None else 0.01,
            tau=args.tau if args.tau is not None else 0.1,
            delta=args.delta,
            learn_every=args.learn_every,
            start_train_n=args.start_train_n,
            end_train_n=end_train_n,
            rng_seed=args.rng_seed,
        )
        input_drive = None
        if args.pulse_start is not None and args.pulse_duration is not None:
            T = len(q)
            if getattr(args, 'pulse_steps', False):
                start_n = int(args.pulse_start)
                dur_n = int(args.pulse_duration)
            else:
                dt_eff = args.dt if args.dt is not None else 0.01
                start_n = int(round(args.pulse_start / dt_eff))
                dur_n = int(round(args.pulse_duration / dt_eff))
            input_drive = np.zeros((T,), dtype=float)
            end_n = min(start_n + dur_n, T)
            if start_n < T:
                input_drive[start_n:end_n] = args.pulse_value
            if args.start_train_n == 0:
                args.start_train_n = end_n
                esn_exact.start_train_n = args.start_train_n
        metrics = esn_exact.train_sequence(q, qdes, n_learn_loops=args.n_learn_loops, input_drive=input_drive)
        d = esn_exact.to_dict(); meta = d.pop("meta")
        meta.update({
            "mode": "matlab_strict",
            "R2_per_loop": metrics["R2_per_loop"],
        })
        import numpy as _np
        out_path = args.output; out_path.parent.mkdir(parents=True, exist_ok=True)
        _np.savez(out_path, **d, meta=meta)
        print(f"[TRAIN_MATLAB_STRICT] 保存: {out_path} R2_per_loop={metrics['R2_per_loop']}")
    elif args.mode == "eval_matlab_strict":
        if args.single_csv is None:
            raise SystemExit("--single-csv を指定してください")
        # ロード（自動で MatlabExactESN インスタンスへ）
        model = load_model(args.model)
        q, qdes = load_q_qdes_series(args.single_csv)
        # 生成 (学習済み重みを使い Out = WOut*X + alpha*q)
        from affetto_nn_ctrl.esn_matlab_exact import MatlabExactESN as _M
        if not isinstance(model, _M):
            print("[WARN] モデルが strict/exact 形式ではありません")
        outs = model.generate(q)  # type: ignore[arg-type]
        rmse = float(np.sqrt(np.mean((outs - qdes)**2)))
        print(f"[EVAL_MATLAB_STRICT] N={len(q)} RMSE={rmse:.6f}")
        try:
            plt.figure(figsize=(7,3))
            plt.plot(q, label='q')
            plt.plot(qdes, label='qdes(target)')
            plt.plot(outs, label='qdes_strict')
            plt.legend(); Path('plots').mkdir(exist_ok=True, parents=True)
            png = Path('plots') / f"matlab_strict_eval_{Path(args.model).stem}_{int(time.time())}.png"
            plt.savefig(png, dpi=130)
            print(f"[EVAL_MATLAB_STRICT] plot: {png}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] plot failed: {e}")
    else:  # pragma: no cover
        raise RuntimeError("unknown mode")


if __name__ == "__main__":  # pragma: no cover
    main()