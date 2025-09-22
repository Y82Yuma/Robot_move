import numpy as np
import matplotlib.pyplot as plt
import time
from model_new import ESN, Tikhonov, dynamic_target_angle_circular_time, animate_esn_trajectory   # model.pyのESNとTikhonovクラスをインポート
import os

# ESNおよびロボットモデルのパラメータ設定
N_x = 200               # リザバーのノード数
input_scale = 0.7       # 入力スケーリング　リザバーへの入力信号の強さ
density = 0.1           # 結合密度
rho = 0.99              # スペクトル半径
leaking_rate = 0.7     # リーキング率
beta = 0.00001           # リッジ回帰の正則化係数
activation_function = np.tanh #活性化関数

# 円軌道生成用パラメータ
center_angle = np.array([np.pi / 2, np.pi / 2])  # 円の中心位置
radius = np.pi/6                                    # 円軌道の半径
total_revolutions = 1                           # 総回転数（2周回る）
start_angle = np.array([np.pi / 2 + np.pi/6, np.pi / 2])
end_angle = np.array([np.pi / 2 + np.pi/6, np.pi / 2])

# ロボットのパラメータ
link_lengths = [0.2, 0.18]  # 各リンクの長さ (0.1 m)
KP = np.diag([60.0, 60.0])  # PDコントローラの比例ゲイン
KD = np.diag([1.0, 1.0])    # PDコントローラの微分ゲイン
dt = 0.01                   # 積分時間ステップ (10ms) 0.005あたりがいいかも

# 順運動学に基づく目標先端位置
def forward_kinematics(angles):
    x = link_lengths[0] * np.cos(angles[0]) + link_lengths[1] * np.cos(angles[0] + angles[1])
    y = link_lengths[0] * np.sin(angles[0]) + link_lengths[1] * np.sin(angles[0] + angles[1])
    return np.array([x, y])

def forward_kinematics_robot(angles):
    x = link_lengths[0] * np.cos(angles[0]+np.pi/20) + link_lengths[1] * np.cos(angles[0]+np.pi/20 + (np.pi-np.pi/12-angles[1]))
    y = link_lengths[0] * np.sin(angles[0]+np.pi/20) + link_lengths[1] * np.sin(angles[0]+np.pi/20 + (np.pi-np.pi/12-angles[1]))
    return np.array([x, y])

# 先端位置を計算
start_position = forward_kinematics_robot(start_angle)
end_position = forward_kinematics_robot(end_angle)

# 保存先ディレクトリ
user_name = os.getlogin()
data_directory = f"C:\\Users\\{user_name}\\OneDrive\\Python\\reservoir\\data"

# **Dynamixelデータの読み込み**
data = np.load(os.path.join(data_directory, "dynamixel_onecircle_data.npz"))
angles = data['angles']  # 角度データ（2次元のみを使用）

# 角度データが確実に2次元配列になるように確認
if angles.ndim == 1:  # 1次元配列ならリシェイプ
    teaching_trajectories = angles.reshape(-1, 2)
else:
    teaching_trajectories = angles  # そのまま使用

teaching_torques = np.gradient(teaching_trajectories, axis=0)  # 簡易的なトルクとして速度を利用

print("teaching_trajectories shape:", teaching_trajectories.shape)

steps = teaching_trajectories.shape[0]*2  # ティーチングデータのステップ数を利用

# ESNを初期化し重みを読み込む
N_u = 2
N_y = 2
esn = ESN(N_u, N_y, N_x, density=density, input_scale=input_scale, rho=rho,
          activation_func=activation_function, leaking_rate=leaking_rate)
esn.Output.Wout = np.load(os.path.join(data_directory,"esn_weights_onecircle.npy"))

# 新しい初期点から目標位置への軌道生成（自律走行）とトルク計算
start_time = time.time()
np.random.seed(4)
test_initial_positions_angles = [np.random.uniform(np.pi/3, np.pi-np.pi/6, size=2) for _ in range(5)]
esn_trajectories = []
esn_torques = []

esn_real_trajectories = []
esn_trajectories_error = []

for pos in test_initial_positions_angles:
    esn.reset_states()  # ESNの内部状態をリセット
    esn_predicted_angle_seq , esn_torque_seq, esn_real_angle_seq, esn_error_seq = esn.trajectory_angle(pos, steps, KP, KD, dt)
    
    esn_trajectories.append(np.array(esn_predicted_angle_seq))
    esn_torques.append(np.array(esn_torque_seq))
    esn_real_trajectories.append(np.array(esn_real_angle_seq))
    esn_trajectories_error.append(np.array(esn_error_seq))
trajectory_generation_time = time.time() - start_time
print(f"ESN trajectory generation time: {trajectory_generation_time:.4f} seconds")

for idx, traj in enumerate(esn_real_trajectories):
    final_position = traj[-1]  # 軌道の最終ステップの位置
    print(f"Trajectory {idx+1} final position: {final_position}")
    
# カラーマップを用意
cmap = plt.cm.rainbow
esn_colors = cmap(np.linspace(0, 1, len(esn_trajectories)))
teaching_colors = cmap(np.linspace(0, 1, len(teaching_trajectories)))

# 4つのプロットのセットアップ
fig, axs = plt.subplots(1, 3, figsize=(15, 10))
axs = axs.flatten()  # axsを1次元配列に変換

# target_angleの時間経過を計算
target_angles_over_time = [dynamic_target_angle_circular_time(step, steps, total_revolutions, center_angle, radius) for step in range(steps)]
target_angles_over_time = np.array(target_angles_over_time)
target_positions_over_time = np.array([forward_kinematics_robot(angle) for angle in target_angles_over_time])

# 1. ESNによる関節角度の動作生成
axs[0].set_title("ESN-based Adaptive Motion Generation (angles)")
for idx, traj in enumerate(esn_real_trajectories):
    axs[0].plot(traj[:, 0], traj[:, 1], color=esn_colors[idx], alpha=0.7)
    axs[0].scatter(traj[0, 0], traj[0, 1], color='green', marker='o')  # 初期位置
axs[0].plot(teaching_trajectories[:, 0], teaching_trajectories[:, 1], color='blue', alpha=0.7,linewidth=0.5)
axs[0].scatter(teaching_trajectories[0, 0], teaching_trajectories[0, 1], color='green', marker='o', label="Start")
axs[0].set_xlabel("Joint Angles1")
axs[0].set_ylabel("Joint Angles2")
axs[0].legend()

# 2. ESNによるXY座標の動作生成
axs[1].set_title("ESN-based Adaptive Motion Generation (XY)")
for idx, traj in enumerate(esn_real_trajectories):
    xy_traj = np.array([forward_kinematics_robot(angles) for angles in traj])  # 角度からXYに変換
    axs[1].plot(xy_traj[:, 0], xy_traj[:, 1], color=esn_colors[idx], alpha=0.7)
    axs[1].scatter(xy_traj[0, 0], xy_traj[0, 1], color='green', marker='o')  # 初期位置
axs[1].scatter(0, 0, color='black', marker='o', label="Origin")
axs[1].scatter(start_position[0], start_position[1], color='black', marker='o', label="Start Position")
axs[1].scatter(end_position[0], end_position[1], color='black', marker='x', label="End Position")
axs[1].plot(target_positions_over_time[:, 0], target_positions_over_time[:, 1], 'k--', label="Target Position Path Over Time")
axs[1].scatter(0, 0, color='black', marker='o', label="Origin")
axs[1].set_xlabel("X Position")
axs[1].set_ylabel("Y Position")

# 3. ESNの現在角度1の時間変化
axs[2].set_title("ESN Current Angle 1 over Time")
for idx, real_traj in enumerate(esn_real_trajectories):
    axs[2].plot(real_traj[:, 0], color=esn_colors[idx], alpha=0.7, linestyle=":", label=f"Current Angle 1 - Traj {idx+1}")
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Current Angle 1")

# # 4. ESNによるトルクの時間変化
# axs[3].set_title("ESN Torque over Time")
# for idx, torque in enumerate(esn_torques):
#     axs[3].plot(torque[:, 0], color=esn_colors[idx], label=f"Joint 1 - Trajectory {idx+1}")

# axs[3].set_xlabel("Time Step")
# axs[3].set_ylabel("Torque")

# # 5. ティーチング・プレイバック法（関節角度）
# axs[4].set_title("Teaching Trajectories (angles)")
# axs[4].plot(teaching_trajectories[:, 0], teaching_trajectories[:, 1], color='blue', alpha=0.7)
# axs[4].scatter(teaching_trajectories[0, 0], teaching_trajectories[0, 1], color='green', marker='o', label="Start")  # 初期位置
# axs[4].set_xlabel("Joint Angles1")
# axs[4].set_ylabel("Joint Angles2")
# axs[4].legend()

# # 6. ティーチング・プレイバック法（XY座標）
# axs[5].set_title("Teaching Trajectories (XY)")
# xy_traj = np.array([forward_kinematics_robot(angles) for angles in teaching_trajectories])  # 全データを一括処理
# axs[5].plot(xy_traj[:, 0], xy_traj[:, 1], color='blue', alpha=0.7)
# axs[5].scatter(xy_traj[0, 0], xy_traj[0, 1], color='green', marker='o')  # 初期位置
# axs[5].scatter(0, 0, color='black', marker='o', label="Origin")
# axs[5].set_xlabel("X Position")
# axs[5].set_ylabel("Y Position")
# axs[5].legend()

# # 7. ティーチング時の現在角度1の時間変化
# axs[6].set_title("Teaching Current Angle 1 over Time")
# for idx, traj in enumerate(teaching_trajectories):  # traj は 1 行ずつ（1ステップ分）を表す
#     axs[6].plot(np.arange(len(teaching_trajectories)), teaching_trajectories[:, 0], color=teaching_colors[idx % len(teaching_colors)], alpha=0.7)
# axs[6].set_xlabel("Time Step")
# axs[6].set_ylabel("Current Angle 1")

# # 8. ティーチングデータにおけるトルクの時間変化
# axs[7].set_title("Teaching Torque over Time")
# axs[7].plot(np.arange(len(teaching_torques)), teaching_torques[:, 0], color='blue', label="Joint 1", alpha=0.7)
# axs[7].plot(np.arange(len(teaching_torques)), teaching_torques[:, 1], color='green', label="Joint 2", alpha=0.7)
# axs[7].set_xlabel("Time Step")
# axs[7].set_ylabel("Torque")
# axs[7].legend()

plt.tight_layout()
plt.show()

# # ESNで生成された一つの軌道を選択
# chosen_trajectory = esn_trajectories[0]

# # アニメーションの生成
# print("Generating ESN trajectory animation...")
# animate_esn_trajectory(
#     esn_trajectory=chosen_trajectory,       # アニメーション用のESN生成軌道
#     link_lengths=link_lengths,             # アームのリンク長
#     target_positions_over_time=target_positions_over_time,  # ターゲットのXY座標（時間ごとに変化）
#     steps=steps                             # シミュレーションステップ数
# )

# print("Animation complete.")
