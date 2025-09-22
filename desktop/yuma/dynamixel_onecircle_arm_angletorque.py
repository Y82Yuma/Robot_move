import numpy as np
import time
from dynamixel_sdk import *  # Dynamixel SDKのクラスをインポート
from model_new import ESN, dynamic_target_angle_circular_time
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

# Dynamixelの設定
ADDR_GOAL_POSITION = 116  # ゴール位置
ADDR_GOAL_CURRENT = 102  # トルク（電流）を設定するアドレス
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
PROTOCOL_VERSION = 2.0
DXL_ID1 = 1  # モータID
DXL_ID2 = 2  # モータID
BAUDRATE = 57600
DEVICENAME = "COM3"
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
CURRENT_BASED_POSITION_MODE = 5  # Current-based Position Control Mode

# トルク制限設定 (電流値)
MAX_CURRENT = 1193  # XM430-W350の最大電流
TORQUE_LIMIT = int(MAX_CURRENT *0.1)  # トルクを制限

# ロボットアームのパラメータ
ANGLE_LIMIT_MIN = 10  # 最小角度 (10度)
ANGLE_LIMIT_MAX = 150  # 最大角度 (150度)
STEP_LIMIT = 0.5  # 1ステップで移動できる最大角度 (ラジアン)
dt = 0.045  # 推定ステップ時間（秒）

# Dynamixel初期化
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    raise Exception("ポートを開けませんでした。")
if not portHandler.setBaudRate(BAUDRATE):
    raise Exception("ボーレートを設定できませんでした。")

for DXL_ID in [DXL_ID1, DXL_ID2]:
    # トルクを無効化
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    # モードをCurrent-based Position Control Modeに変更
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, CURRENT_BASED_POSITION_MODE)
    # トルク制限値（目標電流）を設定
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, TORQUE_LIMIT)
    # トルクを有効化
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

print("Dynamixel initialized with torque limit.")

# 関数: ラジアンをDynamixelの位置値に変換
def radian_to_position(radian):
    position = int(radian * 4096 / (2 * np.pi))
    return max(0, min(position, 4095))


# 関数: 位置値をラジアンに変換
def position_to_radian(position):
    return (position / 4096) * 2 * np.pi

# ESNの設定
N_x = 200  # リザバーのノード数
input_scale = 0.7
rho = 0.99
leaking_rate = 0.7
activation_function = np.tanh
esn = ESN(2, 2, N_x, density=0.1, input_scale=input_scale, rho=rho,
          activation_func=activation_function, leaking_rate=leaking_rate)

# 学習済みの重みをロード
user_name = os.getlogin()
data_directory = f"C:\\Users\\{user_name}\\OneDrive\\Python\\reservoir\\data"
data = np.load(os.path.join(data_directory, "dynamixel_onecircle_data.npz"))
angles = data['angles']  # 角度データ（2次元のみを使用）
esn.Output.Wout = np.load(os.path.join(data_directory, "esn_weights_onecircle.npy"))
print("Loaded ESN weights.")

# データ保存用リスト
current_angles = []
predicted_angles = []

# 初期位置の読み取り
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4)
groupSyncRead.addParam(DXL_ID1)
groupSyncRead.addParam(DXL_ID2)

groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4)

try:
    start_time = time.time()
    step_count = 0

    # 初期位置の取得
    groupSyncRead.txRxPacket()
    position1 = groupSyncRead.getData(DXL_ID1, ADDR_PRESENT_POSITION, 4)
    position2 = groupSyncRead.getData(DXL_ID2, ADDR_PRESENT_POSITION, 4)
    current_angle = np.array([position_to_radian(position1), position_to_radian(position2)])
    print(f"Initial Position: {current_angle}")

    # 制御ループ
    while step_count < 1000:  # 30秒間制御
        step_count += 1
        # ESNで次の角度を予測
        x_in = esn.Input(current_angle)
        x = esn.Reservoir(x_in)
        predicted_angle = esn.Output(x)
        if step_count == 1:
            predicted_angle = np.clip(predicted_angle, current_angle - 0.02, current_angle + 0.02)
        elif 20> step_count:
            predicted_angle = np.clip(predicted_angle, predicted_angles[-1] - 0.02, predicted_angles[-1] + 0.02)
        print(f"Step: {step_count}")
        print(f"Current Angle: {current_angle}")
        print(f"Predicted Angle: {predicted_angle}")
        
        current_angles.append(current_angle)
        predicted_angles.append(predicted_angle)

        # 目標角度が範囲外の場合スキップ
        if np.any(predicted_angle < np.deg2rad(ANGLE_LIMIT_MIN)) or np.any(predicted_angle > np.deg2rad(ANGLE_LIMIT_MAX)):
            print(f"Step {step_count}: Predicted angle out of bounds. Skipping.")
            continue

        # 移動量を制限
        target_angle = np.clip(predicted_angle, current_angle - STEP_LIMIT, current_angle + STEP_LIMIT)
    
        # Dynamixelに送信する位置値に変換
        goal_positions = [radian_to_position(target_angle[0]), radian_to_position(target_angle[1])]
        for idx, DXL_ID in enumerate([DXL_ID1, DXL_ID2]):
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal_positions[idx])),
                                    DXL_HIBYTE(DXL_LOWORD(goal_positions[idx])),
                                    DXL_LOBYTE(DXL_HIWORD(goal_positions[idx])),
                                    DXL_HIBYTE(DXL_HIWORD(goal_positions[idx]))]
            groupSyncWrite.addParam(DXL_ID, param_goal_position)

        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()

        # 現在の位置を更新
        time.sleep(dt)
        groupSyncRead.txRxPacket()
        position1 = groupSyncRead.getData(DXL_ID1, ADDR_PRESENT_POSITION, 4)
        position2 = groupSyncRead.getData(DXL_ID2, ADDR_PRESENT_POSITION, 4)
        current_angle = np.array([position_to_radian(position1), position_to_radian(position2)])

except KeyboardInterrupt:
    print("Control interrupted by user.")

finally:
    # トルク無効化とポートの閉鎖
    for DXL_ID in [DXL_ID1, DXL_ID2]:
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    portHandler.closePort()
    print("Port closed.")
    print("Time elapsed:", time.time() - start_time)

    result_file = os.path.join(data_directory, "arm_control_current_angles.npz")
    np.savez(result_file, current_angles=current_angles)

    # 結果を保存
    np.savez("arm_control_result_torque.npz", current_angles=current_angles, predicted_angles=predicted_angles)
    print("Results saved.")

    current_angles = np.array(current_angles)  # リストをnumpy配列に変換
    predicted_angles = np.array(predicted_angles)

    center_angle = np.array([np.pi / 2, np.pi / 2])  # 円の中心位置
    radius = np.pi/6                                    # 円軌道の半径
    total_revolutions = 1                           # 総回転数（2周回る）
    start_angle = np.array([np.pi / 2 + np.pi/6, np.pi / 2])
    end_angle = np.array([np.pi / 2 + np.pi/6, np.pi / 2])
    steps = 500
    # target_angleの時間経過を計算
    target_angles_over_time = [dynamic_target_angle_circular_time(step, steps, total_revolutions, center_angle, radius) for step in range(steps)]
    target_angles_over_time = np.array(target_angles_over_time)

    # グラフ描画
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # 1. 現在の関節角度とトルクのプロット
    # axs[0].set_title("Joint Angles Over Time")
    axs[0].plot(np.array(current_angles)[:, 0], label=r"$q_1$ (Actual)", color="blue")
    axs[0].plot(np.array(predicted_angles)[:, 0], label=r"$q_1$ (Target)", color="cyan", linestyle=(0, (3, 0.5)))  # さらに細かい点線
    axs[0].plot(np.array(current_angles)[:, 1], label=r"$q_2$ (Actual)", color="red")
    axs[0].plot(np.array(predicted_angles)[:, 1], label=r"$q_2$ (Target)", color="orange", linestyle=(0, (3, 0.5)))  # さらに細かい点線

    axs[0].legend()
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel(r"Angle $\mathrm{[rad]}$")  # 数学表記で [rad]

    # 4. Joint 1 vs Joint 2 with color gradient
    data = np.load(os.path.join(data_directory, "dynamixel_onecircle_data.npz"))
    angles = data['angles']  # 角度データ（2次元のみを使用）
    current_angles_array = np.array(current_angles)
    colors = cm.rainbow(np.linspace(0, 1, len(current_angles_array) - 1))

    for i in range(len(current_angles_array) - 1):
        axs[1].plot(
            [current_angles_array[i, 0], current_angles_array[i + 1, 0]],  # x座標を結ぶ
            [current_angles_array[i, 1], current_angles_array[i + 1, 1]],  # y座標を結ぶ
            color=colors[i],  # カラーマップから取得した色を適用
            linewidth=3.5     # 線の太さを少し細く設定
        )

    sc = axs[1].scatter(
        current_angles_array[:, 0],  # 横軸: Joint 1 Angle
        current_angles_array[:, 1],  # 縦軸: Joint 2 Angle
        c=np.arange(len(current_angles)),  # 色: ステップ数
        cmap="rainbow",  # カラーマップ: Viridis
        edgecolor="none",
        s=20 * 0.85
    )

    # 軸ラベルを数学記号で統一
    axs[1].set_xlabel(r"$q_1$ $\mathrm{[rad]}$")
    axs[1].set_ylabel(r"$q_2$ $\mathrm{[rad]}$")

    plt.colorbar(sc, ax=axs[1], label="Time Step")  # カラーバーを追加
    axs[1].scatter(current_angles_array[0, 0], current_angles_array[0, 1], color='purple', marker='o', label="Start of actual trajectory", zorder=3)  # 初期位置
    axs[1].scatter(current_angles_array[-1, 0], current_angles_array[-1, 1], color='red', marker='o', label="End of actual trajectory", zorder=3)  # 終了位置
    axs[1].plot(angles[:, 0], angles[:, 1], color='black', label="desired trajectory")

    axs[1].scatter(angles[0, 0], angles[0, 1], color='black', marker='o', label="Start of desired trajectory", zorder=3)  # 初期位置
    axs[1].scatter(angles[-1, 0], angles[-1, 1], color='black', marker='x', label="End of desired trajectory", zorder=3)  # 終了位置

    axs[1].legend()

    plt.tight_layout()
    plt.show()


    data_directory_result = f"C:\\Users\\{user_name}\\OneDrive\\Python\\reservoir\\log"

    # 連番ファイル名を決定
    i = 1
    while os.path.exists(os.path.join(data_directory_result, f"circle_control_data_{i}.npz")):
        i += 1

    filename = os.path.join(data_directory_result, f"circle_control_data_{i}.npz")

    current_angles = np.array(current_angles)
    predicted_angles = np.array(predicted_angles)

    # 6次元データを保存
    np.savez(filename, 
            current_angles=current_angles, 
            predicted_angles=predicted_angles)

    print(f"Results saved as {filename}")

