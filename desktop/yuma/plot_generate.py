import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['font.family'] = 'MS Gothic'

user_name = os.getlogin()
# 保存したデータのディレクトリ
data_directory_result = f"C:\\Users\\{user_name}\\OneDrive\\研究\\卒論\\reservoir卒論\\log"

# # 最新の circle_control_data_x.npz を取得
# files = [f for f in os.listdir(data_directory) if f.startswith("circle_control_data_") and f.endswith(".npz")]
# files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # 数値順にソート
# latest_file = os.path.join(data_directory, files[-1])  # 最新のファイルを選択

file_path = os.path.join(data_directory_result, "circle_control_data_500.npz")
#300 700  400 500

if os.path.exists(file_path):
    print(f"ファイルは存在します: {file_path}")
else:
    print(f"ファイルが見つかりません: {file_path}")

# ファイルの存在チェック
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} が存在しません")

# データ読み込み
data = np.load(file_path)
current_angles = data["current_angles"]
predicted_angles = data["predicted_angles"]

print(f"Loaded data from {file_path}")

data_directory = f"C:\\Users\\{user_name}\\OneDrive\\研究\\卒論\\reservoir卒論\\data"

plt.rcParams.update({'font.size': 17})

# 1つ目のプロット (Time Step vs Angles)
# fig1, ax1 = plt.subplots(figsize=(6.8, 5))  # 片方のグラフのサイズを調整
fig1, ax1 = plt.subplots(figsize=(3.2, 5))  # 片方のグラフのサイズを調整

ax1.plot(np.array(current_angles)[:, 0], label=r"$q_1$(実際の角度)", color="blue", zorder=2, linewidth=1.8)
ax1.plot(np.array(predicted_angles)[:, 0], label=r"$q_1$(生成軌道)", color="red", zorder=1 , linewidth=1.8, linestyle=(0, (3, 0.5)))  # 点線
# ax1.plot(np.array(current_angles)[:, 1], label=r"$q_2$", color="red", linewidth=1.8)
# ax1.plot(np.array(predicted_angles)[:, 1], label=r"$q_2$ (Predict)", color="orange", linestyle=(0, (3, 0.5)))  # 点線

ax1.set_yticks([1.0, 1.5, 2.0])
ax1.set_xticks(np.arange(0, len(current_angles), 50))
# ax1.set_xticks([0, 500, 1000])

ax1.legend(
    loc="upper left", 
    borderpad=0.3, 
    handletextpad=0.3, 
    labelspacing=0.3, 
    handlelength=1.5,
    fontsize=14
)
ax1.set_xlabel("Time Step")
ax1.set_ylabel(r"$q_1$ $\mathrm{[rad]}$")  # 数学表記で [rad]


plt.tight_layout()
plt.savefig("plot_time_vs_angles.pdf", format="pdf", bbox_inches="tight", pad_inches=0)

plt.show()



# # 2つ目のプロット (q1 vs q2)
# # fig2, ax2 = plt.subplots(figsize=(6, 5))  # 片方のグラフのサイズを調整
# # fig2, ax2 = plt.subplots(figsize=(6, 3.5))  # 片方のグラフのサイズを調整
# fig2, ax2 = plt.subplots(figsize=(5.2, 3.5))  # 片方のグラフのサイズを調整

# data = np.load(os.path.join(data_directory, "dynamixel_onecircle_data.npz"))
# angles = data['angles']  # 角度データ（2次元のみを使用）
# current_angles_array = np.array(current_angles)
# colors = cm.rainbow(np.linspace(0, 1, len(current_angles_array) - 1))

# for i in range(len(current_angles_array) - 1):
#     ax2.plot(
#         [current_angles_array[i, 0], current_angles_array[i + 1, 0]],  # x座標を結ぶ
#         [current_angles_array[i, 1], current_angles_array[i + 1, 1]],  # y座標を結ぶ
#         color=colors[i],  # カラーマップから取得した色を適用
#         linewidth=3.8     # 線の太さを調整
#     )

# sc = ax2.scatter(
#     current_angles_array[:, 0],  # 横軸: Joint 1 Angle
#     current_angles_array[:, 1],  # 縦軸: Joint 2 Angle
#     c=np.arange(len(current_angles)),  # 色: ステップ数
#     cmap="rainbow",  # カラーマップ: Viridis
#     edgecolor="none",
#     s=20 * 0.95
# )

# # 軸ラベルを数学記号で統一
# ax2.set_xlabel(r"$q_1$ $\mathrm{[rad]}$")
# ax2.set_ylabel(r"$q_2$ $\mathrm{[rad]}$")

# ax2.set_xticks([1.0, 1.5, 2.0])
# ax2.set_yticks([1.0, 1.5, 2.0])

# # cbar = plt.colorbar(sc, ax=ax2, aspect=25, pad=0.03)
# # cbar.set_ticks([0, 500, 1000])  # カラーバーの刻み
# # cbar.ax.set_xlabel("Timestep", fontsize=15, labelpad=10, rotation=0)  # 横向き & 下に配置
# # cbar.ax.xaxis.set_label_coords(1.0, -0.05)  # (横方向, 縦方向)
# # cbar.ax.tick_params(labelsize=14)

# ax2.scatter(current_angles_array[0, 0], current_angles_array[0, 1], color='purple', marker='o', label="開始点", zorder=3)  # 初期位置
# ax2.scatter(current_angles_array[-1, 0], current_angles_array[-1, 1], color='red', marker='o', label="終了点", zorder=3)  # 終了位置
# ax2.plot(angles[:, 0], angles[:, 1], color='black', label="目標軌道", linewidth=1.8)
# ax2.set_ylim(0.98, 1.65)

# # ax2.scatter(current_angles_array[0, 0], current_angles_array[0, 1], color='purple', marker='o', label="start of trajectory", zorder=3)  # 初期位置
# # ax2.scatter(current_angles_array[-1, 0], current_angles_array[-1, 1], color='red', marker='o', label="end of trajectory", zorder=3)  # 終了位置
# # ax2.plot(angles[:, 0], angles[:, 1], color='black', label="target trajectory")

# # ax2.scatter(angles[0, 0], angles[0, 1], color='black', marker='o', label="Start of reference trajectory", zorder=3)  # 初期位置
# # ax2.scatter(angles[-1, 0], angles[-1, 1], color='black', marker='x', label="End of reference trajectory", zorder=3)  # 終了位置

# ax2.legend(
#     borderpad=0.3, 
#     handletextpad=0.3, 
#     labelspacing=0.3, 
#     handlelength=1.5,
#     fontsize=17
# )

# plt.tight_layout()
# plt.savefig("plot_q1_vs_q2.pdf")  # 2枚目のプロットを保存
# plt.show()




fig2, ax2 = plt.subplots(figsize=(5.0, 3.5))  # 片方のグラフのサイズを調整

data = np.load(os.path.join(data_directory, "dynamixel_onecircle_data.npz"))
angles = data['angles']  # 角度データ（2次元のみを使用）
start_step = 450
end_step = 580
current_angles_array = np.array(current_angles)
current_angles_sub = current_angles_array[start_step:end_step + 1]
colors_sub = cm.rainbow(np.linspace(0, 1, len(current_angles_sub) - 1))

for i in range(len(current_angles_sub) - 1):
    ax2.plot(
        [current_angles_sub[i, 0], current_angles_sub[i + 1, 0]],
        [current_angles_sub[i, 1], current_angles_sub[i + 1, 1]],
        color=colors_sub[i],
        linewidth=3.8
    )

# カラーバー用の scatter を用意
sc = ax2.scatter(
    current_angles_sub[:, 0],
    current_angles_sub[:, 1],
    c=np.arange(start_step, end_step + 1),
    cmap="rainbow",
    edgecolor="none",
    s=20
)

# 軸ラベルを数学記号で統一
ax2.set_xlabel(r"$q_1$ $\mathrm{[rad]}$", fontsize=16)
ax2.set_ylabel(r"$q_2$ $\mathrm{[rad]}$", fontsize=16)
ax2.yaxis.set_label_coords(-0.12, 0.45)  

ax2.tick_params(axis='both', labelsize=16)

ax2.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8])
ax2.set_yticks([1.0, 1.2, 1.4, 1.6])
ax2.set_xlim(1.2, 1.95)
ax2.set_ylim(1.0, 1.38)

# 目標軌道と開始・終了点
ax2.plot(angles[:, 0], angles[:, 1], color='black', label="目標軌道", linewidth=1.8)

# カラーバーの追加（ステップ番号で表示）
cbar = plt.colorbar(sc, ax=ax2, aspect=25, pad=0.03)
cbar.set_ticks([450, 580])
cbar.ax.set_xlabel("Timestep", fontsize=15, labelpad=10, rotation=0)
cbar.ax.xaxis.set_label_coords(1.0, -0.05)
cbar.ax.tick_params(labelsize=14)

ax2.legend(
    borderpad=0.3, 
    handletextpad=0.3, 
    labelspacing=0.3, 
    handlelength=1.5,
    fontsize=15
)

plt.tight_layout()
plt.savefig("plot_q1_vs_q2.pdf")  # 2枚目のプロットを保存
plt.show()









# # グラフ描画
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# # 1. 現在の関節角度とトルクのプロット
# # axs[0].set_title("Joint Angles Over Time")
# axs[0].plot(np.array(current_angles)[:, 0], label=r"$q_1$ (Actual)", color="blue")
# axs[0].plot(np.array(predicted_angles)[:, 0], label=r"$q_1$ (Target)", color="cyan", linestyle=(0, (3, 0.5)))  # さらに細かい点線
# axs[0].plot(np.array(current_angles)[:, 1], label=r"$q_2$ (Actual)", color="red")
# axs[0].plot(np.array(predicted_angles)[:, 1], label=r"$q_2$ (Target)", color="orange", linestyle=(0, (3, 0.5)))  # さらに細かい点線

# axs[0].legend()
# axs[0].set_xlabel("Time Step")
# axs[0].set_ylabel(r"Angle $\mathrm{[rad]}$")  # 数学表記で [rad]

# # 4. Joint 1 vs Joint 2 with color gradient
# data = np.load(os.path.join(data_directory, "dynamixel_onecircle_data.npz"))
# angles = data['angles']  # 角度データ（2次元のみを使用）
# current_angles_array = np.array(current_angles)
# colors = cm.rainbow(np.linspace(0, 1, len(current_angles_array) - 1))

# for i in range(len(current_angles_array) - 1):
#     axs[1].plot(
#         [current_angles_array[i, 0], current_angles_array[i + 1, 0]],  # x座標を結ぶ
#         [current_angles_array[i, 1], current_angles_array[i + 1, 1]],  # y座標を結ぶ
#         color=colors[i],  # カラーマップから取得した色を適用
#         linewidth=3.5     # 線の太さを少し細く設定
#     )

# sc = axs[1].scatter(
#     current_angles_array[:, 0],  # 横軸: Joint 1 Angle
#     current_angles_array[:, 1],  # 縦軸: Joint 2 Angle
#     c=np.arange(len(current_angles)),  # 色: ステップ数
#     cmap="rainbow",  # カラーマップ: Viridis
#     edgecolor="none",
#     s=20 * 0.85
# )

# # 軸ラベルを数学記号で統一
# axs[1].set_xlabel(r"$q_1$ $\mathrm{[rad]}$")
# axs[1].set_ylabel(r"$q_2$ $\mathrm{[rad]}$")

# plt.colorbar(sc, ax=axs[1], label="Time Step")  # カラーバーを追加
# axs[1].scatter(current_angles_array[0, 0], current_angles_array[0, 1], color='purple', marker='o', label="Start of actual trajectory", zorder=3)  # 初期位置
# axs[1].scatter(current_angles_array[-1, 0], current_angles_array[-1, 1], color='red', marker='o', label="End of actual trajectory", zorder=3)  # 終了位置
# axs[1].plot(angles[:, 0], angles[:, 1], color='black', label="desired trajectory")

# axs[1].scatter(angles[0, 0], angles[0, 1], color='black', marker='o', label="Start of desired trajectory", zorder=3)  # 初期位置
# axs[1].scatter(angles[-1, 0], angles[-1, 1], color='black', marker='x', label="End of desired trajectory", zorder=3)  # 終了位置

# axs[1].legend()

# plt.tight_layout()
# plt.show()
