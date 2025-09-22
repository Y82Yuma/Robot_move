import numpy as np
from model_new_delay import ESN, Tikhonov
import os

# 保存先ディレクトリ
user_name = os.getlogin()
data_directory = f"C:\\Users\\{user_name}\\OneDrive\\Python\\reservoir\\data"

# データファイルを読み込み
data_file = os.path.join(data_directory, "dynamixel_onecircle_data.npz")
data = np.load(data_file)
angles = data['angles']

# 最初のステップを複製して新しいuを作成
angle_initial = angles[10:1]  # 最初の1ステップを取得
angle_prefix = np.tile(angle_initial, (0, 1))  # 5ステップ分複製
angle_rest = angles[0:]  # 1ステップ目以降のデータ
angle_total = np.vstack((angle_prefix, angle_rest))  # 複製したデータと元のデータを結合

# 教師データの作成
u = angle_total[:-1]  # 入力データ (t時刻の関節角度)
d = angle_total[1:]   # 出力データ (t+1時刻の関節角度)
print(f"Training data shape: u={u.shape}, d={d.shape}")

# ESNの設定
N_x = 200               # リザバーのノード数
input_scale = 0.7       # 入力スケーリング
density = 0.1           # 結合密度
rho = 0.99              # スペクトル半径
leaking_rate = 0.7     # リーキング率
beta = 0.00001           # リッジ回帰の正則化係数
activation_function = np.tanh # 活性化関数
num_epochs = 10         # トレーニングを繰り返す回数
gamma = 0    #遅延項のスケール係数
tau = 5                 #無駄時間
fb_scale = 0         # フィードバックスケーリング
fb_delay_scale = 0 # フィードバック遅延スケーリング

# ESNの初期化
N_u = u.shape[1]  # 入力次元
N_y = d.shape[1]  # 出力次元
esn = ESN(N_u, N_y, N_x, density=density, input_scale=input_scale, rho=rho, activation_func=activation_function, leaking_rate=leaking_rate, 
          fb_scale=fb_scale, fb_delay_scale=fb_delay_scale, gamma=gamma, tau=tau)

# Tikhonov正則化(リッジ回帰)による学習
optimizer = Tikhonov(N_x, N_y, beta)

# 学習プロセス
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    esn.train(u, d, optimizer)

print("Training complete.")

# 学習後の重みを保存
weights_file = os.path.join(data_directory, "esn_weights_onecircle_delay.npy")
np.save(weights_file, esn.Output.Wout)
print(f"Weights saved as {weights_file}.")
