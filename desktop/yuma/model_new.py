import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from numpy.linalg import pinv

# 恒等写像
def identity(x):
    return x

# 入力層
class Input:
    # 入力結合重み行列Winの初期化
    def __init__(self, N_u, N_x, input_scale, seed=0):
        '''
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param input_scale: 入力スケーリング
        '''
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    # 入力結合重み行列Winによる重みづけ
    def __call__(self, u):
        '''
        param u: N_u次元のベクトル
        return: N_x次元のベクトル
        '''
        return np.dot(self.Win, u)

# リザバー
class Reservoir:
    # リカレント結合重み行列Wの初期化
    def __init__(self, N_x, density, rho, activation_func, leaking_rate,
                 seed=0):
        '''
        param N_x: リザバーのノード数
        param density: ネットワークの結合密度
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: ノードの活性化関数
        param leaking_rate: leaky integratorモデルのリーク率
        param seed: 乱数の種
        '''
        self.seed = seed
        self.W = self.make_connection(N_x, density, rho)
        self.x = np.zeros(N_x)  # リザバー状態ベクトルの初期化
        self.activation_func = activation_func
        self.alpha = leaking_rate

    # リカレント結合重み行列の生成
    def make_connection(self, N_x, density, rho):
        # Erdos-Renyiランダムグラフ
        m = int(N_x*(N_x-1)*density/2)  # 総結合数
        G = nx.gnm_random_graph(N_x, m, self.seed)

        # 行列への変換(結合構造のみ）
        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        # 非ゼロ要素を一様分布に従う乱数として生成
        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # スペクトル半径の計算
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        # 指定のスペクトル半径rhoに合わせてスケーリング
        W *= rho / sp_radius

        return W

    # リザバー状態ベクトルの更新
    def __call__(self, x_in):
        '''
        param x_in: 更新前の状態ベクトル
        return: 更新後の状態ベクトル
        '''
        #self.x = self.x.reshape(-1, 1)
        self.x = (1.0 - self.alpha) * self.x \
                 + self.alpha * self.activation_func(np.dot(self.W, self.x) \
                 + x_in)
        return self.x

    # リザバー状態ベクトルの初期化
    def reset_reservoir_state(self):
        self.x *= 0.0


# 出力層
class Output:
    # 出力結合重み行列の初期化
    def __init__(self, N_x, N_y, seed=0):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param seed: 乱数の種
        '''
        # 正規分布に従う乱数
        np.random.seed(seed=seed)
        self.Wout = np.random.normal(size=(N_y, N_x))

    # 出力結合重み行列による重みづけ
    def __call__(self, x):
        '''
        param x: N_x次元のベクトル
        return: N_y次元のベクトル
        '''
        return np.dot(self.Wout, x)

    # 学習済みの出力結合重み行列を設定
    def setweight(self, Wout_opt):
        self.Wout = Wout_opt


# 出力フィードバック
class Feedback:
    # フィードバック結合重み行列の初期化
    def __init__(self, N_y, N_x, fb_scale, seed=0):
        '''
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param fb_scale: フィードバックスケーリング
        param seed: 乱数の種
        '''
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        self.Wfb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))

    # フィードバック結合重み行列による重みづけ
    def __call__(self, y):
        '''
        param y: N_y次元のベクトル
        return: N_x次元のベクトル
        '''
        return np.dot(self.Wfb, y)


# Moore-Penrose擬似逆行列
class Pseudoinv:
    def __init__(self, N_x, N_y):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        '''
        self.X = np.empty((N_x, 0))
        self.D = np.empty((N_y, 0))
        
    # 状態集積行列および教師集積行列の更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X = np.hstack((self.X, x))
        self.D = np.hstack((self.D, d))
        
    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self):
        Wout_opt = np.dot(self.D, np.linalg.pinv(self.X))
        return Wout_opt


# リッジ回帰（beta=0のときは線形回帰）
class Tikhonov:
    def __init__(self, N_x, N_y, beta):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param beta: 正則化パラメータ
        '''
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_x))
        self.N_x = N_x

    # 学習用の行列の更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)

    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self):
        X_pseudo_inv = np.linalg.inv(self.X_XT \
                                     + self.beta*np.identity(self.N_x))
        Wout_opt = np.dot(self.D_XT, X_pseudo_inv)
        return Wout_opt


# 逐次最小二乗（RLS）法
class RLS:
    def __init__(self, N_x, N_y, delta, lam, update):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param delta: 行列Pの初期条件の係数（P=delta*I, 0<delta<<1）
        param lam: 忘却係数 (0<lam<1, 1に近い値)
        param update: 各時刻での更新繰り返し回数
        '''
        self.delta = delta
        self.lam = lam
        self.update = update
        self.P = (1.0/self.delta)*np.eye(N_x, N_x) 
        self.Wout = np.zeros([N_y, N_x])
        
    # Woutの更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        for i in np.arange(self.update):
            v = d - np.dot(self.Wout, x)
            gain = (1/self.lam*np.dot(self.P, x))
            gain = gain/(1+1/self.lam*np.dot(np.dot(x.T, self.P), x))
            self.P = 1/self.lam*(self.P-np.dot(np.dot(gain, x.T), self.P))
            self.Wout += np.dot(v, gain.T)

        return self.Wout

# PDコントローラ
def pd_control(desired_angle, current_angle, current_velocity, KP, KD):
        error = desired_angle - current_angle                       #誤差＝目標角度ー現在角度
        control_input = KP @ error - KD @ current_velocity          #制御入力(トルク)＝PD制御
        return control_input

# 動力学シミュレーション(17)
def dynamics(q, q_dot, torque):                     #関節角度、関節各速度、トルク
        inertia_matrix = np.diag([0.01, 0.01])          # 簡略化した慣性行列H(1*2)
        q_ddot = np.linalg.inv(inertia_matrix) @ torque #角加速度=慣性行列の逆行列＊トルク
        return q_ddot

def arm_dynamics(q, q_dot, torque, link_lengths, link_masses, damping_factors):
    """
    二リンクロボットアームの動力学を計算する。

    Parameters:
        q (array): 関節角度 [theta1, theta2]
        q_dot (array): 関節角速度 [theta1dot, theta2dot]
        torque (array): 各リンクに加わるトルク [torque1, torque2]
        link_lengths (array): 各リンクの長さ [l1, l2]
        link_masses (array): 各リンクの質量 [m1, m2]
        damping_factors (tuple): 各リンクの粘性摩擦係数 (b1, b2)

    Returns:
        array: 次の状態変数 [theta1dot, theta2dot, theta1ddot, theta2ddot]
    """
    # リンクの長さと質量
    l1, l2 = link_lengths
    m1, m2 = link_masses

    # 粘性摩擦係数
    b1, b2 = damping_factors

    # 入力変数
    theta1, theta2 = q
    theta1dot, theta2dot = q_dot

    # 慣性行列の要素
    M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(theta2)
    M12 = m2 * l2**2 + m2 * l1 * l2 * np.cos(theta2)
    M21 = M12  # 対称性を利用
    M22 = m2 * l2**2
    M = np.array([[M11, M12], [M21, M22]])

    # コリオリ力・遠心力ベクトル
    V1 = -m2 * l1 * l2 * (2 * theta1dot * theta2dot + theta2dot**2) * np.sin(theta2)
    V2 = m2 * l1 * l2 * theta1dot**2 * np.sin(theta2)
    V = np.array([V1, V2])

    # 粘性摩擦ベクトル
    F = np.array([b1 * theta1dot, b2 * theta2dot])

    # 角加速度を計算（M の逆行列が安全に計算できるよう pinv を使用）
    thetaddot = pinv(M) @ (np.array(torque) - F - V)

    # 次の状態を計算
    output = np.array([
        theta1dot,           # theta1 の角速度
        theta2dot,           # theta2 の角速度
        thetaddot[0],        # theta1 の角加速度
        thetaddot[1]         # theta2 の角加速度
    ])

    return output

def calculate_dynamics(q, q_dot, q_ddot, link_lengths, link_masses):
    """
    2リンクアームの動力学を計算。
    q: 現在の関節角度（配列）
    q_dot: 現在の関節角速度（配列）
    q_ddot: 現在の関節角加速度（配列）
    link_lengths: 各リンクの長さ（配列）
    link_masses: 各リンクの質量（配列）
    """
    m1, m2 = link_masses
    l1, l2 = link_lengths
    g = 9.81  # 重力加速度

    # 質量行列 M(q)
    M = np.array([
        [m1 * (l1 / 2)**2 + m2 * (l1**2 + (l2 / 2)**2) + 2 * m2 * l1 * (l2 / 2) * np.cos(q[1]), 
         m2 * ((l2 / 2)**2 + l1 * (l2 / 2) * np.cos(q[1]))],
        [m2 * ((l2 / 2)**2 + l1 * (l2 / 2) * np.cos(q[1])), 
         m2 * (l2 / 2)**2]
    ])

    # コリオリ力・遠心力行列 C(q, q_dot)
    C = np.array([
        [-m2 * l1 * (l2 / 2) * np.sin(q[1]) * q_dot[1], -m2 * l1 * (l2 / 2) * np.sin(q[1]) * (q_dot[0] + q_dot[1])],
        [m2 * l1 * (l2 / 2) * np.sin(q[1]) * q_dot[0], 0]
    ])

    # 重力項 G(q)
    G = np.array([
        m1 * g * (l1 / 2) * np.cos(q[0]) + m2 * g * (l1 * np.cos(q[0]) + (l2 / 2) * np.cos(q[0] + q[1])),
        m2 * g * (l2 / 2) * np.cos(q[0] + q[1])
    ])

    return M, C, G

def pd_control_with_dynamics(desired_angle, current_angle, current_velocity, KP, KD, link_lengths, link_masses):
    """
    動力学を考慮したPD制御によるトルク計算
    """
    error = desired_angle - current_angle
    control_input = KP @ error - KD @ current_velocity  # 基本PD制御

    # 動力学を計算
    M, C, G = calculate_dynamics(current_angle, current_velocity, np.zeros(2), link_lengths, link_masses)

    # 必要トルクを計算
    torque = M @ control_input + C @ current_velocity + G
    return torque

# ティーチング軌道生成
def generate_teaching_trajectories(initial_angles_set, target_angles, steps, KP, KD, dt):   #初期関節角度、目標角度、ステップ数
        trajectories = []                                                           #各初期点からの軌道を格納するリスト
        torques = []                                                                # 各時刻でのトルクを格納するリスト
        for initial_angles in initial_angles_set:                                   #初期角度の数だけ軌道作成
            q = initial_angles                                                      #初期角度を関節角度qに入れる(初期化)
            q_dot = np.zeros(2)                                                     #関節角速度を0に初期化
            trajectory = [q]                                                        #xy座標として軌道を格納
            torque_seq = []                                                         # 各軌道のトルク時系列を保存
            for _ in range(steps):
                torque = pd_control(target_angles, q, q_dot, KP, KD)                #pd制御での制御入力をトルクとする
                q_ddot = dynamics(q, q_dot, torque)                                 #角加速度を計算
                q = q + q_dot * dt                                                  #オイラー法で角度と角速度を更新
                q_dot = q_dot + q_ddot * dt
                trajectory.append(q)                            #計算した角度で軌道を求める
                torque_seq.append(torque)
            trajectories.append(np.array(trajectory))                               #各初期点からの軌道をリストに格納
            torques.append(np.array(torque_seq))
        return trajectories, torques

# ティーチング軌道生成
def generate_teaching_trajectories_move(initial_angles_set, start_angle, end_angle, steps, steps_per_second, move_duration, hold_duration, KP, KD, dt):   
        """
        初期関節角度、目標角度の開始と終了、ステップ数に基づいてティーチング軌道を生成する。
        """
        trajectories = []  # 各初期点からの軌道を格納するリスト
        torques = []       # 各時刻でのトルクを格納するリスト
        transition_steps = steps  # 1秒間かけて目標角度を変化させる

        for initial_angles in initial_angles_set:  # 初期角度の数だけ軌道作成
            q = initial_angles  # 初期角度を関節角度qに入れる(初期化)
            q_dot = np.zeros(2)  # 関節角速度を0に初期化
            trajectory = [q]  # 軌道を格納
            torque_seq = []   # 各軌道のトルク時系列を保存

            for step in range(steps):
                # 時間に応じて目標角度を更新
                target_angle = dynamic_target_angle_sin2(step, steps_per_second, move_duration, hold_duration, start_angle, end_angle)
                
                # pd制御での制御入力をトルクとする
                torque = pd_control(target_angle, q, q_dot, KP, KD)
                # 角加速度を計算
                q_ddot = dynamics(q, q_dot, torque)
                # オイラー法で角度と角速度を更新
                q = q + q_dot * dt
                q_dot = q_dot + q_ddot * dt

                # 計算した角度とトルクを記録
                trajectory.append(q)
                torque_seq.append(torque)

            # 軌道とトルクをリストに格納
            trajectories.append(np.array(trajectory))
            torques.append(np.array(torque_seq))

        return trajectories, torques


# ティーチング軌道生成
def generate_teaching_trajectories_circular(initial_angles_set, steps, steps_per_second, center_angle, radius, total_revolutions, KP, KD, dt):
    """
    円軌道を目標とするティーチング軌道を生成する。
    """
    trajectories = []
    torques = []

    for initial_angles in initial_angles_set:
        q = initial_angles  # 初期角度
        q_dot = np.zeros(2)  # 初期角速度
        trajectory = [q]  # 軌道を保存
        torque_seq = []  # トルクを保存

        for step in range(steps):
            # 円軌道に基づく目標角度を計算
            target_angle = dynamic_target_angle_circular(step, steps, total_revolutions, center_angle, radius)  # 修正済み

            # PD制御を使用してトルクを計算
            torque = pd_control(target_angle, q, q_dot, KP, KD)
            q_ddot = dynamics(q, q_dot, torque)  # 動力学モデルで角加速度を計算
            q = q + q_dot * dt  # オイラー法で角度を更新
            q_dot = q_dot + q_ddot * dt  # オイラー法で角速度を更新

            # 軌道とトルクを記録
            trajectory.append(q)
            torque_seq.append(torque)

        trajectories.append(np.array(trajectory))
        torques.append(np.array(torque_seq))

    return trajectories, torques

# トルクと角度の計算    
def calculate_torque_and_angle(current_angle, target_angle, q_dot, KP, KD, dt):  
    torque = pd_control(target_angle,current_angle, q_dot, KP, KD)               
    q_ddot = dynamics(current_angle, q_dot, torque)                                
    current_angle = current_angle + q_dot * dt                                                  
    q_dot = q_dot + q_ddot * dt
    return current_angle, q_dot, torque, q_ddot  

# 目標角度を動的に生成する関数
def dynamic_target_angle(step, total_steps, start_angle, end_angle):
    """ 
    目標角度を時間とともに動的に変化させる。
    step: 現在のステップ
    total_steps: 目標角度の移動が完了するまでのステップ数
    start_angle: 初期角度（開始時の目標角度）
    end_angle: 最終角度（移動完了時の目標角度）
    """
    if step <= total_steps:
        # 線形補間による目標角度の変化
        return start_angle + (end_angle - start_angle) * (step / total_steps)
    else:
        # 目標角度移動完了後は固定
        return end_angle 

# 目標角度を動的に生成する関数
def dynamic_target_angle_sin(step, total_steps, start_angle, end_angle):
    """ 
    目標角度を時間とともに曲線的に変化させる。
    step: 現在のステップ
    total_steps: 目標角度の移動が完了するまでのステップ数
    start_angle: 初期角度（開始時の目標角度）
    end_angle: 最終角度（移動完了時の目標角度）
    """
    if step <= total_steps:
        # 正規化された時間
        t = step / total_steps

        # 一方の軸（例えば x 軸）を線形的に移動
        linear_component = start_angle[0] + (end_angle[0] - start_angle[0]) * t
        
        # もう一方の軸（例えば y 軸）をsin的に変化させ、0から1まで滑らかに変化
        sin_component = start_angle[1] + (end_angle[1] - start_angle[1]) * np.sin(np.pi * t / 2)

        return np.array([linear_component, sin_component])
    else:
        # 目標角度移動完了後は固定
        return end_angle
    
def dynamic_target_angle_sin2(step, steps_per_second, move_duration, hold_duration, start_angle, end_angle):
    """ 
    目標角度を動的に生成し、一定期間固定する。
    
    step: 現在のステップ
    steps_per_second: 1秒間のステップ数
    move_duration: 目標角度を動かす時間（秒）
    hold_duration: 終了値で固定する時間（秒）
    start_angle: 初期角度（開始時の目標角度）
    end_angle: 最終角度（移動完了時の目標角度）
    """
    move_steps = int(steps_per_second * move_duration)  # 動的目標値生成に要するステップ数
    total_steps = move_steps + int(steps_per_second * hold_duration)  # 全体のステップ数

    if step <= move_steps:  # 目標角度を動かすフェーズ
        # 正規化された時間
        t = step / move_steps

        # x軸（線形変化）とy軸（sin変化）の目標角度を計算
        linear_component = start_angle[0] + (end_angle[0] - start_angle[0]) * t
        sin_component = start_angle[1] + (end_angle[1] - start_angle[1]) * np.sin(np.pi * t / 2)

        return np.array([linear_component, sin_component])
    elif step <= total_steps:  # 終了値で固定するフェーズ
        return end_angle
    else:  # 全体のステップ数を超えた場合（安全策として終了値を返す）
        return end_angle

# 動的目標角度生成関数（時間ベース）
def dynamic_target_angle_sin_time(current_time, move_duration, hold_duration, start_angle, end_angle):
    """
    目標角度を動的に生成し、一定期間固定する。
    """
    if current_time <= move_duration:
        t = current_time / move_duration
        linear_component = start_angle[0] + (end_angle[0] - start_angle[0]) * t
        sin_component = start_angle[1] + (end_angle[1] - start_angle[1]) * np.sin(np.pi * t / 2)
        return np.array([linear_component, sin_component])
    elif current_time <= move_duration + hold_duration:
        return end_angle
    else:
        return end_angle

# 円軌道を描くように目標角度を動的に生成
def dynamic_target_angle_circular(step, steps, total_revolutions, center_angle, radius):
    """
    ステップ数全体を基に円軌道を計算し、複数周回を実現する。
    """
    angle_progress = (2 * np.pi * total_revolutions * step / steps) % (2 * np.pi)
    x = center_angle[0] + radius * np.cos(angle_progress)
    y = center_angle[1] + radius * np.sin(angle_progress)
    return np.array([x, y])

def dynamic_target_angle_circular_time(elapsed_time, total_duration, total_revolutions, center_angle, radius):
    """
    時間ベースで円軌道を計算し、指定された時間内で複数周回を実現。
    """
    if elapsed_time <= 0:
        # 初期値として円の右端を指定
        return np.array([center_angle[0] + radius, center_angle[1]])

    # 経過時間に基づく進行角度
    angle_progress = (2 * np.pi * total_revolutions * elapsed_time / total_duration) % (2 * np.pi)
    
    # 円軌道の計算
    x = center_angle[0] + radius * np.cos(angle_progress)
    y = center_angle[1] + radius * np.sin(angle_progress)
    
    return np.array([x, y])

def animate_esn_trajectory(esn_trajectory, link_lengths, target_positions_over_time, steps):
    """
    ESNの生成軌道のアニメーションを表示および保存する。
    
    Parameters:
        esn_trajectory: ndarray
            ESNで生成された一つの関節角度軌道。
        link_lengths: list
            各リンクの長さ。
        target_positions_over_time: ndarray
            目標位置の時間変化。
        steps: int
            シミュレーションステップ数。
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 固定目標か動的目標かを判定
    if np.all(target_positions_over_time == target_positions_over_time[0]):
        # 固定目標の場合
        fixed_target = target_positions_over_time[0]
        ax.plot(fixed_target[0], fixed_target[1], 'ro', label="Fixed Target", markersize=10)
    else:
        # 動的目標の場合
        ax.plot(target_positions_over_time[:, 0], target_positions_over_time[:, 1], 'k--', label="Target Trajectory")
    
    # 初期ラインとテキストの設定
    line, = ax.plot([0, 0, 0], [0, 0, 0], linewidth=10, label="Robot Arm")
    time_text = ax.text(0.5, -1.0, f'Time: {0:.2f} s', fontsize=15, ha='right')

    # 動的な範囲設定
    all_x = np.hstack((target_positions_over_time[:, 0], esn_trajectory[:, 0]))
    all_y = np.hstack((target_positions_over_time[:, 1], esn_trajectory[:, 1]))
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
        
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend(loc="upper right")

    # アームの先端位置を計算
    def forward_kinematics(angles):
        x1 = link_lengths[0] * np.cos(angles[0])
        y1 = link_lengths[0] * np.sin(angles[0])
        x2 = x1 + link_lengths[1] * np.cos(angles[0] + angles[1])
        y2 = y1 + link_lengths[1] * np.sin(angles[0] + angles[1])
        return np.array([[0, x1, x2], [0, y1, y2]])

    # アニメーション更新関数
    def update(frame):
        i = frame
        fk = forward_kinematics(esn_trajectory[i])
        line.set_xdata(fk[0])
        line.set_ydata(fk[1])
        time_text.set_text(f'Time: {i * 0.01:.2f} s')
        return line, time_text

    # アニメーション設定
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=20, blit=False)

    # アニメーション表示
    plt.show()

# ReLU関数の定義
def relu(x):
    return np.maximum(0, x)

# エコーステートネットワーク
class ESN:
    # 各層の初期化
    def __init__(self, N_u, N_y, N_x, density=0.05, input_scale=1.0,
                 rho=0.95, activation_func=np.tanh, fb_scale = None,
                 fb_seed=0, noise_level = None, leaking_rate=1.0,
                 output_func=identity, inv_output_func=identity,
                 classification = False, average_window = None):
        '''
        param N_u: 入力次元
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param density: リザバーのネットワーク結合密度
        param input_scale: 入力スケーリング
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: リザバーノードの活性化関数
        param fb_scale: フィードバックスケーリング（default: None）
        param fb_seed: フィードバック結合重み行列生成に使う乱数の種
        param leaking_rate: leaky integratorモデルのリーク率
        param output_func: 出力層の非線形関数（default: 恒等写像）
        param inv_output_func: output_funcの逆関数
        param classification: 分類問題の場合はTrue（default: False）
        param average_window: 分類問題で出力平均する窓幅（default: None）
        '''
        self.Input = Input(N_u, N_x, input_scale)
        self.Reservoir = Reservoir(N_x, density, rho, activation_func, 
                                   leaking_rate)
        self.Output = Output(N_x, N_y)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.y_prev = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func
        self.classification = classification

        # 出力層からのリザバーへのフィードバックの有無
        if fb_scale is None:
            self.Feedback = None
        else:
            self.Feedback = Feedback(N_y, N_x, fb_scale, fb_seed)

        # リザバーの状態更新おけるノイズの有無
        if noise_level is None:
            self.noise = None
        else:
            np.random.seed(seed=0)
            self.noise = np.random.uniform(-noise_level, noise_level, 
                                           (self.N_x, 1))

        # 分類問題か否か
        if classification:
            if average_window is None:
                raise ValueError('Window for time average is not given!')
            else:
                self.window = np.zeros((average_window, N_x))

    def reset_states(self):
        self.Reservoir.x = np.zeros(self.N_x)
        
    # バッチ学習
    def train(self, U, D, optimizer, trans_len = None):
        '''
        U: 教師データの入力, データ長×N_u
        D: 教師データの出力, データ長×N_y
        optimizer: 学習器
        trans_len: 過渡期の長さ
        return: 学習前のモデル出力, データ長×N_y
        '''
        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []

        # 時間発展
        for n in range(train_len):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # ノイズ
            if self.noise is not None:
                x_in += self.noise

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1),
                                        axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            # 目標値
            d = D[n]
            d = self.inv_output_func(d)

            # 学習器
            if n > trans_len:  # 過渡期を過ぎたら
                optimizer(d, x)

            # 学習前のモデル出力
            y = self.Output(x)
            Y.append(self.output_func(y))
            self.y_prev = d

        # 学習済みの出力結合重み行列を設定
        self.Output.setweight(optimizer.get_Wout_opt())

        # モデル出力（学習前）
        return np.array(Y)

    # バッチ学習後の予測
    def predict(self, U):
        test_len = len(U)
        Y_pred = []

        # 時間発展
        for n in range(test_len):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1),
                                        axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            # 学習後のモデル出力
            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            self.y_prev = y_pred

        # モデル出力（学習後）
        return np.array(Y_pred)
    
    # バッチ学習後の軌道予測位置制御
    def trajectory_angle(self, U0, steps, KP, KD, dt): 
        Y_pred = []  # 予測した角度
        torque_set = [np.zeros(2)]  # トルクは計算しないのでゼロで初期化
        current_angle_set = [U0]
        error_set = [np.zeros(2)]

        max_step_angle = 0.05  # 1ステップで動ける角度の最大値

        # 時間発展
        for n in range(steps):
            x_in = self.Input(current_angle_set[n])
            x = self.Reservoir(x_in)
            y_pred = self.Output(x)

            # 1ステップで動ける角度の制限を適用
            y_pred = np.clip(y_pred, current_angle_set[n] - max_step_angle, current_angle_set[n] + max_step_angle)
            
            Y_pred.append(y_pred)


            current_angle_set.append(y_pred)

            # 誤差は予測角度と次の角度の差を計算
            error = y_pred - current_angle_set[n]
            error_set.append(error)

            # トルクはゼロのまま保持
            torque_set.append(np.zeros(2))

        # モデル出力
        return np.array(Y_pred), np.array(torque_set), np.array(current_angle_set), np.array(error_set)

    # バッチ学習後の軌道予測トルク制御
    def trajectory1(self, U0, steps, KP, KD, dt): 
        Y_pred = [U0] #予測した角度
        torque_set =[np.zeros(2)] #予測したトルク
        current_angle_set = [U0]
        q_dot_set = [np.zeros(2)]
        q_ddot_set = [np.zeros(2)]
        error_set = [np.zeros(2)]

        # 時間発展
        for n in range(steps):
            
            x_in = self.Input(current_angle_set[n])
            x = self.Reservoir(x_in)
            y_pred = self.Output(x)
            
            Y_pred.append(y_pred) 
            
            # PDコントローラ
            current_angle, q_dot, torque, q_ddot = calculate_torque_and_angle(current_angle_set[n], Y_pred[n+1], q_dot_set[n], KP, KD, dt)
            
            current_angle_set.append(current_angle)
            q_dot_set.append(q_dot)
            q_ddot_set.append(q_ddot)
            torque_set.append(torque)
            
            error = Y_pred[n+1] - current_angle_set[n+1]
            error_set.append(error)
            
        # モデル出力（学習後）
        return np.array(Y_pred) , np.array(torque_set), np.array(current_angle_set), np.array(error_set)

    def trajectory_delay_angle(self, U0, steps, KP, KD, dt, deadtime_steps=1): 
        Y_pred = [U0]  # 予測した角度
        torque_set = [np.zeros(2)]  # トルクは計算しないのでゼロで初期化
        current_angle_set = [U0]
        error_set = [np.zeros(2)]

        # 最初の無駄時間ステップでは角度を固定
        for n in range(deadtime_steps):
            Y_pred.append(U0)  # 角度は変化しない
            torque_set.append(np.zeros(2))  # トルクゼロ
            current_angle_set.append(U0)  # 現在角度も変化しない
            error_set.append(np.zeros(2))  # 誤差もゼロ

        # 無駄時間終了後の軌道生成
        for n in range(deadtime_steps, steps):
            # 無駄時間分遅れた予測を使用
            delayed_index = n - deadtime_steps

            x_in = self.Input(current_angle_set[delayed_index])  # 遅れた現在角度を使用
            x = self.Reservoir(x_in)
            y_pred = self.Output(x)

            # 1ステップで動ける角度の制限を適用
            y_pred = np.clip(y_pred, current_angle_set[n] - 0.5, current_angle_set[n] + 0.5)

            Y_pred.append(y_pred)

            # 次の現在角度は予測した角度をそのまま使用
            current_angle_set.append(y_pred)

            # 誤差は予測角度と次の角度の差を計算
            error = y_pred - current_angle_set[n]
            error_set.append(error)

            # トルクはゼロのまま保持
            torque_set.append(np.zeros(2))
        
        # モデル出力
        return np.array(Y_pred), np.array(torque_set), np.array(current_angle_set), np.array(error_set)

    def trajectory_delay_angle_smith(self, U0, steps, KP, KD, dt, deadtime_steps=0): 
        Y_pred = [U0]  # 予測した角度
        torque_set = [np.zeros(2)]  # トルクは計算しないのでゼロで初期化
        current_angle_set = [U0]
        error_set = [np.zeros(2)]
        ideal_angle_set = [U0]  # 理想的な応答（無駄時間なしの応答）

        # 最初の無駄時間ステップでは角度を固定
        for n in range(deadtime_steps):
            Y_pred.append(U0)  # 角度は変化しない
            torque_set.append(np.zeros(2))  # トルクゼロ
            current_angle_set.append(U0)  # 現在角度も変化しない
            error_set.append(np.zeros(2))  # 誤差もゼロ
            ideal_angle_set.append(U0)  # 理想的な応答も固定

        # 無駄時間終了後の軌道生成
        for n in range(deadtime_steps, steps):
            # 無駄時間分遅れた予測を使用
            delayed_index = n - deadtime_steps

            x_in = self.Input(current_angle_set[delayed_index])  # 遅れた現在角度を使用
            x = self.Reservoir(x_in)
            y_pred = self.Output(x)

            # 理想的な応答（無駄時間なしの応答）を計算
            x_in_ideal = self.Input(ideal_angle_set[n-1])
            x_ideal = self.Reservoir(x_in_ideal)
            ideal_angle = self.Output(x_ideal)
            ideal_angle_set.append(ideal_angle)

            # 1ステップで動ける角度の制限を適用
            y_pred = np.clip(y_pred, current_angle_set[n] - 0.5, current_angle_set[n] + 0.5)

            Y_pred.append(y_pred)

            a=1

            # スミス予見制御による補正
            corrected_angle = ideal_angle + a*(y_pred - ideal_angle_set[delayed_index])

            # 次の現在角度は補正された角度を使用
            current_angle_set.append(corrected_angle)

            # 誤差は補正された角度と現在角度の差を計算
            error = corrected_angle - current_angle_set[n]
            error_set.append(error)

            # トルクはゼロのまま保持
            torque_set.append(np.zeros(2))

        # モデル出力
        return np.array(Y_pred), np.array(torque_set), np.array(current_angle_set), np.array(error_set)

    def trajectory_delay_torque(self, U0, steps, KP, KD, dt, deadtime_steps=5): 
        Y_pred = [U0]  # 予測した角度
        torque_set = [np.zeros(2)]  # 予測したトルク
        current_angle_set = [U0]
        q_dot_set = [np.zeros(2)]
        q_ddot_set = [np.zeros(2)]
        error_set = [np.zeros(2)]

        # 最初の無駄時間ステップでは角度を固定し、トルクはゼロ
        for n in range(deadtime_steps):
            Y_pred.append(U0)  # 角度は変化しない
            torque_set.append(np.zeros(2))  # トルクゼロ
            current_angle_set.append(U0)  # 現在角度も変化しない
            q_dot_set.append(np.zeros(2))  # 速度もゼロ
            q_ddot_set.append(np.zeros(2))  # 加速度もゼロ
            error_set.append(np.zeros(2))  # 誤差もゼロ

        # 無駄時間終了後の軌道生成
        for n in range(deadtime_steps, steps):
            # 無駄時間分遅れた予測を使用
            delayed_index = n - deadtime_steps

            x_in = self.Input(current_angle_set[delayed_index])  # 遅れた現在角度を使用
            x = self.Reservoir(x_in)
            y_pred = self.Output(x)

            Y_pred.append(y_pred)

            # PDコントローラ
            current_angle, q_dot, torque, q_ddot = calculate_torque_and_angle(
                current_angle_set[n], Y_pred[n+1], q_dot_set[n], KP, KD, dt
            )

            current_angle_set.append(current_angle)
            q_dot_set.append(q_dot)
            q_ddot_set.append(q_ddot)
            torque_set.append(torque)

            error = Y_pred[n+1] - current_angle_set[n+1]
            error_set.append(error)

        # モデル出力（学習後）
        return np.array(Y_pred), np.array(torque_set), np.array(current_angle_set), np.array(error_set)

    # バッチ学習後の予測（自律系のフリーラン）
    def run(self, U):
        test_len = len(U)
        Y_pred = []
        y = U[0]

        # 時間発展
        for n in range(test_len):
            x_in = self.Input(y)

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 学習後のモデル出力
            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            y = y_pred
            self.y_prev = y

        return np.array(Y_pred)

    # オンライン学習と予測
    def adapt(self, U, D, optimizer):
        data_len = len(U)
        Y_pred = []
        Wout_abs_mean = []

        # 出力結合重み更新
        for n in np.arange(0, data_len, 1):
            x_in = self.Input(U[n])
            x = self.Reservoir(x_in)
            d = D[n]
            d = self.inv_output_func(d)
            
            # 学習
            Wout = optimizer(d, x)

            # モデル出力
            y = np.dot(Wout, x)
            Y_pred.append(y)
            Wout_abs_mean.append(np.mean(np.abs(Wout)))

        return np.array(Y_pred), np.array(Wout_abs_mean)
