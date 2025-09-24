# 🍓 Raspberry Pi でのMizukiROBOT実行ガイド

## 🚀 推奨セットアップ方法

### 方法1: venv + pip (最も安定・推奨)

```bash
# プロジェクトをラズパイにコピー
scp -r MizukiROBOT/ pi@raspberrypi.local:~/

# ラズパイにSSH接続
ssh pi@raspberrypi.local

# プロジェクトディレクトリに移動
cd ~/MizukiROBOT

# セットアップスクリプトを実行
chmod +x setup-raspi-alternatives.sh
./setup-raspi-alternatives.sh
# 選択肢で「1」を選択

# 環境を有効化
source .venv-raspi/bin/activate

# スクリプト実行
export PYTHONPATH="./src:$PYTHONPATH"
python apps/your_robot_script.py
```

### 方法2: UV を使用（実験的）

```bash
# セットアップ
chmod +x setup-raspi-uv.sh
./setup-raspi-uv.sh

# プロジェクトの設定
cp pyproject-raspi.toml pyproject.toml
uv sync

# 実行
uv run python apps/your_robot_script.py
```

## 🔧 トラブルシューティング

### 一般的な問題と解決方法

#### 1. NumPy/SciPy のコンパイルエラー
```bash
# システムライブラリの追加インストール
sudo apt install libatlas-base-dev libopenblas-dev
pip install --no-cache-dir numpy scipy
```

#### 2. メモリ不足エラー
```bash
# スワップファイルの増加
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048 に変更
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### 3. GPIO権限エラー
```bash
# ユーザーをgpioグループに追加
sudo usermod -a -G gpio $USER
# 再ログインが必要
```

#### 4. UV が失敗する場合
- venv + pip に切り替える（方法1を使用）
- または requirements-raspi.txt を使用:
```bash
pip install -r requirements-raspi.txt
```

## 📁 ファイル構成

- `pyproject-raspi.toml`: ラズパイ用UV設定
- `requirements-raspi.txt`: ラズパイ用pip依存関係
- `setup-raspi-uv.sh`: UV自動セットアップ
- `setup-raspi-alternatives.sh`: 代替方法セットアップ

## 🎯 実行例

```bash
# 仮想環境の有効化
source .venv-raspi/bin/activate

# Pythonパスの設定
export PYTHONPATH="./src:$PYTHONPATH"

# ロボット制御スクリプトの実行
python apps/collect_data_myrobot.py --config raspi_config.toml
python apps/track_trajectory.py --output ./data/raspi_data/
```

## 💡 パフォーマンス最適化

```bash
# CPUガバナーをperformanceに設定
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# GPUメモリの調整（config.txtで）
# gpu_mem=128
```

## 📋 システム要件

- Raspberry Pi 4/5 (推奨: 8GB RAM)
- Raspberry Pi OS 64bit
- Python 3.11+
- 32GB以上のSDカード (Class 10推奨)