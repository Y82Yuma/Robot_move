# 🔍 デスクトップ vs ラズパイ環境の違い分析

## 🖥️ デスクトップ (Windows)
- **OS**: Windows 11
- **Python**: 3.12.4 (uv管理)
- **アーキテクチャ**: x86_64
- **メモリ**: 充分 (8GB+)
- **パッケージ**: プリコンパイル済みwheel利用可能

## 🍓 ラズパイ (推測される問題)
- **OS**: Raspberry Pi OS (Debian系)
- **Python**: システムPython 3.9-3.11
- **アーキテクチャ**: ARM64/aarch64
- **メモリ**: 限られた容量 (4-8GB)
- **パッケージ**: ARM64版が無い場合はソースビルド

## ❌ よくある失敗パターン

### 1. **Pythonバージョンの互換性問題**
```toml
# あなたの設定 (厳しすぎる可能性)
requires-python = ">=3.12.4"

# 推奨設定
requires-python = ">=3.9"
```

### 2. **ARM64パッケージの不足**
```bash
# よくあるエラー
ERROR: Could not find a version that satisfies the requirement XXX
ERROR: No matching distribution found for XXX (from versions: none)
```

### 3. **メモリ不足**
```bash
# uvの依存関係解決でメモリ不足
Killed (out of memory)
```

### 4. **システム依存関係の不足**
```bash
# コンパイル時のエラー
fatal error: 'Python.h' file not found
ERROR: Microsoft Visual C++ 14.0 is required
```

## ✅ 成功している人の環境 (推測)

### パターンA: システムパッケージ利用
```bash
# ラズパイOSの標準パッケージを使用
sudo apt install python3-numpy python3-scipy python3-sklearn
sudo apt install python3-matplotlib python3-pandas
pip3 install --user joblib tomli pyserial
```

### パターンB: 軽量conda環境
```bash
# miniforge (ARM64対応)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
conda create -n robot python=3.11
conda activate robot
conda install numpy scipy scikit-learn matplotlib pandas joblib
```

### パターンC: Docker利用
```bash
# 完全に隔離された環境
docker run --privileged -v $(pwd):/app python:3.11-slim
```

## 🎯 あなたが試すべき順序

### 1. **Pythonバージョン要件の緩和**
```toml
requires-python = ">=3.9"  # 3.12.4から変更
```

### 2. **システムパッケージ優先のハイブリッド方式**
```bash
# システムパッケージをまずインストール
sudo apt install python3-dev python3-numpy python3-scipy
# 足りないものだけpipで追加
uv add --system-site-packages <missing-packages>
```

### 3. **段階的インストール**
```bash
# 一つずつ確認しながらインストール
uv add numpy  # 成功?
uv add scipy  # 成功?
uv add scikit-learn  # 成功?
```