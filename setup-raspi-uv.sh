#!/bin/bash
# setup-raspi-uv.sh - ラズパイでのuv環境セットアップスクリプト

set -e

echo "🍓 Raspberry Pi用 UV環境セットアップ開始..."

# システムパッケージの更新
echo "📦 システムパッケージを更新中..."
sudo apt update && sudo apt upgrade -y

# 必要なシステムライブラリをインストール
echo "🔧 システム依存関係をインストール中..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    pkg-config \
    libfreetype6-dev \
    libpng-dev

# uvのインストール（ARM64対応）
echo "⚡ uvをインストール中..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# プロジェクトディレクトリに移動（実際のパスに変更してください）
cd /path/to/MizukiROBOT

# ラズパイ用設定ファイルを使用
echo "🔄 ラズパイ用設定でuvを初期化中..."
if [ -f "pyproject-raspi.toml" ]; then
    cp pyproject-raspi.toml pyproject.toml
fi

# Python環境の設定
echo "🐍 Python環境をセットアップ中..."
uv python install 3.11

# 依存関係のインストール（段階的に実行）
echo "📚 基本依存関係をインストール中..."
# まず基本的なパッケージのみをインストール
uv add numpy scipy
uv add scikit-learn joblib
uv add matplotlib pandas

# ラズパイ特有のパッケージを追加
echo "🤖 ラズパイ用パッケージをインストール中..."
if [ "$(uname -m)" = "aarch64" ]; then
    uv add RPi.GPIO gpiozero pyserial
    # 必要に応じてカメラパッケージも追加
    # uv add picamera2
fi

echo "✅ セットアップ完了！"
echo ""
echo "使用方法:"
echo "  uv run python apps/your_script.py"
echo ""
echo "環境の有効化:"
echo "  source .venv/bin/activate"