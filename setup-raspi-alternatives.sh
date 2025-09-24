#!/bin/bash
# setup-raspi-alternatives.sh - ラズパイでの代替環境セットアップ

set -e

echo "🍓 ラズパイ用代替環境セットアップ"
echo "選択してください:"
echo "1) venv + pip (推奨・最も安定)"
echo "2) conda (重いが互換性が高い)"
echo "3) Docker (完全に隔離された環境)"
read -p "選択 (1-3): " choice

case $choice in
    1)
        echo "📦 venv + pip セットアップ..."
        
        # システム依存関係のインストール
        sudo apt update
        sudo apt install -y python3-dev python3-pip python3-venv \
            build-essential cmake git libatlas-base-dev libopenblas-dev \
            liblapack-dev gfortran libhdf5-dev pkg-config
        
        # 仮想環境の作成
        python3 -m venv .venv-raspi
        source .venv-raspi/bin/activate
        
        # pipのアップグレード
        pip install --upgrade pip setuptools wheel
        
        # 基本依存関係のインストール
        pip install numpy scipy scikit-learn joblib pandas matplotlib
        
        # ラズパイ用パッケージ
        if [ "$(uname -m)" = "aarch64" ]; then
            pip install RPi.GPIO gpiozero pyserial
        fi
        
        echo "✅ venv環境完成！"
        echo "有効化: source .venv-raspi/bin/activate"
        ;;
        
    2)
        echo "🐍 conda セットアップ..."
        
        # miniforge (ARM64対応conda) のインストール
        if ! command -v conda &> /dev/null; then
            wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
            bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
            export PATH="$HOME/miniforge3/bin:$PATH"
            echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.bashrc
        fi
        
        # 環境の作成
        conda create -n mizukirobot python=3.11 -y
        conda activate mizukirobot
        
        # パッケージのインストール
        conda install numpy scipy scikit-learn joblib pandas matplotlib -y
        pip install RPi.GPIO gpiozero pyserial
        
        echo "✅ conda環境完成！"
        echo "有効化: conda activate mizukirobot"
        ;;
        
    3)
        echo "🐳 Docker セットアップ..."
        
        # Dockerfileの生成
        cat > Dockerfile.raspi << EOF
FROM python:3.11-slim-bullseye

# システム依存関係
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    libatlas-base-dev \\
    libopenblas-dev \\
    liblapack-dev \\
    gfortran \\
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# Python依存関係
COPY requirements-raspi.txt .
RUN pip install --no-cache-dir -r requirements-raspi.txt

# アプリケーションコピー
COPY . .

# 実行
CMD ["python", "apps/your_script.py"]
EOF

        # requirements-raspi.txtの生成
        cat > requirements-raspi.txt << EOF
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0
joblib>=1.2.0
matplotlib>=3.5.0
pandas>=1.4.0
RPi.GPIO>=0.7.1
gpiozero>=1.6.0
pyserial>=3.5
EOF

        echo "✅ Docker設定完成！"
        echo "ビルド: docker build -f Dockerfile.raspi -t mizukirobot ."
        echo "実行: docker run --privileged --device /dev/gpiomem mizukirobot"
        ;;
esac