#!/bin/bash
# raspi-debug.sh - ラズパイ環境の詳細診断スクリプト

echo "🔍 ラズパイ環境診断開始..."

echo "=== システム情報 ==="
uname -a
cat /etc/os-release | grep PRETTY_NAME
free -h
df -h /

echo -e "\n=== Python環境 ==="
which python3
python3 --version
which pip3
pip3 --version

echo -e "\n=== インストール済みシステムパッケージ (Python関連) ==="
dpkg -l | grep python3-numpy || echo "python3-numpy: 未インストール"
dpkg -l | grep python3-scipy || echo "python3-scipy: 未インストール"
dpkg -l | grep python3-sklearn || echo "python3-sklearn: 未インストール"
dpkg -l | grep python3-matplotlib || echo "python3-matplotlib: 未インストール"

echo -e "\n=== Python import テスト ==="
python3 -c "import sys; print(f'Python: {sys.version}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: インポートエラー"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')" 2>/dev/null || echo "SciPy: インポートエラー"
python3 -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "scikit-learn: インポートエラー"

echo -e "\n=== UV環境 (存在する場合) ==="
which uv 2>/dev/null && uv --version || echo "uv: 未インストール"
[ -f "pyproject.toml" ] && echo "pyproject.toml: 存在" || echo "pyproject.toml: 不存在"
[ -d ".venv" ] && echo ".venv: 存在" || echo ".venv: 不存在"

echo -e "\n=== メモリ使用量 ==="
python3 -c "
import psutil
mem = psutil.virtual_memory()
print(f'総メモリ: {mem.total/1024/1024/1024:.1f}GB')
print(f'使用可能: {mem.available/1024/1024/1024:.1f}GB')
print(f'使用率: {mem.percent:.1f}%')
" 2>/dev/null || echo "psutil未インストールのため取得不可"

echo -e "\n=== 推奨解決策 ==="
echo "1. システムパッケージを先にインストール:"
echo "   sudo apt update && sudo apt install python3-dev python3-numpy python3-scipy python3-sklearn python3-matplotlib python3-pandas"
echo ""
echo "2. 足りないパッケージのみpipでインストール:"
echo "   pip3 install --user joblib tomli pyserial"
echo ""
echo "3. Pythonバージョン要件を緩和:"
echo "   pyproject.tomlで requires-python = \">=3.9\" に変更"