#!/bin/bash
# fix-vscode-pylance-raspi.sh - ラズパイでのVS Code + Pylance 警告修正スクリプト

echo "🔧 ラズパイでのVS Code Pylance警告を修正..."

# 1. システムPythonパッケージの確認とインストール
echo "📦 システムPythonパッケージの確認..."
MISSING_PACKAGES=()

# 基本パッケージの確認
for pkg in python3-numpy python3-scipy python3-sklearn python3-matplotlib python3-pandas; do
    if ! dpkg -l | grep -q "^ii.*${pkg}"; then
        MISSING_PACKAGES+=($pkg)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "不足パッケージをインストール: ${MISSING_PACKAGES[*]}"
    sudo apt update
    sudo apt install -y "${MISSING_PACKAGES[@]}"
fi

# 2. pip3でのパッケージインストール（システムに不足しているもの）
echo "🐍 pip3での追加パッケージインストール..."
pip3 install --user joblib tomli pyserial pypdf

# 3. VS Code設定の作成
echo "⚙️ VS Code設定ファイルの作成..."
mkdir -p .vscode

# Python環境の検出
PYTHON_PATH=$(which python3)
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SYSTEM_PACKAGES="/usr/lib/python3/dist-packages"
USER_PACKAGES="$HOME/.local/lib/python${PYTHON_VERSION}/site-packages"

cat > .vscode/settings.json << EOF
{
    "python.pythonPath": "${PYTHON_PATH}",
    "python.defaultInterpreterPath": "${PYTHON_PATH}",
    "python.languageServer": "Pylance",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoSearchPaths": true,
    "python.analysis.autoImportCompletions": true,
    "python.analysis.extraPaths": [
        "./src",
        "${SYSTEM_PACKAGES}",
        "${USER_PACKAGES}"
    ],
    "python.analysis.include": [
        "./src/**",
        "./apps/**",
        "./desktop/**"
    ],
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingImports": "information",
        "reportMissingTypeStubs": "none",
        "reportUnknownMemberType": "none",
        "reportUnknownParameterType": "none",
        "reportUnknownVariableType": "none",
        "reportMissingModuleSource": "none"
    },
    "python.terminal.activateEnvironment": false,
    "python.linting.enabled": false,
    "files.watcherExclude": {
        "**/.git/objects/**": true,
        "**/node_modules/**": true,
        "**/.venv/**": true,
        "**/__pycache__/**": true
    }
}
EOF

# 4. 環境変数の設定
echo "🌍 環境変数の設定..."
cat >> ~/.bashrc << 'EOF'

# MizukiROBOT用の環境設定
export PYTHONPATH="$HOME/MizukiROBOT/src:$PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"
EOF

# 5. パッケージの確認テスト
echo "🧪 インポートテストを実行..."
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

test_packages = [
    'numpy', 'scipy', 'sklearn', 'matplotlib', 'pandas', 
    'joblib', 'tomli', 'serial', 'pypdf'
]

for pkg in test_packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}: OK")
    except ImportError as e:
        print(f"❌ {pkg}: {e}")

# affetto_nn_ctrlのテスト
sys.path.insert(0, './src')
try:
    from affetto_nn_ctrl import DEFAULT_SEED
    print(f"✅ affetto_nn_ctrl: OK (DEFAULT_SEED={DEFAULT_SEED})")
except ImportError as e:
    print(f"❌ affetto_nn_ctrl: {e}")
EOF

echo ""
echo "✅ VS Code + Pylance設定完了！"
echo ""
echo "次の手順:"
echo "1. VS Codeを再起動してください"
echo "2. Ctrl+Shift+P → 'Python: Select Interpreter' → システムPython選択"
echo "3. Ctrl+Shift+P → 'Python: Refresh IntelliSense' を実行"
echo "4. 必要に応じて 'Developer: Reload Window' を実行"