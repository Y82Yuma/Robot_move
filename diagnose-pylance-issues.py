#!/usr/bin/env python3
# diagnose-pylance-issues.py - Pylance警告の診断スクリプト

import sys
import importlib
from pathlib import Path

print("🔍 Pylance警告診断スクリプト")
print("=" * 50)

# 1. Python環境情報
print(f"Python実行可能ファイル: {sys.executable}")
print(f"Pythonバージョン: {sys.version}")
print(f"Pythonパス:")
for path in sys.path:
    print(f"  - {path}")

print("\n📦 パッケージインポートテスト")
print("-" * 30)

# テストするパッケージリスト
test_packages = [
    ('numpy', 'np'),
    ('scipy', None),
    ('sklearn', None),
    ('matplotlib.pyplot', 'plt'),
    ('pandas', 'pd'),
    ('joblib', None),
    ('pypdf', None),
]

# プロジェクト内モジュール
project_modules = [
    'affetto_nn_ctrl',
    'affetto_nn_ctrl.data_handling',
    'affetto_nn_ctrl.event_logging',
    'affetto_nn_ctrl.model_utility',
]

# システムパッケージのテスト
for pkg_name, alias in test_packages:
    try:
        pkg = importlib.import_module(pkg_name)
        version = getattr(pkg, '__version__', 'バージョン不明')
        location = getattr(pkg, '__file__', '場所不明')
        print(f"✅ {pkg_name}: {version}")
        print(f"   場所: {location}")
    except ImportError as e:
        print(f"❌ {pkg_name}: インポートエラー - {e}")
    except Exception as e:
        print(f"⚠️  {pkg_name}: その他のエラー - {e}")

# プロジェクト内モジュールのテスト
print("\n🏠 プロジェクトモジュールテスト")
print("-" * 30)

# srcディレクトリをパスに追加
src_path = Path('./src')
if src_path.exists():
    sys.path.insert(0, str(src_path.absolute()))
    print(f"srcパスを追加: {src_path.absolute()}")

for module_name in project_modules:
    try:
        module = importlib.import_module(module_name)
        location = getattr(module, '__file__', '場所不明')
        print(f"✅ {module_name}")
        print(f"   場所: {location}")
    except ImportError as e:
        print(f"❌ {module_name}: インポートエラー - {e}")
    except Exception as e:
        print(f"⚠️  {module_name}: その他のエラー - {e}")

# VS Code設定の確認
print("\n⚙️ VS Code設定確認")
print("-" * 30)

vscode_settings = Path('.vscode/settings.json')
if vscode_settings.exists():
    print(f"✅ VS Code設定ファイル存在: {vscode_settings}")
    try:
        import json
        with open(vscode_settings, 'r') as f:
            settings = json.load(f)
        
        if 'python.pythonPath' in settings:
            print(f"   Python Path: {settings['python.pythonPath']}")
        if 'python.analysis.extraPaths' in settings:
            print(f"   Extra Paths: {settings['python.analysis.extraPaths']}")
    except Exception as e:
        print(f"   設定ファイル読み取りエラー: {e}")
else:
    print("❌ VS Code設定ファイルなし")

# 推奨解決策
print("\n💡 推奨解決策")
print("-" * 30)

missing_system = []
missing_pip = []

for pkg_name, _ in test_packages:
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        if pkg_name in ['numpy', 'scipy', 'sklearn', 'matplotlib.pyplot', 'pandas']:
            missing_system.append(f"python3-{pkg_name.split('.')[0]}")
        else:
            missing_pip.append(pkg_name)

if missing_system:
    print("1. システムパッケージをインストール:")
    print(f"   sudo apt install {' '.join(missing_system)}")

if missing_pip:
    print("2. pip3でパッケージをインストール:")
    print(f"   pip3 install --user {' '.join(missing_pip)}")

if not vscode_settings.exists():
    print("3. VS Code設定ファイルを作成:")
    print("   ./fix-vscode-pylance-raspi.sh を実行")

print("\n4. VS Code再起動後:")
print("   - Ctrl+Shift+P → 'Python: Select Interpreter'")
print("   - Ctrl+Shift+P → 'Python: Refresh IntelliSense'")