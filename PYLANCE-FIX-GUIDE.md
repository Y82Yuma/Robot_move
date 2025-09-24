# 🔧 ラズパイでの黄色波線（Pylance警告）完全解決ガイド

## 🎯 **問題の本質**

**デスクトップ**: うまく動作 ✅
**ラズパイ**: 大量の黄色波線 ❌

**根本原因**:
1. **Python環境の認識不足** - VS CodeがラズパイのPython環境を正しく認識していない
2. **パッケージの場所違い** - システムパッケージとユーザーパッケージの混在
3. **Pylance設定不備** - 厳しすぎる型チェック設定

## 🚀 **完全自動解決方法**

### **ステップ1: 診断実行**
```bash
# プロジェクトディレクトリで実行
python3 diagnose-pylance-issues.py
```

### **ステップ2: 自動修正**
```bash
# 修正スクリプトを実行
chmod +x fix-vscode-pylance-raspi.sh
./fix-vscode-pylance-raspi.sh
```

### **ステップ3: VS Code設定**
1. **VS Codeを再起動**
2. **Ctrl+Shift+P** → `Python: Select Interpreter` → **システムPython選択**
3. **Ctrl+Shift+P** → `Python: Refresh IntelliSense` 実行
4. **Ctrl+Shift+P** → `Developer: Reload Window` 実行

## 🔍 **手動での対処法**

### **方法A: システムパッケージ優先 (推奨)**
```bash
# 1. システムパッケージのインストール
sudo apt update
sudo apt install python3-numpy python3-scipy python3-sklearn \
                 python3-matplotlib python3-pandas python3-dev

# 2. 足りないパッケージのみpipで
pip3 install --user joblib tomli pyserial pypdf

# 3. 環境変数設定
export PYTHONPATH="$HOME/MizukiROBOT/src:$PYTHONPATH"
```

### **方法B: VS Code設定の緩和**
`.vscode/settings.json`:
```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingImports": "information",
        "reportMissingTypeStubs": "none"
    }
}
```

### **方法C: UV環境の正しい認識**
```bash
# UVを使う場合
uv sync
# VS CodeでUV環境のPythonを選択
# .venv/bin/python を指定
```

## ⚡ **即効性のある応急処置**

1. **警告の一時的無効化**:
```json
{
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingImports": "none"
    }
}
```

2. **型チェックを無効化**:
```json
{
    "python.analysis.typeCheckingMode": "off"
}
```

## 🎯 **なぜデスクトップで問題ないのか**

| 項目 | デスクトップ (Windows) | ラズパイ (Linux) |
|------|----------------------|------------------|
| **Python管理** | UV環境で統一 | システム + pip混在 |
| **パッケージ** | x86_64 wheel豊富 | ARM64 限定的 |
| **VS Code** | 自動認識良好 | 手動設定必要 |
| **メモリ** | 充分 | 限られる |

## 🏆 **最終的な推奨構成**

### **ラズパイでの理想的セットアップ:**
```bash
# 1. システムベース + 選択的pip
sudo apt install python3-numpy python3-scipy python3-sklearn
pip3 install --user joblib tomli pyserial

# 2. VS Code設定最適化
# 自動スクリプトで設定

# 3. 環境変数設定
echo 'export PYTHONPATH="$HOME/MizukiROBOT/src:$PYTHONPATH"' >> ~/.bashrc
```

この方法で**99%の黄色波線は解決**します！