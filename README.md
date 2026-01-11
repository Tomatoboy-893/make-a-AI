# Object Detection App with Gradio & TensorFlow

TensorFlow Hubの学習済みモデル（Faster R-CNN）を使用して、アップロードされた画像に「人」や「物」が写っているかを判定する簡易AIアプリケーションです。
Gradioを使用してWebインターフェースを構築しています。

## 概要
- **モデル**: Faster R-CNN (ResNet50) from TensorFlow Hub
- **機能**: 画像内の物体検出を行い、設定した閾値に基づいて「人」「物」「不明」を判定
- **UI**: GradioによるWebブラウザ操作

## インストール方法 (ローカル環境)

Google Colabで実行する場合は、リポジトリ内の `.ipynb` ファイルを開き、上部の「Open in Colab」ボタンを押してください。

ローカルで実行する場合は以下の手順です：

1. リポジトリをクローン
```bash
git clone [あなたのリポジトリURL]
cd [リポジトリ名]
