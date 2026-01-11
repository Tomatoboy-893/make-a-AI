# Human vs Object Detection AI

TensorFlow Hubの物体検出モデル（Faster R-CNN）を使用し、画像に写っているのが「人」か「物」かを自動判定するWebアプリケーションです。

## デモ
実行するとGradioによりWebブラウザ上で判定アプリが立ち上がります。

## 必要要件
* Python 3.8以上推奨
* 以下のライブラリ（`requirements.txt`に含まれています）
    * gradio
    * tensorflow
    * tensorflow_hub
    * numpy
    * Pillow

## インストールと実行方法

1. リポジトリをクローンまたはダウンロードします。
2. 必要なライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
アプリを実行します。

Bash

python app.py
ターミナルに表示されるURL（例: http://127.0.0.1:7860）にブラウザでアクセスしてください。
