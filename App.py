import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# --- グローバル変数としてモデルを定義 ---
detector = None

def load_ai_model():
    """AIモデルをロードする関数"""
    global detector
    model_url = "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1"
    print("物体検出AIモデルをロードしています... しばらくお待ちください。")
    try:
        detector = hub.load(model_url)
        print("AIモデルのロードが完了しました。")
        return True
    except Exception as e:
        print(f"AIモデルのロード中にエラーが発生しました: {e}")
        return False

def is_person_or_object(image_np, detection_threshold=0.45):
    """物体検出と判定を行う関数"""
    global detector
    if detector is None:
        return "エラー: AIモデルがロードされていません。"

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...] # バッチ次元を追加

    try:
        detections = detector(input_tensor)
    except Exception as e:
        return f"推論中にエラーが発生しました: {e}"

    detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)
    detection_scores = detections['detection_scores'][0].numpy()

    # COCOデータセットのクラスID 1 は 'person'
    person_class_id = 1

    # 判定ロジック
    for i in range(len(detection_scores)):
        if detection_scores[i] > detection_threshold:
            if detection_classes[i] == person_class_id:
                return "人" # 人が検出された

    if np.any(detection_scores > detection_threshold):
        return "物" # 人以外の物体が検出された
    else:
        return "不明 (閾値を超える物体は検出されませんでした)"

def predict_with_gradio(image_pil):
    """Gradioインターフェース用の予測ラッパー関数"""
    if detector is None:
        return "AIモデルのロードに失敗しています。"
    
    if image_pil is None:
        return "画像が提供されていません。"

    # PIL ImageをNumPy配列に変換
    try:
        image_np = np.array(image_pil)
    except Exception as e:
        return f"画像の形式変換中にエラーが発生しました: {e}"

    # 画像の前処理（RGBA -> RGBなど）
    if image_np.ndim == 3 and image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    elif image_np.ndim == 2:
        image_np = np.stack((image_np,)*3, axis=-1)

    return is_person_or_object(image_np)

# --- メイン処理 ---
if __name__ == "__main__":
    # 起動時にモデルをロード
    if load_ai_model():
        # Gradioインターフェースの作成
        iface = gr.Interface(
            fn=predict_with_gradio,
            inputs=gr.Image(type="pil", label="判定したい画像をアップロード"),
            outputs=gr.Label(label="判定結果"),
            title="【AIデモ】画像が「人」か「物」か判定します",
            description="TensorFlow Hubのモデルを使用して、アップロードされた画像を解析します。",
            allow_flagging='never'
        )
        
        # アプリの起動
        print("Gradioアプリを起動します...")
        iface.launch(share=True) 
    else:
        print("モデルのロードに失敗したため、アプリを終了します。")
