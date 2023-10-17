import streamlit as st
import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# 定数の宣言
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def load_model():
    model = hub.load(SAVED_MODEL_PATH)
    return model

model = load_model()

def preprocess_image(image):
  """ 入力画像を前処理してモデル化する
      引数:
        image: PIL 画像オブジェクト
  """
  hr_image = np.array(image)
  # PNG の場合は、アルファ チャネルを削除します。モデルのみサポート
  # 3 色チャンネルの画像
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """
    スケーリングされていない Tensor イメージを保存します。
    引数:
        image: 3D 画像テンソル。 [高さ、幅、チャンネル]
        filename: 保存するファイルの名前。
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    # Convert the image to an RGB mode if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)


def plot_image(image, title=""):
  """
    画像テンソルから画像をプロットします。
    引数:
      image: 3D 画像テンソル。 [高さ、幅、チャンネル]。
      title: プロットに表示するタイトル。
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  st.image(image, caption=title, use_column_width=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("高解像")
st.sidebar.title("設定")
contrast = st.sidebar.slider('コントラスト', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
st.sidebar.write('コントラスト:', contrast)

brightness = st.sidebar.slider('輝度', min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
st.sidebar.write('輝度:', brightness)

gamma = st.sidebar.slider('ガンマ', min_value=0.1, max_value=2.0, value=1.0, step=0.01)
st.sidebar.write('ガンマ:', gamma)

hue = st.sidebar.slider('色調', min_value=-0.5, max_value=0.5, value=0.0, step=0.01)
st.sidebar.write('色調:', hue)

image_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image, caption="オリジナルの画像", use_column_width=True)
    hr_image = preprocess_image(input_image)

    if st.button('高画質化'):
        if hr_image is not None:
            # Loading the model
            start = time.time()
            fake_image = model(hr_image)
            fake_image = tf.squeeze(fake_image)
            st.write("所要時間 : ", time.time() - start)

            # 超解像画像の表示
            st.write("")
            st.write("## 高画質化")
            st.write("")

            # コントラスト、明るさ、ガンマ補正を適用する
            fake_image = tf.image.adjust_contrast(fake_image, contrast)
            fake_image = tf.image.adjust_brightness(fake_image, brightness)
            fake_image = tf.image.adjust_gamma(fake_image, gamma)
            fake_image = tf.image.adjust_hue(fake_image, hue)




            # 色とコントラストを調整した画像を表示する
            plot_image(tf.squeeze(fake_image), title="色とコントラストを調整した画像")

            # 色とコントラストを調整した超解像画像を保存する
            save_image(tf.squeeze(fake_image), filename="解像度調整済み")

else:
    st.write("画像をアップロードして開始します")
