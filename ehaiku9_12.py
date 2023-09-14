import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

import os
import openai
#openai.api_key = os.environ.get('CHATGPT_API_KEY')
openai.api_key = st.secrets["api_key"]

import json
import requests
import torchvision.models as models
import torchvision.transforms as transforms

# ImageNetクラスラベルをダウンロードする
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(url)
labels = response.json()

with open("imagenet_classes.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

# 画像の前処理を行う関数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# ResNet50モデルをロードして画像を分類する関数
def classify_image(image_tensor):
    model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()  # モデルを評価モードに設定

    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        class_index = predicted.item()

    with open("imagenet_classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    english_label = classes[class_index].replace(" ", "_")  # スペースをアンダースコアに置き換える

    with open("imagenet_class_index.json", "r", encoding="utf-8") as f2:
        labels_jp = json.load(f2)

    # japanese_labelの取得方法を修正
    japanese_label = next((item["ja"] for item in labels_jp if item["en"] == english_label), "未知のラベル")

    return japanese_label

#image_path = '/content/海岸.jpg'
#predicted_class_jp = classify_image(image_path)
#print(f"この画像はおそらく{predicted_class_jp}です。")

# 2. 画像の内容を基に俳句を生成する関数
def generate_haiku(label):

    input_text = label
    #openai.api_key= os.environ.get("OPENAI_API_KEY")
    # 俳句を生成
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",

        messages=[
        {"role": "system", "content": """あなたは俳句の達人です。喜怒哀楽や季節感を表現する詩人です。
    俳句とは、ユーザーが提供するテーマを元に、説明ではなく、5・7・5という音節数で俳句としての詩的な表現を行ってください。
    俳句の例として以下のようなものがあります。

    * 古池や 蛙飛びこむ 水の音
    * 若草や つわものどもが 夢の跡
    * 柿食えば 鐘が鳴るなり 法隆寺
    * 梅一輪 一輪ほどの あたたかさ
    * 静かさや 岩にしみ入る 蝉の声
        """},



        {"role": "user", "content": input_text}
      ],
        temperature=0.7,
        max_tokens=25,
        n=5
    )

    # 生成した俳句を表示

#    print(response['choices'])
    choices = response.choices
#    for i, output in enumerate(choices):
#        print(f'{i+1}. {output["message"]["content"]}')
    return choices
#    for i, output in enumerate(response['choices']):
#       st.write(f'{i+1}. {output["text"]}')
#        print(f'{i+1}. {output["text"]}')


# 3. Streamlitを使用してユーザーインターフェースを作成
st.title("画像から俳句を生成")

uploaded_file = st.file_uploader("画像を入れて...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("考え中...")

    image_tensor = preprocess_image(uploaded_file)
    label = classify_image(image_tensor)
    choices = generate_haiku(label)

    st.write("こんなのできました！！:")
    #st.write(haiku)
    for i, output in enumerate(choices):
        st.write(f'{i+1}. {output["message"]["content"]}')