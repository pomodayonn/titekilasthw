from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from PIL import Image
from datetime import date
from io import BytesIO
import requests
import os
import base64
import time


app = Flask(__name__)

# Gemini API
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY が設定されていません。\n"
    )

client = genai.Client(api_key=api_key)

# 文章生成用モデル
GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

# 画像生成用モデル
IMAGE_MODELS = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
]


# 天気コードを日本語に変換
def weather_code_to_text(code):
    if code == 0:
        return "晴れ"
    elif 1 <= code <= 3:
        return "くもり"
    elif 45 <= code <= 57:
        return "霧"
    elif 61 <= code <= 67:
        return "雨"
    elif 71 <= code <= 77:
        return "雪"
    elif 80 <= code <= 82:
        return "にわか雨"
    elif 95 <= code <= 99:
        return "雷雨"
    else:
        return "不明"


# 天気API
def fetch_weather(latitude=35.68, longitude=139.76):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&daily=temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max,weather_code"
        "&timezone=Asia/Tokyo"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        daily = data["daily"]

        max_temp = daily["temperature_2m_max"][0]
        min_temp = daily["temperature_2m_min"][0]
        rain_prob = daily["precipitation_probability_max"][0]
        weather_code = daily["weather_code"][0]
        weather_text = weather_code_to_text(weather_code)

        return (
            f"最高気温: {max_temp}℃ / "
            f"最低気温: {min_temp}℃ / "
            f"降水確率: {rain_prob}% / "
            f"天気: {weather_text}"
        )

    except Exception as e:
        print("天気情報の取得に失敗しました:", e)
        return "天気情報は取得できませんでした"


# 祝日API
def fetch_holiday():
    url = "https://holidays-jp.github.io/api/v1/date.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        holidays = response.json()

        today = date.today().isoformat()

        if today in holidays:
            return f"本日は祝日です: {holidays[today]}"
        else:
            return "本日は平日です"

    except Exception as e:
        print("祝日情報の取得に失敗しました:", e)
        return "祝日情報は取得できませんでした"


# Base64画像をPIL画像に変換
def decode_image(base64_image):
    header, encoded = base64_image.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")

    return image


# ファッション提案文章を生成
def analyze_fashion(
    image,
    weather_text,
    holiday_text,
    trend_keywords,
    event_keywords,
    purpose_text,
    body_type,
    height_cm,
    fit_preference
):
    prompt = f"""
あなたはファッションアドバイザーです。
以下の画像はユーザーの現在の服装です。

ユーザーの服装画像と、天気・祝日・流行キーワード・行くイベント情報をもとに、
今日に合ったファッションアドバイスを日本語でしてください。
スマホ画面でも読みやすいように、1文を短めにしてください。

以下の構成で出力してください。

## 👀 画像から抽出した服装特徴
- 服の種類：
- 色：
- 雰囲気：
- 季節感：
- 天気との相性：

## 🔍 抽出した重要キーワード
ユーザーの入力文、イベント情報、流行キーワードから、服装提案に重要なキーワードを3〜5個抽出してください。
- 
- 
- 

## 現在の服装の良いポイント
- 良い点を2〜3個

## 今日の天気に対する注意点
- 気温、雨、寒暖差に合わせた注意点

## トレンドを取り入れるなら
- 流行キーワードを使った改善アイデア

## イベントに合わせた工夫
- 行く場所や予定に合わせた服装の工夫

## 追加するとよいアイテム
- バッグ、靴、羽織、傘など

## 骨格・サイズ感を考慮した提案
- 骨格タイプ、身長、サイズ感の好みが入力されている場合のみ、丈感・シルエット・着心地の観点で提案してください。
- 入力されていない情報には触れないでください。

## 今日のおすすめコーデ
- 具体的な組み合わせを提案



## ひとことまとめ
**短くおすすめをまとめてください。**

【天気情報】
{weather_text}

【祝日情報】
{holiday_text}

【流行キーワード】
{trend_keywords}

【行くイベント・予定】
{event_keywords}

【服装の目的】
{purpose_text}

【ユーザーが任意入力した体型・サイズ情報】
骨格タイプ: {body_type}
身長: {height_cm}cm
サイズ感の好み: {fit_preference}

【注意】
- 体型や容姿を否定しない
- 体重や体型を評価しない
- 骨格タイプや身長は、丈感・シルエット・着心地の提案にのみ使う
- 入力が空欄の場合は、その情報には触れない
- 顔や身体的特徴ではなく、服・色・素材・季節感・天気との相性に注目する
- やさしく具体的に提案する
- 雨や寒暖差がある場合は、傘・羽織・靴なども提案する
"""

    last_error = None

    for model_name in GEMINI_MODELS:
        try:
            print(f"使用モデル: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image]
            )

            if response.text:
                return response.text

            print(f"{model_name} は空のテキストを返しました。")

        except Exception as e:
            print(f"{model_name} でエラーが発生しました。")
            print(e)
            last_error = e
            time.sleep(2)

    raise RuntimeError(f"すべてのGemini文章生成モデルで失敗しました: {last_error}")


# 提案内容からコーデ例画像を生成
def generate_outfit_image(advice_text, weather_text, event_keywords, trend_keywords):
    image_prompt = f"""
Create a clean fashion flat lay image based on the following Japanese fashion advice.

Important rules:
- Do not generate a real person.
- Do not generate a face.
- Do not generate a body shape.
- Generate only clothing items arranged neatly on a light background.
- No text in the image.
- No logo.
- No brand name.

Style:
- fashion magazine flat lay
- clean and cute
- realistic clothing items
- suitable for Japanese young adults
- light background
- one complete outfit example
- top, bottom, shoes, bag, and accessories

Weather:
{weather_text}

Event:
{event_keywords}

Trend keywords:
{trend_keywords}

Fashion advice:
{advice_text}

Generate one example outfit image that matches the advice.
"""

    image_models = [
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
    ]

    last_error = None

    for model_name in image_models:
        try:
            print(f"画像生成モデル: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["Image"]
                )
            )

            print("画像生成レスポンスを受信しました")

            # response.parts が使える場合
            parts = []

            if hasattr(response, "parts") and response.parts:
                parts = response.parts
            elif hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        parts.extend(candidate.content.parts)

            for part in parts:
                # テキストが返ってきた場合はログだけ出す
                if getattr(part, "text", None):
                    print("画像生成モデルからのテキスト:", part.text)
                    continue

                # 画像が返ってきた場合
                if getattr(part, "inline_data", None):
                    try:
                        generated_image = part.as_image()

                        buffer = BytesIO()
                        generated_image.save(buffer, format="PNG")

                        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

                        print("画像生成に成功しました")
                        return f"data:image/png;base64,{encoded}"

                    except Exception as image_parse_error:
                        print("画像データの変換に失敗しました:", image_parse_error)

            print(f"{model_name} では画像パーツが見つかりませんでした。")

        except Exception as e:
            print(f"{model_name} で画像生成に失敗しました")
            print("エラー内容:", repr(e))
            last_error = e
            time.sleep(2)

    print("すべての画像生成モデルで失敗しました:", repr(last_error))
    return None


# 画面表示
@app.route("/")
def index():
    return render_template("index.html")


# 分析API
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        print("=== /analyze が呼ばれました ===")

        data = request.get_json(silent=True)

        if data is None:
            print("JSONデータが受け取れませんでした")
            return jsonify({
                "error": "JSONデータが受け取れませんでした"
            }), 400

        image_data = data.get("image")
        trend_keywords = data.get("trend_keywords", "")
        event_keywords = data.get("event_keywords", "")
        purpose_text = data.get("purpose_text", "")
        latitude = data.get("latitude", 35.68)
        longitude = data.get("longitude", 139.76)
        body_type = data.get("body_type", "")
        height_cm = data.get("height_cm", "")
        fit_preference = data.get("fit_preference", "")

        print("画像データあり:", bool(image_data))
        print("流行キーワード:", trend_keywords)
        print("イベント:", event_keywords)
        print("緯度:", latitude)
        print("経度:", longitude)
        print("骨格タイプ:", body_type)
        print("身長:", height_cm)
        print("サイズ感の好み:", fit_preference)

        if not image_data:
            return jsonify({
                "error": "画像が送信されていません"
            }), 400

        print("画像を変換中...")
        image = decode_image(image_data)

        print("天気情報を取得中...")
        weather_text = fetch_weather(latitude, longitude)
        print("天気:", weather_text)

        print("祝日情報を取得中...")
        holiday_text = fetch_holiday()
        print("祝日:", holiday_text)

        print("Geminiで文章分析中...")
        result = analyze_fashion(
            image=image,
            weather_text=weather_text,
            holiday_text=holiday_text,
            trend_keywords=trend_keywords,
            event_keywords=event_keywords,
            purpose_text=purpose_text,
            body_type=body_type,
            height_cm=height_cm,
            fit_preference=fit_preference
        )

        print("文章分析完了")

        outfit_image = None

        try:
            print("コーデ画像を生成中...")
            outfit_image = generate_outfit_image(
                advice_text=result,
                weather_text=weather_text,
                event_keywords=event_keywords,
                trend_keywords=trend_keywords
            )
        except Exception as image_error:
            print("コーデ画像生成でエラーが発生しました:", image_error)
            outfit_image = None

        print("分析完了")

        return jsonify({
            "weather": weather_text,
            "holiday": holiday_text,
            "trend_keywords": trend_keywords,
            "event_keywords": event_keywords,
            "purpose_text": purpose_text,
            "result": result,
            "outfit_image": outfit_image
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": f"Flask側でエラーが発生しました: {str(e)}"
        }), 500


# 起動
if __name__ == "__main__":
    app.run(debug=False)