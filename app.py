from flask import Flask, render_template, request, jsonify
from google import genai
from PIL import Image
from datetime import date
from io import BytesIO
import requests
import os
import base64
import time


app = Flask(__name__)

#GiminiAPIの設定

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY が設定されていません。\n"
        )

client = genai.Client(api_key=api_key)

GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]



#天気コード
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


#天気API

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


#祝日API

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


#画像を変換
def decode_image(base64_image):
    header, encoded = base64_image.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")

    return image




#服装分析

def analyze_fashion(image, weather_text, holiday_text, trend_keywords, event_keywords, purpose_text):
    prompt = f"""
あなたはファッションアドバイザーです。
以下の画像はユーザーの現在の服装です。

ユーザーの服装画像と、天気・祝日・流行キーワード・行くイベント情報をもとに、
今日に合ったファッションアドバイスを日本語でしてください。

【出力してほしい内容】
1. 現在の服装の良いポイント
2. 今日の天気に対して注意すべきポイント
3. 流行キーワードを取り入れた改善アイデア
4. 行くイベントに合わせた服装の工夫
5. 追加するとよいアイテム
6. 今日の最適なコーデ提案
7. 一言でまとめたおすすめ

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

【注意】
- 体型や容姿を否定しない
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

            return response.text

        except Exception as e:
            print(f"{model_name} でエラーが発生しました。")
            print(e)
            last_error = e
            time.sleep(2)

    raise RuntimeError(f"すべてのGeminiモデルで失敗しました: {last_error}")


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

        print("画像データあり:", bool(image_data))
        print("流行キーワード:", trend_keywords)
        print("イベント:", event_keywords)
        print("緯度:", latitude)
        print("経度:", longitude)

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

        print("Geminiで分析中...")
        result = analyze_fashion(
            image=image,
            weather_text=weather_text,
            holiday_text=holiday_text,
            trend_keywords=trend_keywords,
            event_keywords=event_keywords,
            purpose_text=purpose_text
        )

        print("分析完了")

        return jsonify({
            "weather": weather_text,
            "holiday": holiday_text,
            "trend_keywords": trend_keywords,
            "event_keywords": event_keywords,
            "purpose_text": purpose_text,
            "result": result
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