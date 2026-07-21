from google import genai

client = genai.Client()  # GEMINI_API_KEY を自動で読む

history = []  # 会話の履歴をここに溜めていく

print("Geminiとの会話を始めます（終わるには quit と入力）")

while True:
    user_input = input("あなた: ")

    if user_input.strip().lower() in ("quit", "exit"):
        print("会話を終了します。")
        break

    # 1. 今回のあなたの発言を履歴に追加
    history.append({"role": "user", "parts": [{"text": user_input}]})

    # 2. これまでの履歴を丸ごと渡して返事をもらう
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=history,
    )
    reply = response.text

    # 3. モデルの返事も履歴に追加（次のターンの文脈になる）
    history.append({"role": "model", "parts": [{"text": reply}]})

    print("Gemini:", reply)