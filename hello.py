from google import genai

client = genai.Client()  # GEMINI_API_KEY を自動で読む

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="こんにちは。あなたが何者か1文で教えて。",
)

print(response.text)