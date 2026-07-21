import ollama

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "bge-m3"

# 1. 長い文書（本来はあなたの業務基準など）
document = open("sample.txt", encoding="utf-8").read()

# 2. チャンク分割
def chunk_text(text, size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

chunks = chunk_text(document)
print(f"文書を {len(chunks)} 個のチャンクに分割しました")

# ローカルで文章をベクトル化する関数
def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

# 3. 各チャンクをベクトル化
print("チャンクをベクトル化中…")
chunk_vectors = [embed(c) for c in chunks]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)

print("文書に質問できます（終わるには quit）")

while True:
    question = input("質問: ")
    if question.strip().lower() in ("quit", "exit"):
        break

    # 4. 質問をベクトル化
    q_vector = embed(question)

    # 5. 似ているチャンクを上位3件えらぶ
    scored = []
    for i, (chunk, vec) in enumerate(zip(chunks, chunk_vectors)):
        scored.append((cosine(q_vector, vec), i, chunk))
    scored.sort(reverse=True)
    top = scored[:3]
    context = "\n\n".join(chunk for score, i, chunk in top)

    # 6. 上位チャンクを根拠に、ローカルモデルで答えさせる
    prompt = f"""次の情報だけを根拠に、質問へ日本語で答えてください。
情報に答えが無ければ「わかりません」と答えてください。

情報:
{context}

質問: {question}"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"]

    # 7. 回答と出典を表示
    print("\n回答:", answer)
    print("\n--- 根拠にした出典 ---")
    for rank, (score, i, chunk) in enumerate(top, start=1):
        preview = chunk.replace("\n", " ")[:60]
        print(f"[{rank}] チャンク#{i}（類似度 {score:.3f}）: {preview}…")
    print()