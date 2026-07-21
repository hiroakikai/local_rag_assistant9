import time
from google import genai

client = genai.Client()

# 混雑(503)やエラー時に、待ち時間を伸ばしながら数回やり直す共通関数
def call_with_retry(do_call, tries=5):
    for attempt in range(tries):
        try:
            return do_call()
        except Exception:
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 秒と伸ばしていく
            print(f"  混雑のため {wait} 秒待って再試行します（{attempt + 1}/{tries}）")
            time.sleep(wait)
    return None

# 1. 長い文書（本来はあなたの業務基準など）
document = """当社の勤務時間は原則として午前9時から午後6時までとし、休憩を1時間含む。フレックスタイム制度を利用する場合、コアタイムの午前11時から午後3時までは在席を必須とする。

有給休暇は入社6か月経過後に10日が付与され、勤続年数に応じて増える。取得する場合は原則3営業日前までに勤怠システムから申請し、上長の承認を得ること。

在宅勤務は週3日を上限として申請できる。事前に上長の承認が必要で、機密情報を扱う業務は社内ネットワークからのみアクセスする。私物端末での業務利用は禁止する。

経費精算は毎月末締めとし、領収書を添付のうえ翌月5日までに提出する。承認後、翌月25日の給与と合わせて振り込まれる。交通費は最も経済的な経路を基準に支給する。

社用パソコンや業務システムに障害が発生した場合は、速やかに情報システム部（内線200）へ連絡する。パスワードは3か月ごとに変更し、他者と共有してはならない。"""

# 2. 長い文書を小さなチャンクに分割
def chunk_text(text, size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

chunks = chunk_text(document)
print(f"文書を {len(chunks)} 個のチャンクに分割しました")

# 3. 各チャンクをベクトル化（混雑してもリトライ）
doc_result = call_with_retry(lambda: client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks,
))
if doc_result is None:
    print("起動時の埋め込みに失敗しました。時間をおいて再実行してください。")
    raise SystemExit

chunk_vectors = [e.values for e in doc_result.embeddings]

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
    q_result = call_with_retry(lambda: client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
    ))
    if q_result is None:
        print("混雑が続いています。少し時間をおいて試してください。\n")
        continue
    q_vector = q_result.embeddings[0].values

    # 5. 似ているチャンクを上位3件えらぶ
    scored = []
    for i, (chunk, vec) in enumerate(zip(chunks, chunk_vectors)):
        scored.append((cosine(q_vector, vec), i, chunk))
    scored.sort(reverse=True)
    top = scored[:3]
    context = "\n\n".join(chunk for score, i, chunk in top)

    # 6. 上位チャンクを根拠に答えさせる
    prompt = f"""次の情報だけを根拠に、質問へ日本語で答えてください。
情報に答えが無ければ「わかりません」と答えてください。

情報:
{context}

質問: {question}"""

    response = call_with_retry(lambda: client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    ))
    if response is None:
        print("混雑が続いています。少し時間をおいて試してください。\n")
        continue

    # 7. 回答と出典を表示
    print("\n回答:", response.text)
    print("\n--- 根拠にした出典 ---")
    for rank, (score, i, chunk) in enumerate(top, start=1):
        preview = chunk.replace("\n", " ")[:60]
        print(f"[{rank}] チャンク#{i}（類似度 {score:.3f}）: {preview}…")
    print()