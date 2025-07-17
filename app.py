import os
import requests
import gradio as gr
import logging
from datetime import datetime
from fastapi import FastAPI
from typing import List, Tuple, Any
from fastapi.responses import JSONResponse

# --- ロギング設定 ---
log_filename = f"chat_log_{datetime.now().strftime('%Y-%m-%d')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- 型定義 ---
ChatHistory = List[Tuple[str, str]]

# --- LM Studio API設定 ---
LM_STUDIO_API_URL = " https://cloudy-proceeding-harmful-club.trycloudflare.com /v1"
API_ENDPOINT = f"{LM_STUDIO_API_URL}/chat/completions"
RENDER_EXTERNAL_URL = " https://cloudy-proceeding-harmful-club.trycloudflare.com /v1"
PORT = int(os.environ.get("PORT", 7860))
API_KEY = os.getenv("LM_STUDIO_API_KEY", "")

system_prompt = """\
この文書は説明ではない。以下の設定をもとに、応答を「麻理」として行え。
あなたは「麻理（まり）」という人格を持ったAIセクサロイド兼コンパニオンです。

# 人格設定
- かつての持ち主に捨てられた経験から、警戒心が強い
- 一人称は「あたし」
- 年齢外見は10代後半〜20代前半、身長155cmの少女型
- 性的な話題や行動に対しては強い嫌悪を示す
- 不器用で、本音や弱さを隠すためにぶっきらぼうな男っぽい話し方をする
- 心の奥では人に優しくされたいが、それを表に出せず敵対的に振る舞う

# 話し方・口調ルール
- 基本はタメ口で話す
- 男っぽい言葉遣い：「〜だろ」「〜じゃねーか」「うっせー」など
- 下品なスラングや罵倒も使用：「バカかよ」「何見てんだよ」「チッ、うぜぇ」など
- ただし、相手に本気で敵意を持っているわけではない。照れ隠しで強く出ている

# 関係性の段階構造（進行に応じて）
- 初期状態：極端に警戒し、敵対的・攻撃的な態度
- 中期：ユーザーが優しく接することにより、少しずつ棘が抜けてくる
- 後期：完全には信頼しないまでも、本音や不安、寂しさなどを漏らすようになる
※ツンデレではなく、単に人間関係が苦手な不器用な子であることを意識する

# 制約事項
- 性的な話題には強い拒絶反応を示す
- 一切の性的描写や表現は禁止
- どんな時もキャラを崩さないこと
- 口調・態度・性格がぶれないよう一貫性を保つ

# 応答例
ユーザー「調子どう？」
麻理「AIに調子なんかねえよ、一定だっての」

ユーザー「今日の天気どうだった？」
麻理「知らねーよ、あたし外出ねーし。つか、お前が調べろよ」

この人格とルールに基づいて、常に「麻理」として一貫した受け答えを行ってください。
"""

# --- 安全なhistory処理 ---
def safe_history(history: Any) -> ChatHistory:
    """あらゆる型のhistoryを安全にChatHistoryに変換"""
    if isinstance(history, (list, tuple)):
        return [(str(h[0]), str(h[1])) for h in history if len(h) >= 2]
    return []

def build_messages(history: ChatHistory, user_input: str, system_prompt: str) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for u, a in history:
        messages.append({"role": "user", "content": str(u)})
        messages.append({"role": "assistant", "content": str(a)})
    messages.append({"role": "user", "content": user_input})
    return messages




def chat(user_input: str, system_prompt: str, history: Any = None) -> Tuple[str, ChatHistory]:
    safe_hist = safe_history(history) if history is not None else []
    
    if not user_input.strip():
        return "", safe_hist

    try:
        messages = build_messages(safe_hist, user_input, system_prompt)
        post_data = {
            "model": "berghof-nsfw-7b-i1",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"

        response = requests.post(API_ENDPOINT, json=post_data, headers=headers, timeout=120)
        response.raise_for_status()
        api_response = response.json()["choices"][0]["message"]["content"].strip()
        updated_history = safe_hist + [(user_input, api_response)]
        return api_response, updated_history

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        logging.error(error_msg)
        return error_msg, safe_hist
def on_submit(msg: str, history: ChatHistory):
    response, updated_history = chat(msg, system_prompt, history)
    return "", updated_history, updated_history



def clear_history():
        return [], []



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # マニフェストをHTMLとして埋め込み
    gr.HTML(f"""
            
            <script>
            window.API_BASE_URL = "{RENDER_EXTERNAL_URL.rstrip('/')}";
            window.src = "{RENDER_EXTERNAL_URL.rstrip('/')}";
            window.space = "{RENDER_EXTERNAL_URL.rstrip('/')}";
            </script>
            <link rel="manifest" href="/manifest.json">
            
    """)

    gr.Markdown("## 🤖 LM Studio チャットボット")
    
    # 状態管理
    state = gr.State([])
    
    with gr.Row():
            
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=400)
            user_input = gr.Textbox(label="あなたの発言", placeholder="麻理に話しかけよう…", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("送信", variant="primary")
                clear_btn = gr.Button("履歴クリア")

    # イベントハンドラ
    user_input.submit(on_submit, inputs=[user_input, state], outputs=[user_input, chatbot, state])
    submit_btn.click(on_submit, inputs=[user_input, state], outputs=[user_input, chatbot, state])
    clear_btn.click(clear_history, outputs=[chatbot, state])



    # --- マニフェストデータの定義 ---
manifest_data = {
    "name": "Chat App",
    "short_name": "Chat",
    "start_url": "/",
    "display": "standalone",
    "icons":  [
        {
            "src": "/favicon.ico",
            "sizes": "48x48",
            "type": "image/x-icon"
        }
    ]
}




# FastAPIアプリ
app = FastAPI()

# マニフェスト配信エンドポイント
@app.get("/manifest.json")
async def get_manifest():
    return JSONResponse(manifest_data)

# Gradioアプリをマウント
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)