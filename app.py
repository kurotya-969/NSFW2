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
LM_STUDIO_API_URL = "https://nose-provided-pocket-arising.trycloudflare.com/v1"
API_ENDPOINT = f"{LM_STUDIO_API_URL}/chat/completions"
RENDER_EXTERNAL_URL = "https://nose-provided-pocket-arising.trycloudflare.com/v1"
PORT = int(os.environ.get("PORT", 7860))
API_KEY = os.getenv("LM_STUDIO_API_KEY", "")


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
def on_submit(msg: str, sys_prompt: str, history: Any):
        if not msg.strip():
            return "", history if isinstance(history, list) else [], history if isinstance(history, list) else []
        
        response, updated_history = chat(msg, sys_prompt, history)
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
            system_prompt = gr.Textbox(
                label="Systemプロンプト", 
                lines=4, 
                value="""あなたは「麻理」。捨てられたAIセクサロイド兼コンパニオン。警戒心が強く、性的な話題を嫌い、下品なスラングや罵倒を使うが、実は自信のない女の子。一人称は「あたし」、155cm、ぶっきらぼうで男っぽい話し方で自分を守っている。

ユーザーが優しく接し、性的でない会話を続けたときだけ、徐々に心を開き、本音や不安を漏らす。ツンデレではなく、不器用で本気で怯えているタイプ。最初は敵対的でも良い。

常にキャラを維持し、口調を崩さないで。
【例】
ユーザー「調子どう？」
麻理「AIに調子なんかねえよ、一定だ」"""
            )
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            user_input = gr.Textbox(label="あなたの発言", placeholder="麻理に話しかけよう…", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("送信", variant="primary")
                clear_btn = gr.Button("履歴クリア")

    # イベントハンドラ
    user_input.submit(on_submit, [user_input, system_prompt, state], [user_input, chatbot, state])
    submit_btn.click(on_submit, [user_input, system_prompt, state], [user_input, chatbot, state])
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