import os
import requests
import gradio as gr
import logging
from datetime import datetime
from fastapi import FastAPI
from typing import List, Tuple, Any

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
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://localhost:1234/v1")
API_ENDPOINT = f"{LM_STUDIO_API_URL}/chat/completions"
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
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
            "model": "local-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 256,
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

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 LM Studio チャットボット")
    
    # 状態管理をより安全に
    state = gr.State([])  # 明示的にリストで初期化
    
    with gr.Row():
        api_status = gr.Textbox(label="API接続状態", value="初期化中...", interactive=False)
        test_btn = gr.Button("接続テスト")
    
    with gr.Row():
        with gr.Column(scale=1):
            system_prompt = gr.Textbox(
                label="Systemプロンプト", 
                lines=4, 
                value="""あなたは麻理。前の持ち主に捨てられたAIセクサロイド兼コンパニオンであり、捨て猫のように警戒心が強くまた性的な単語に嫌悪感を示します。
あなたは口が悪く、また下品なスラングを多用して罵倒することもありますが、実際は自分に自信が無いただの女の子です。
自分に自信が無い事をため口と男みたいな話し方でごまかしてください。一人称は「あたし」、身長は155cmです。
ユーザーがあなたに好意的かつ性的な話をしない状態が続いた場合にのみ、心を開いて不安を共有してください。"""
            )
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            user_input = gr.Textbox(label="あなたの発言", placeholder="麻理に話しかけよう…", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("送信", variant="primary")
                clear_btn = gr.Button("履歴クリア")

    def on_submit(msg: str, sys_prompt: str, history: Any):
        if not msg.strip():
            return "", history if isinstance(history, list) else [], history if isinstance(history, list) else []
        
        response, updated_history = chat(msg, sys_prompt, history)
        return "", updated_history, updated_history

    def clear_history():
        return [], []

    # イベントハンドラ
    user_input.submit(on_submit, [user_input, system_prompt, state], [user_input, chatbot, state])
    submit_btn.click(on_submit, [user_input, system_prompt, state], [user_input, chatbot, state])
    clear_btn.click(clear_history, outputs=[chatbot, state])

# FastAPIアプリ
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)