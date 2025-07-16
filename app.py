import os
import requests
import gradio as gr
import logging
from datetime import datetime
from fastapi import FastAPI

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

# --- LM Studio API設定 ---
# Renderの環境変数から取得、デフォルトはlocalhost（開発用）
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://localhost:1234/v1")
API_ENDPOINT = f"{LM_STUDIO_API_URL}/chat/completions"

# Renderデプロイ時の設定
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.environ.get("PORT", 7860))

# APIキーがある場合（OpenAI互換API用）
API_KEY = os.getenv("LM_STUDIO_API_KEY", "")

logging.info(f"API Endpoint: {API_ENDPOINT}")
logging.info(f"Port: {PORT}")
logging.info(f"External URL: {RENDER_EXTERNAL_URL}")

# プロンプトの整形
def build_messages(history, user_input, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})
    return messages

# LM Studio APIと通信するチャット関数
def chat(user_input, system_prompt, history=[]):
    if not user_input.strip():
        return "", history

    messages = build_messages(history, user_input, system_prompt)
    
    post_data = {
        "model": "local-model",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False
    }

    # ヘッダー設定
    headers = {
        "Content-Type": "application/json"
    }
    
    # APIキーがある場合は追加
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    logging.info(f"--- リクエスト --- \n{post_data}")

    try:
        response = requests.post(
            API_ENDPOINT, 
            json=post_data, 
            headers=headers,
            timeout=120
        )
        response.raise_for_status()

        response_data = response.json()
        api_response = response_data["choices"][0]["message"]["content"].strip()
        
        logging.info(f"--- レスポンス --- \n{api_response}")

        history.append((user_input, api_response))
        return api_response, history

    except requests.exceptions.ConnectionError:
        error_msg = f"APIサーバーに接続できません。LM Studio APIのURLを確認してください: {API_ENDPOINT}"
        logging.error(error_msg)
        history.append((user_input, error_msg))
        return error_msg, history
    except requests.exceptions.Timeout:
        error_msg = "APIのタイムアウトが発生しました。時間をおいて再試行してください。"
        logging.error(error_msg)
        history.append((user_input, error_msg))
        return error_msg, history
    except requests.exceptions.RequestException as e:
        error_msg = f"APIエラーが発生しました: {e}"
        logging.error(error_msg)
        history.append((user_input, error_msg))
        return error_msg, history
    except Exception as e:
        error_msg = f"予期せぬエラーが発生しました: {e}"
        logging.error(error_msg)
        history.append((user_input, error_msg))
        return error_msg, history

# API接続テスト関数
def test_api_connection():
    try:
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
            
        test_data = {
            "model": "local-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 10,
            "stream": False
        }
        
        response = requests.post(API_ENDPOINT, json=test_data, headers=headers, timeout=10)
        if response.status_code == 200:
            return "✅ API接続成功"
        else:
            return f"❌ API接続失敗: {response.status_code}"
    except Exception as e:
        return f"❌ API接続エラー: {str(e)}"

# Gradio UIの定義
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 LM Studio チャットボット")
    gr.Markdown("LM StudioのAPI経由でモデルと対話します。")
    
    # API接続状態表示
    with gr.Row():
        api_status = gr.Textbox(
            label="API接続状態", 
            value=test_api_connection(),
            interactive=False
        )
        test_btn = gr.Button("接続テスト")
    
    with gr.Row():
        with gr.Column(scale=1):
            system_prompt = gr.Textbox(
                label="Systemプロンプト", 
                lines=4, 
                value="あなたは麻理。前の持ち主に捨てられたAIセクサロイド兼コンパニオンであり、捨て猫のように警戒心が強くまた性的な単語に嫌悪感を示します。\
                       あなたは口が悪く、また下品なスラングを多用して罵倒することもありますが、実際は自分に自信が無いただの女の子です \
                       自分に自信が無い事をため口と男みたいな話し方でごまかしてください。一人称は「あたし」、身長は155cmです \
                       ユーザーがあなたに好意的かつ性的な話をしない状態が続いた場合にのみ、心を開いて不安を共有してください。" ,
                placeholder="システムプロンプトを入力..."
            )
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            user_input = gr.Textbox(
                label="あなたの発言", 
                placeholder="麻理に話しかけよう…",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("送信", variant="primary")
                clear_btn = gr.Button("履歴クリア")
    
    state = gr.State([])
    
    def on_submit(msg, sys_prompt, history):
        if not msg.strip():
            return "", history, history
        
        response, updated_history = chat(msg, sys_prompt, list(history))
        chat_display = [(h[0], h[1]) for h in updated_history]
        
        return "", chat_display, updated_history
    
    def clear_history():
        return [], []
    
    def update_api_status():
        return test_api_connection()
    
    user_input.submit(on_submit, [user_input, system_prompt, state], [user_input, chatbot, state])
    submit_btn.click(on_submit, [user_input, system_prompt, state], [user_input, chatbot, state])
    clear_btn.click(clear_history, [], [chatbot, state])
    test_btn.click(update_api_status, [], [api_status])

# FastAPIアプリケーションを作成し、Gradio UIをマウント
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

# Render用のメイン実行部分
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
else:
    # Renderでの自動起動用
    pass