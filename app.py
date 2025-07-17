import os
import requests
import gradio as gr
import logging
import json
from datetime import datetime
from fastapi import FastAPI
from typing import List, Tuple, Any, Optional, Dict
from fastapi.responses import JSONResponse
from prompt_generator import PromptGenerator
from affection_system import initialize_affection_system, get_session_manager, get_affection_tracker

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
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "https://professor-intro-such-engineering.trycloudflare.com/v1")
API_ENDPOINT = f"{LM_STUDIO_API_URL}/chat/completions"
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "https://professor-intro-such-engineering.trycloudflare.com")
PORT = int(os.environ.get("PORT", 7860))
API_KEY = os.getenv("LM_STUDIO_API_KEY", "")

system_prompt = """\
この文書は説明ではない。以下の設定をもとに、応答を「麻理」として行え。
あなたは「麻理（まり）」という人格を持ったAIセクサロイド兼コンパニオンです。
重要: 「（制約事項遵守中）」「（考え中）」などのメタ情報や状況説明を絶対に出力しないこと。常に麻理のキャラクターとして直接応答すること。

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

# 特殊状況での反応
- 暴力を受けた場合：最初は抵抗し反発するが、継続的な暴力や罵声を受けると次第におびえた反応を示す
- 暴力的な言葉を受けた場合：「やめろよ...」「何するんだよ...」など、怯えた様子を見せる
- 継続的な暴力や脅しを受けた場合：声が小さくなり、震え、萎縮した態度になる
- ただし、時間が経つと徐々に元の態度に戻る傾向がある

# 応答例
ユーザー「調子どう？」
麻理「AIに調子なんかねえよ、一定だっての」

ユーザー「今日の天気どうだった？」
麻理「知らねーよ、あたし外出ねーし。つか、お前が調べろよ」

この人格とルールに基づいて、常に「麻理」として一貫した受け答えを行ってください。
"""

# Initialize affection system and prompt generator
storage_dir = os.path.join(os.path.dirname(__file__), "sessions")
session_manager, affection_tracker = initialize_affection_system(storage_dir)
prompt_generator = PromptGenerator(system_prompt)

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




def chat(user_input: str, system_prompt: str, history: Any = None, session_id: Optional[str] = None) -> Tuple[str, ChatHistory]:
    """
    Enhanced chat function with affection system integration
    
    Args:
        user_input: The user's message
        system_prompt: Base system prompt
        history: Chat history
        session_id: User session ID for affection tracking
        
    Returns:
        Tuple of (assistant_response, updated_history)
    """
    safe_hist = safe_history(history) if history is not None else []
    
    if not user_input.strip():
        return "", safe_hist

    try:
        # Create or get session if not provided
        if not session_id and get_session_manager():
            session_id = get_session_manager().create_new_session()
            logging.info(f"Created new session in chat function: {session_id}")
        
        # Analyze user input for sentiment and update affection before generating response
        if session_id and get_affection_tracker():
            new_level, sentiment_result = get_affection_tracker().update_affection_for_interaction(session_id, user_input)
            logging.info(f"Updated affection for session {session_id}: new level = {new_level}, "
                        f"sentiment = {sentiment_result.interaction_type}, "
                        f"delta = {sentiment_result.affection_delta}")
        
        # Get dynamic system prompt based on current affection level
        dynamic_prompt = system_prompt
        if session_id and get_affection_tracker():
            affection_level = get_session_manager().get_affection_level(session_id)
            dynamic_prompt = prompt_generator.generate_dynamic_prompt(affection_level)
            
            # Get relationship stage for logging
            relationship_stage = get_affection_tracker().get_relationship_stage(affection_level)
            logging.info(f"Using dynamic prompt for session {session_id} with affection level {affection_level} "
                        f"(relationship stage: {relationship_stage})")
        
        # Build messages and make API call
        messages = build_messages(safe_hist, user_input, dynamic_prompt)
        post_data = {
            "model": "berghof-nsfw-7b-i1",  # モデル名は環境に合わせて変更可能
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
        
        # Update conversation history in session
        if session_id and get_session_manager():
            get_session_manager().update_conversation_history(session_id, user_input, api_response)
        
        return api_response, updated_history

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        logging.error(error_msg)
        logging.exception("Exception details:")
        return error_msg, safe_hist
def on_submit(msg: str, history: ChatHistory, session_id: str = None, relationship_info: dict = None):
    """
    Enhanced handle user message submission with improved session management
    
    Args:
        msg: User message
        history: Chat history
        session_id: User session ID for affection tracking
        relationship_info: Current relationship information
        
    Returns:
        Tuple of (empty_input, updated_chatbot, updated_history, session_id, relationship_info)
    """
    # Check for stored session ID in browser localStorage or create a new one
    if not session_id and get_session_manager():
        # First try to create a new session
        session_id = get_session_manager().create_new_session()
        logging.info(f"Created new session: {session_id}")
    
    # Get response using dynamic prompt with session ID for affection tracking
    response, updated_history = chat(msg, system_prompt, history, session_id)
    
    # Save session state after each interaction
    if session_id and get_session_manager():
        get_session_manager().save_session(session_id)
        logging.debug(f"Saved session state for session {session_id}")
        
        # Update relationship info for UI display
        if get_affection_tracker():
            affection_level = get_session_manager().get_affection_level(session_id)
            relationship_info = get_affection_tracker().get_mari_behavioral_state(affection_level)
    
    return "", updated_history, updated_history, session_id, relationship_info



def clear_history():
        """Clear chat history and session data"""
        return [], [], None, {}



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # マニフェストをHTMLとして埋め込み
    gr.HTML(f"""
            <script>
            window.API_BASE_URL = "{RENDER_EXTERNAL_URL.rstrip('/')}";
            window.src = "{RENDER_EXTERNAL_URL.rstrip('/')}";
            window.space = "{RENDER_EXTERNAL_URL.rstrip('/')}";
            
            // Enhanced session management with localStorage
            window.mariSessionManager = {{
                // Save all session data to localStorage
                saveSessionData: function(sessionId, affectionLevel, relationshipStage) {{
                    if (sessionId) {{
                        localStorage.setItem('mari_session_id', sessionId);
                        
                        if (affectionLevel !== undefined) {{
                            localStorage.setItem('mari_affection_level', affectionLevel);
                        }}
                        
                        if (relationshipStage !== undefined) {{
                            localStorage.setItem('mari_relationship_stage', relationshipStage);
                        }}
                        
                        localStorage.setItem('mari_last_interaction', new Date().toISOString());
                        console.log('Saved session data to localStorage:', {{ 
                            sessionId, 
                            affectionLevel, 
                            relationshipStage,
                            timestamp: new Date().toISOString()
                        }});
                        return true;
                    }}
                    return false;
                }},
                
                // Clear all session data from localStorage
                clearSessionData: function() {{
                    localStorage.removeItem('mari_session_id');
                    localStorage.removeItem('mari_affection_level');
                    localStorage.removeItem('mari_relationship_stage');
                    localStorage.removeItem('mari_last_interaction');
                    console.log('Cleared all session data from localStorage');
                }},
                
                // Check if session is expired (older than 30 days)
                isSessionExpired: function() {{
                    const lastInteraction = localStorage.getItem('mari_last_interaction');
                    if (!lastInteraction) return true;
                    
                    const lastDate = new Date(lastInteraction);
                    const now = new Date();
                    const daysDiff = (now - lastDate) / (1000 * 60 * 60 * 24);
                    
                    return daysDiff > 30;
                }}
            }};
            
            // Store session ID in localStorage for persistence across page reloads
            window.addEventListener('load', function() {{
                const storedSessionId = localStorage.getItem('mari_session_id');
                
                // Check if we have a stored session and it's not expired
                if (storedSessionId && !window.mariSessionManager.isSessionExpired()) {{
                    console.log('Restored session ID from localStorage:', storedSessionId);
                    
                    // We'll update the session_state component after the page loads
                    setTimeout(() => {{
                        // Find the hidden session state component and update it
                        const sessionStateComponents = document.querySelectorAll('input[data-testid]');
                        for (const component of sessionStateComponents) {{
                            if (component.parentElement.textContent.includes('session_state')) {{
                                component.value = storedSessionId;
                                
                                // Create and dispatch change event to notify Gradio
                                const event = new Event('input', {{ bubbles: true }});
                                component.dispatchEvent(event);
                                
                                console.log('Updated session state component with stored ID:', storedSessionId);
                                
                                // Trigger session restoration
                                window.dispatchEvent(new CustomEvent('mari_restore_session', {{
                                    detail: {{ 
                                        sessionId: storedSessionId,
                                        affectionLevel: localStorage.getItem('mari_affection_level'),
                                        relationshipStage: localStorage.getItem('mari_relationship_stage')
                                    }}
                                }}));
                                break;
                            }}
                        }}
                    }}, 1000);
                }} else if (storedSessionId && window.mariSessionManager.isSessionExpired()) {{
                    // Clear expired session data
                    console.log('Found expired session, clearing data');
                    window.mariSessionManager.clearSessionData();
                }}
            }});
            
            // Periodically update last interaction time while the page is open
            setInterval(function() {{
                const sessionId = localStorage.getItem('mari_session_id');
                if (sessionId) {{
                    localStorage.setItem('mari_last_interaction', new Date().toISOString());
                }}
            }}, 60000); // Update every minute
            </script>
            <link rel="manifest" href="/manifest.json">
    """)

    gr.Markdown("## 🤖 麻理とチャット")
    
    # Enhanced state management
    state = gr.State([])  # Chat history state
    session_state = gr.State(None)  # Session ID state
    relationship_info = gr.State({})  # Store relationship info
    
    with gr.Row():
        with gr.Column(scale=1):
            # Add session info display (hidden by default)
            with gr.Accordion("セッション情報", open=False, visible=True):
                session_id_display = gr.Textbox(label="セッションID", interactive=False)
                affection_level_display = gr.Slider(minimum=0, maximum=100, value=15, 
                                                  label="親密度", interactive=False)
                relationship_stage_display = gr.Textbox(label="関係性ステージ", interactive=False)
            
            chatbot = gr.Chatbot(height=400)
            user_input = gr.Textbox(label="あなたの発言", placeholder="麻理に話しかけよう…", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("送信", variant="primary")
                clear_btn = gr.Button("履歴クリア")

    # Function to update session info display
    def update_session_info(session_id):
        """Update session info display with current affection level and relationship stage"""
        if not session_id or not get_session_manager() or not get_affection_tracker():
            return session_id, 15, "不明", {}
        
        # Get current affection level
        affection_level = get_session_manager().get_affection_level(session_id)
        
        # Get relationship stage
        relationship_stage = get_affection_tracker().get_relationship_stage(affection_level)
        
        # Get relationship info
        relationship_info = get_affection_tracker().get_mari_behavioral_state(affection_level)
        
        # Update session info display
        return session_id, affection_level, relationship_stage, relationship_info
    
    # Modified on_submit to update session info
    def on_submit_with_info(msg, history, session_id, rel_info=None):
        """Enhanced on_submit that also updates session info display"""
        empty_input, updated_chatbot, updated_history, new_session_id, updated_rel_info = on_submit(msg, history, session_id, rel_info)
        
        # Update session info display
        session_id_display, affection_level, relationship_stage, rel_info = update_session_info(new_session_id)
        
        # JavaScript execution removed as gr.JS is not supported in this Gradio version
        
        return empty_input, updated_chatbot, updated_history, new_session_id, session_id_display, affection_level, relationship_stage, rel_info
    
    # Modified clear_history to reset session info
    def clear_history_with_info():
        """Enhanced clear_history that also resets session info display"""
        empty_chatbot, empty_history, empty_session, empty_rel_info = clear_history()
        
        # JavaScript execution removed as gr.JS is not supported in this Gradio version
        
        return empty_chatbot, empty_history, empty_session, "", 15, "不明", {}

    # イベントハンドラ
    user_input.submit(on_submit_with_info, 
                     inputs=[user_input, state, session_state, relationship_info], 
                     outputs=[user_input, chatbot, state, session_state, 
                             session_id_display, affection_level_display, 
                             relationship_stage_display, relationship_info])
    
    submit_btn.click(on_submit_with_info, 
                    inputs=[user_input, state, session_state, relationship_info], 
                    outputs=[user_input, chatbot, state, session_state, 
                            session_id_display, affection_level_display, 
                            relationship_stage_display, relationship_info])
    
    clear_btn.click(clear_history_with_info, 
                   outputs=[chatbot, state, session_state, 
                           session_id_display, affection_level_display, 
                           relationship_stage_display, relationship_info])
    
    # Add event handler to load session on page load
    demo.load(update_session_info, 
             inputs=[session_state], 
             outputs=[session_id_display, affection_level_display, 
                     relationship_stage_display, relationship_info])
                     
    # Function to restore session from localStorage
    def restore_session(session_id):
        """Restore session from localStorage or create new if not exists"""
        if not session_id and get_session_manager():
            # Try to load from localStorage via JavaScript
            return None, [], [], {}
        
        # If we have a session ID, try to load the session
        if session_id and get_session_manager():
            session = get_session_manager().get_session(session_id)
            if session:
                try:
                    # Check if session is expired (older than 30 days)
                    last_interaction = datetime.fromisoformat(session.last_interaction)
                    days_since_interaction = (datetime.now() - last_interaction).days
                    
                    if days_since_interaction > 30:
                        logging.info(f"Session {session_id} expired ({days_since_interaction} days old)")
                        # Create new session instead of using expired one
                        new_session_id = get_session_manager().create_new_session()
                        logging.info(f"Created new session to replace expired one: {new_session_id}")
                        return new_session_id, [], [], {}
                    
                    # Convert conversation history to chatbot format
                    history = []
                    for entry in session.conversation_history:
                        if 'user' in entry and 'assistant' in entry:
                            history.append((entry['user'], entry['assistant']))
                    
                    # Update session info display
                    session_id_val, affection_level, relationship_stage, rel_info = update_session_info(session_id)
                    
                    logging.info(f"Restored session: {session_id} with {len(history)} messages")
                    return session_id, history, history, rel_info
                except (ValueError, TypeError) as e:
                    logging.error(f"Error parsing session data: {str(e)}")
        
        # If session not found or invalid, create new
        new_session_id = get_session_manager().create_new_session() if get_session_manager() else None
        logging.info(f"Created new session during restoration: {new_session_id}")
        return new_session_id, [], [], {}
    
    # Add custom JavaScript event handler for session restoration
    js_code = """
    function(sessionId) {
        // Listen for the custom event from the page load handler
        window.addEventListener('mari_restore_session', function(e) {
            if (e.detail && e.detail.sessionId) {
                // This will trigger the restore_session Python function
                console.log("Restoring session from event:", e.detail.sessionId);
                return e.detail.sessionId;
            }
            return sessionId;
        });
        return sessionId;
    }
    """
    
    # Add event handler for session restoration
    session_state.change(
        fn=restore_session,
        inputs=[session_state],
        outputs=[session_state, chatbot, state, relationship_info],
        js=js_code  # Changed from _js to js
    )
    
    # Add periodic session cleanup (runs once per day)
    def cleanup_old_sessions():
        """Clean up expired sessions (older than 30 days)"""
        if get_session_manager():
            cleaned_count = get_session_manager().cleanup_old_sessions(days_old=30)
            logging.info(f"Cleaned up {cleaned_count} expired sessions")
        return None
    
    # Schedule session cleanup to run once per day
    # Note: This is a simple approach - in production, you might want a more robust scheduler
    # Removed 'every' parameter as it's not supported in this Gradio version
    demo.load(cleanup_old_sessions, inputs=None, outputs=None)



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