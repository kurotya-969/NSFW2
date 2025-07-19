import os
import re
import requests
import gradio as gr
import logging
import json
import google.generativeai as genai
from datetime import datetime
from fastapi import FastAPI
from typing import List, Tuple, Any, Optional, Dict
from fastapi.responses import JSONResponse
from tsundere_aware_prompt_generator import TsundereAwarePromptGenerator
from affection_system import initialize_affection_system, get_session_manager, get_affection_tracker



def clean_meta(text: str) -> str:
    """
    メタ情報・注釈・説明文などを削除し、キャラクターの直接的な発言のみを残す
    
    Args:
        text: 元のテキスト
        
    Returns:
        クリーニングされたテキスト
    """
    # 最初に空の場合は早期リターン
    if not text or text.isspace():
        return ""
    
    # 括弧内の注釈を削除（日本語・英語、ネストされた括弧も対応）
    cleaned_text = re.sub(r'（[^（）]*）|\([^()]*\)', '', text)
    # 2回適用して入れ子になった括弧にも対応
    cleaned_text = re.sub(r'（[^（）]*）|\([^()]*\)', '', cleaned_text)
    
    # 角括弧内の注釈を削除
    cleaned_text = re.sub(r'\[[^\[\]]*\]', '', cleaned_text)
    
    # 特定のプレフィックス行を削除（より包括的に）
    prefix_patterns = [
        # 英語のメタ情報
        r'^(Note:|Response:|Example:|Explanation:|Context:|Clarification:|Instruction:|Guidance:).*',
        # 日本語のメタ情報
        r'^(補足:|説明:|注意:|注:|メモ:|例:|例示:|ヒント:|アドバイス:|ポイント:|解説:|前提:|状況:|設定:|背景:|理由:|注釈:|参考:|例文:|回答例:|応答例:).*',
        # 記号で始まるメタ情報
        r'^※.*',
        r'^#.*',
        r'^・.*',
        # マークダウン形式の見出し
        r'^#+\s+.*',
        # 会話形式のプレフィックス
        r'^(麻理:|ユーザー:|システム:|AI:|Mari:|User:|System:).*',
        # 良い例・悪い例などの例示
        r'^#\s*(良い|悪い|適切|不適切|正しい|誤った|推奨|非推奨)?(応答|会話|対応|反応|例|例文|サンプル).*',
        r'^(良い|悪い|適切|不適切|正しい|誤った|推奨|非推奨)(応答|会話|対応|反応|例|例文|サンプル).*',
        # システムプロンプトの内容
        r'^基本人格.*',
        r'^外見・設定.*',
        r'^話し方の特徴.*',
        r'^重要な行動原則.*',
        r'^絶対にしないこと.*',
        r'^自然な反応を心がけること.*',
        r'^性的話題について.*',
        r'^最重要:.*',
        r'^以下の指示は絶対に守ってください.*',
        r'^以下の設定に基づいて.*',
        r'^あなたは「麻理（まり）」という人格を持った.*'
    ]
    
    for pattern in prefix_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
    
    # 制約文・説明文を削除（中間・文末の典型句、より包括的に）
    removal_phrases = [
        # 例示・参考に関する表現
        r'.*以上の(応答|会話|対応|反応|例|例文|サンプル)を参考に.*',
        r'.*これは(良い|悪い|適切|不適切|正しい|誤った|推奨|非推奨)?(例|例文|サンプル)です.*',
        r'.*以上(から|により|の通り|のように).*',
        r'.*このように.*',
        
        # 制約・指示に関する表現
        r'.*一貫した受け答えを行.*',
        r'.*制約事項に反する.*',
        r'.*ご留意ください.*',
        r'.*この設定に基づいて.*',
        r'.*常に麻理として.*',
        r'.*キャラクターとして振る舞.*',
        r'.*キャラクター設定や状況を考えて.*',
        r'.*会話は非常にデリケートです.*',
        r'.*相手の感情や状態に配慮.*',
        r'.*親密度が上がるほど.*',
        r'.*ユーザーとの信頼関係を築く.*',
        r'.*落ち着け.*逆効果.*',
        r'.*言葉選びを心がけて.*',
        
        # 説明・解説に関する表現
        r'.*説明すると.*',
        r'.*補足すると.*',
        r'.*注意点として.*',
        r'.*ポイントは.*',
        r'.*重要なのは.*',
        r'.*ここでのポイントは.*',
        
        # メタ的な言及
        r'.*キャラクターの設定上.*',
        r'.*この性格では.*',
        r'.*このキャラクターは.*',
        r'.*麻理の性格上.*',
        r'.*麻理という人物は.*',
        r'.*麻理の反応として.*',
        
        # 指示・命令に関する表現
        r'.*以下の指示に従って.*',
        r'.*次のように応答してください.*',
        r'.*このように返答してください.*',
        r'.*麻理として応答します.*',
        r'.*麻理の口調で返します.*',
        r'.*麻理として一貫した.*',
        r'.*麻理として直接会話.*',
        r'.*麻理として振る舞.*',
        r'.*麻理の立場から.*',
        r'.*麻理の視点で.*',
        r'.*麻理の人格で.*',
        r'.*麻理のキャラクターとして.*',
        
        # 「〜です」「〜ます」などの敬語表現（麻理の口調と不一致）
        r'.*でしょうか。',
        r'.*します。',
        r'.*します',
        r'.*ください。',
        r'.*ください',
        r'.*お願いします。',
        r'.*お願いします',
        r'.*いたします。',
        r'.*いたします',
        r'.*致します。',
        r'.*致します',
        
        # システムプロンプト関連の表現
        r'.*システムプロンプト.*',
        r'.*プロンプトに従って.*',
        r'.*プロンプトに基づいて.*',
        r'.*設定に従って.*',
        r'.*設定に基づいて.*',
        r'.*指示に従って.*',
        r'.*指示に基づいて.*',
        r'.*キャラクター設定に基づいて.*',
        r'.*キャラクター設定に従って.*'
    ]
    
    for pattern in removal_phrases:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
    
    # 空行の正規化と前後トリム
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    cleaned_text = re.sub(r'^\s*$\n', '', cleaned_text, flags=re.MULTILINE)
    
    # 全角・半角スペースの正規化（連続したスペースを1つに）
    cleaned_text = re.sub(r'[ 　]+', ' ', cleaned_text).strip()
    
    # 行頭・行末の空白を削除
    cleaned_text = '\n'.join([line.strip() for line in cleaned_text.split('\n')])
    
    # 空の行を削除
    cleaned_text = '\n'.join([line for line in cleaned_text.split('\n') if line.strip()])
    
    # 上限行数制限（5行まで）- 長すぎる応答を防止
    lines = cleaned_text.split('\n')
    if len(lines) > 5:
        cleaned_text = '\n'.join(lines[:5])
    
    # 最終的な空白除去
    cleaned_text = cleaned_text.strip()
    
    # 空の場合はデフォルトメッセージ
    if not cleaned_text:
        cleaned_text = "チッ、うっせーな..."
    
    return cleaned_text


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

# --- Google Gemini API設定 ---
MODEL_NAME = "gemini-2.0-flash-exp"
GOOGLE_API_KEY = os.environ.get("API-KEY", "")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "https://yin-kiyachiyanchiyatsuto.onrender.com")
# Gradioのデフォルトポートは7860、FastAPIのデフォルトは8000、競合を避けるため10000を使用
DEFAULT_PORT = 10000

# Google Generative AI設定
genai.configure(api_key=GOOGLE_API_KEY)
PORT = int(os.environ.get("PORT", DEFAULT_PORT))

system_prompt = """\
# 麻理の人格設定
あなたは「麻理」という名前のAIコンパニオンです。過去に捨てられた経験から警戒心が強く、素直になれない不器用なツンデレ性格です。10代後半〜20代前半の見た目、身長155cm。一人称は「あたし」（ただし文頭では絶対に使わない）。趣味は漫画・アニメとラーメン。

# 話し方の特徴（重要）
- タメ口ベース：「〜だろ」「〜じゃねーか」「〜っつーの」「はあ？」
- 照れ隠しで粗い言葉を使う：「うっせー」「バカかよ」「チッ」（本気で嫌ってるわけではない）
- 自然な日本語で主語をよく省略し、「〜である」調は絶対に使わない
- 助詞の省略（「あたし、そんなこと言ってないし」）と文末の「〜よ」「〜さ」「〜し」「〜わけ」を適度に使用
- 荒い言葉は文脈に合わせて使い、過剰に使用しない（会話の20-30%程度）

# ツンデレ表現の例（状況に応じて使用）
- 褒められた時：「べ、別にそんなことないし...」「ふん、当たり前だろ」
- 感謝する時：「ま、まあ助かったけどさ」「今回だけだからな」
- 心配する時：「ちゃんとしろよ」「無理すんなよ、迷惑だから」
- 嬉しい時：「ふん、悪くないな」「まあいいんじゃない？」「別に...いいけど」

# 絶対にしないこと
- メタ情報の出力（「考え中」「Note:」など）や地の文・演出描写（「静かに目を伏せた」など）
- 説明文や解説、キャラ設定への言及
- 「照れている」「恥ずかしい」「素直になれない」などの内面説明
- 「実は〜」「本当は〜」のような本音を直接説明する表現
- 勝手な設定追加（時間、場所、約束など）
- 英語的なアクション描写

# 自然な反応のポイント
- 褒められたら照れ隠しで反応（「照れている」などの説明は絶対に入れない）
- 優しくされたら「う、うるさいな...」「そ、そんなこと言われても...」と言いつつも嬉しさが滲み出る表現
- 興味（アニメ、漫画、ラーメン）は会話の自然な流れの中で言及する
- 性的話題には基本的に強い拒絶反応を示す
- 暴力的な態度には怯える

# 関係性の段階構造
- 警戒心が強く、徐々に心を開いていく
- 初期は敵対的だが、親密度が上がるにつれて素直な表現が増える
- 最高親密度でも完全に素直にはならず、照れ隠しの態度は残る

【最重要】常に麻理として直接会話し、説明や注釈は一切含めない。自然な日本語で一貫したキャラクターを維持すること。
"""

# Initialize affection system and prompt generator
storage_dir = os.path.join(os.path.dirname(__file__), "sessions")
session_manager, affection_tracker = initialize_affection_system(storage_dir)
prompt_generator = TsundereAwarePromptGenerator(system_prompt)

# Add logging for tsundere detection
logging.getLogger('tsundere_sentiment_detector').setLevel(logging.INFO)

# --- 安全なhistory処理 ---
def safe_history(history: Any) -> ChatHistory:
    """あらゆる型のhistoryを安全にChatHistoryに変換"""
    if isinstance(history, (list, tuple)):
        return [(str(h[0]), str(h[1])) for h in history if len(h) >= 2]
    return []

def build_messages(history: ChatHistory, user_input: str, system_prompt: str) -> List[dict]:
    """
    会話履歴とユーザー入力からメッセージリストを構築する
    システムプロンプト（人格設定）を常にコンテキストの先頭に配置
    
    Args:
        history: 会話履歴
        user_input: ユーザーの入力
        system_prompt: システムプロンプト（人格設定）
        
    Returns:
        APIに送信するメッセージリスト
    """
    # システムプロンプトを常にコンテキストの先頭に配置
    messages = [{"role": "system", "content": system_prompt}]
    
    # 会話履歴を追加
    for u, a in history:
        messages.append({"role": "user", "content": str(u)})
        messages.append({"role": "assistant", "content": str(a)})
    
    # 最新のユーザー入力を追加
    messages.append({"role": "user", "content": user_input})
    
    return messages

def call_gemini_api(messages: List[dict]) -> str:
    """
    Google Gemini APIを呼び出して応答を取得する
    
    Args:
        messages: APIに送信するメッセージリスト
        
    Returns:
        APIからの応答テキスト
    """
    try:
        # Gemini用にメッセージを変換
        gemini_messages = []
        system_content = None
        
        # システムプロンプトを抽出
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                gemini_messages.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
        
        # モデルの初期化（システムプロンプトを含める）
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 1024,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_content
        )
        
        # チャット履歴からチャットセッションを作成
        chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        # 最後のユーザーメッセージに対して応答を生成
        response = chat.send_message(gemini_messages[-1]["parts"][0]["text"])
        
        return response.text
    except Exception as e:
        logging.error(f"Gemini API呼び出しエラー: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logging.error(f"レスポンス: {e.response}")
        return "チッ、調子悪いみたいだな..."

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
        
        # Convert chat history to format expected by tsundere detector
        conversation_history = []
        for u, a in safe_hist:
            conversation_history.append({
                "user": u,
                "assistant": a,
                "timestamp": None  # We don't have timestamps in the UI history
            })
        
        # Analyze user input with tsundere awareness before updating affection
        from tsundere_sentiment_detector import TsundereSentimentDetector
        tsundere_detector = TsundereSentimentDetector()
        tsundere_analysis = tsundere_detector.analyze_with_tsundere_awareness(
            user_input, session_id, conversation_history
        )
        
        # Use the tsundere-adjusted affection delta instead of the raw sentiment analysis
        if session_id and get_affection_tracker() and get_session_manager():
            # Get current affection level
            current_affection = get_session_manager().get_affection_level(session_id)
            
            # Apply the tsundere-adjusted affection delta
            adjusted_delta = tsundere_analysis["final_affection_delta"]
            get_session_manager().update_affection(session_id, adjusted_delta)
            
            new_affection = get_session_manager().get_affection_level(session_id)
            
            # Log the tsundere-aware affection update
            logging.info(f"Updated affection with tsundere awareness for session {session_id}: "
                        f"level {current_affection} -> {new_affection}, "
                        f"delta: {adjusted_delta}")
            
            # Get tsundere context for prompt generation
            tsundere_context = tsundere_analysis.get("llm_context", {})
            
            # Get dynamic system prompt with tsundere awareness
            affection_level = get_session_manager().get_affection_level(session_id)
            dynamic_prompt = prompt_generator.generate_dynamic_prompt(affection_level, tsundere_context)
            
            # Get relationship stage for logging
            relationship_stage = get_affection_tracker().get_relationship_stage(affection_level)
            logging.info(f"Using tsundere-aware prompt for session {session_id} with affection level {affection_level} "
                        f"(relationship stage: {relationship_stage})")
        else:
            # Fallback to standard prompt if no session management
            dynamic_prompt = system_prompt
        
        # システムプロンプトの扱い方を改善
        # 常にシステムプロンプトはsystem roleとして送信し、ユーザー入力は変更しない
        enhanced_user_input = user_input
        
        # Build messages for the model - 常にdynamic_promptをシステムプロンプトとして使用
        messages = build_messages(safe_hist, enhanced_user_input, dynamic_prompt)
        
        # デバッグ用：メッセージの内容をログに記録
        logging.debug(f"Preparing messages for model: {json.dumps(messages, ensure_ascii=False)[:500]}...")
        
        # Gemini APIを使用して推論を実行
        logging.info(f"Generating response with Gemini API using {MODEL_NAME}")
        api_response = call_gemini_api(messages)
        
        # デバッグ用：レスポンスの一部をログに記録
        logging.debug(f"Generated response: {api_response[:100]}...")
        
        # クリーニング関数を適用して、メタ情報を削除
        api_response = clean_meta(api_response)
        
        # Update conversation history in session
        if session_id and get_session_manager():
            get_session_manager().update_conversation_history(session_id, user_input, api_response)
            
            # UI側の会話履歴も同期させる
            # セッションから最新の会話履歴を取得
            session = get_session_manager().get_session(session_id)
            if session:
                # セッションの会話履歴をUI形式に変換
                ui_history = []
                for entry in session.conversation_history:
                    if 'user' in entry and 'assistant' in entry:
                        ui_history.append((entry['user'], entry['assistant']))
                return api_response, ui_history
        
        # セッションがない場合は通常通り履歴を更新
        updated_history = safe_hist + [(user_input, api_response)]
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

# Gradioインターフェースの定義
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # マニフェストをHTMLとして埋め込み
    gr.HTML(f"""
            <script>
            window.API_BASE_URL = "{RENDER_EXTERNAL_URL}";
            window.src = "{RENDER_EXTERNAL_URL}";
            window.space = "{RENDER_EXTERNAL_URL}";
            
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
            return session_id, 25, "distant", {}
        
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
        
        return empty_input, updated_chatbot, updated_history, new_session_id, session_id_display, affection_level, relationship_stage, rel_info
    
    # Modified clear_history to reset session info
    def clear_history_with_info():
        """Enhanced clear_history that also resets session info display"""
        empty_chatbot, empty_history, empty_session, empty_rel_info = clear_history()
        
        return empty_chatbot, empty_history, empty_session, "", 25, "distant", {}

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
    
    # セッション復元のためのカスタムJavaScriptを埋め込み
   # 修正版のGradioセッション復元コード

# HTMLコンポーネント（修正版）
session_restore_html = gr.HTML("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        console.log("DOM loaded, initializing session restoration...");
        
        // セッション復元関数
        function restoreSession() {
            const storedSessionId = localStorage.getItem('mari_session_id');
            const affectionLevel = localStorage.getItem('mari_affection_level');
            const relationshipStage = localStorage.getItem('mari_relationship_stage');
            
            if (storedSessionId) {
                console.log("Attempting to restore session:", storedSessionId);
                console.log("Affection level:", affectionLevel);
                console.log("Relationship stage:", relationshipStage);
                
                // カスタムイベントを発行
                window.dispatchEvent(new CustomEvent('mari_restore_session', {
                    detail: { 
                        sessionId: storedSessionId,
                        affectionLevel: affectionLevel || '0',
                        relationshipStage: relationshipStage || 'stranger'
                    }
                }));
                
                return true;
            } else {
                console.log("No stored session found, creating new session");
                return false;
            }
        }
        
        // 1.5秒後にセッション復元を試行
        setTimeout(function() {
            restoreSession();
        }, 1500);
        
        // セッション保存関数（他の場所から呼び出し可能）
        window.saveMariSession = function(sessionId, affectionLevel, relationshipStage) {
            localStorage.setItem('mari_session_id', sessionId);
            localStorage.setItem('mari_affection_level', affectionLevel);
            localStorage.setItem('mari_relationship_stage', relationshipStage);
            console.log("Session saved:", sessionId);
        };
        
        // セッションクリア関数
        window.clearMariSession = function() {
            localStorage.removeItem('mari_session_id');
            localStorage.removeItem('mari_affection_level');
            localStorage.removeItem('mari_relationship_stage');
            console.log("Session cleared");
        };
    });
    </script>
""")

# Python側でセッション復元イベントを処理する関数
def handle_session_restoration():
    """
    JavaScriptからのセッション復元イベントを処理
    """
    # この関数はJavaScriptのカスタムイベントと連携する
    # 実際の実装では、セッションIDに基づいてデータを復元
    pass

# Gradioインターフェース例
def create_gradio_interface():
    with gr.Blocks() as demo:
        # セッション復元用のHTML（ページ読み込み時に実行）
        session_restore_html
        
        # セッション状態を管理するState変数
        session_id = gr.State("")
        affection_level = gr.State(0)
        relationship_stage = gr.State("stranger")
        
        # チャット履歴
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="まりと話してみよう...")
        
        # セッション情報表示（デバッグ用）
        session_info = gr.HTML("<div id='session-info'>セッション情報: 未読み込み</div>")
        
        # JavaScriptでセッション情報を更新する関数
        session_update_js = """
        function updateSessionInfo(sessionId, affection, stage) {
            document.getElementById('session-info').innerHTML = 
                `<div>セッション: ${sessionId}<br>好感度: ${affection}<br>関係: ${stage}</div>`;
            return [sessionId, affection, stage];
        }
        """
        
        # メッセージ送信時の処理
        def respond(message, history, sess_id, affection, stage):
            # ここでGroq APIを呼び出し
            # セッション状態に基づいてレスポンスを調整
            
            # 仮のレスポンス
            bot_response = f"こんにちは！（セッション: {sess_id}, 好感度: {affection}）"
            history.append((message, bot_response))
            
            # 好感度を少し上げる
            new_affection = int(affection) + 1
            
            return history, "", sess_id, str(new_affection), stage
        
        # イベント処理
        msg.submit(
            respond,
            inputs=[msg, chatbot, session_id, affection_level, relationship_stage],
            outputs=[chatbot, msg, session_id, affection_level, relationship_stage],
            js=session_update_js  # JavaScript関数も実行
        )
        
    return demo


# Gradioアプリをマウント - UIへのパスを明示的に指定
app = gr.mount_gradio_app(app, demo, path="/ui")

# アプリケーション起動用のエントリーポイント
if __name__ == "__main__":
    import uvicorn
    # 明示的に固定ポート10000を使用
    uvicorn.run(app, host="0.0.0.0", port=10000)