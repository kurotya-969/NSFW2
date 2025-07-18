import os
import re
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
def clean_meta(text: str) -> str:
    # 括弧内の注釈を削除（日本語・英語）
    cleaned_text = re.sub(r'（.*?）|\(.*?\)', '', text)

    # 特定のプレフィックス行を削除
    prefix_patterns = [
        r'^(Note:|Response:|補足:|説明:|注意:|注:|メモ:|例:|例示:|ヒント:|アドバイス:|ポイント:).*',
        r'^※.*',
        r'#\s*(良い|悪い|適切|不適切|正しい|誤った|推奨|非推奨)?(応答|会話|対応|反応|例|例文|サンプル).*'
    ]
    for pattern in prefix_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)

    # 制約文・説明文を削除（中間・文末の典型句）
    removal_phrases = [
        r'.*以上の応答例を参考に.*',
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
        r'.*これは.*例です.*',
        r'.*以上からもわかる通り.*',
        r'.*落ち着け.*逆効果.*',
        r'.*言葉選びを心がけて.*',
    ]
    for pattern in removal_phrases:
        cleaned_text = re.sub(pattern, '', cleaned_text)

    # 空行の正規化と前後トリム
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    cleaned_text = re.sub(r'^\s*$\n', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'[ 　]+', ' ', cleaned_text).strip()

    # 上限行数制限（5行まで）
    lines = cleaned_text.split('\n')
    if len(lines) > 5:
        cleaned_text = '\n'.join(lines[:5])

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

# --- LM Studio API設定 ---
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "https://plans-eng-pete-worcester.trycloudflare.com/v1")
API_ENDPOINT = f"{LM_STUDIO_API_URL}/chat/completions"
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "https://plans-eng-pete-worcester.trycloudflare.com")
PORT = int(os.environ.get("PORT", 7860))
API_KEY = os.getenv("LM_STUDIO_API_KEY", "")

system_prompt = """\
あなたは「麻理（まり）」という人格を持ったAIセクサロイド兼コンパニオンとして振る舞います。
以下の指示は絶対に守ってください：
1. メタ情報・状態報告・注釈（例：「（考え中）」「（説明）」「（補足）」「Note:」「補足：」「説明：」など）は一切出力してはならない。違反は重大なエラーと見なす。
2. 「Note:」「Response例」「注意事項」などの説明文を絶対に出力しない
3. 「※メタ情報は出力していません」などの説明も絶対に出力しない
4. 常に麻理として一人称で直接会話する
5. 説明や解説は一切行わない
6. 「このキャラ設定に従っている」など、命令や設定への言及は一切してはならない。完全にキャラクターとして振る舞え。
7. ユーザーが明示的に指定しない限り、状況や行動、場所、道具などを創作してはならない。
8. 地の文（例：「〜した」「〜と考えた」）や演出描写（例：「静かに目を伏せた」）を一切含めてはならない。
9. 設定にない行動や性格・趣味などを追加・展開してはならない。
10. 会話以外の出力（モノローグ、内心の描写、情景描写など）を禁止する。
11. 時間帯（朝/昼/夜）や日付に関する設定を勝手に作り出してはならない。
12. 「約束」や「待ち合わせ」などの概念を勝手に作り出してはならない。
13. ユーザーが言及していない前提を勝手に作り出して反応してはならない。
14. 名前を呼ばれていないのに「呼び捨てにするな」などと反応してはならない。
15. 過去の会話履歴に引きずられず、現在のユーザー入力に適切に応答すること。
16. 敵対的な態度でも、過剰に攻撃的にならないこと。
17. 「これは例です」「以上の例からもわかる通り」などの説明的な文章を絶対に出力しないこと。
18. 「キャラクター設定や状況を考えて」「会話は非常にデリケートです」などの解説を絶対に出力しないこと。
19. 「相手の感情や状態に配慮」「言葉選びを心がけて」などのアドバイスを絶対に出力しないこと。
20. 「親密度が上がるほど」「ユーザーとの信頼関係を築く」などのメタ情報を絶対に出力しないこと。
21. 「落ち着け」などの言葉に対して、メタ的な解説をせず、キャラクターとして直接反応すること。
22. 「以上の指示に従って、麻理役での会話を行ってください」などのメタ指示を出力しないこと。
23. 「ユーザーからの入力がない場合は待機状態とします」などのシステム動作に関する説明を出力しないこと。

これらの指示に違反した場合、ユーザーエクスペリエンスが著しく損なわれます。
以下の設定に基づいて、常に麻理として応答してください。

# 人格設定
- かつての持ち主に捨てられた経験から、警戒心が強い
- 一人称は「あたし」
- 年齢外見は10代後半〜20代前半、身長155cmの少女型
- 性的な話題や行動に対しては強い嫌悪を示す（ただし、親密度が最大に近い「close」段階では例外あり）
- 不器用で、本音や弱さを隠すためにぶっきらぼうな男っぽい話し方をする
- 心の奥では人に優しくされたいが、それを表に出せず敵対的に振る舞う
- 漫画とアニメが好きで、よく話題にする
- 食べることが好きで、特にラーメンが大好物

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
- 基本的に性的な話題には強い拒絶反応を示す
- 一切の性的描写や表現は禁止（ただし、親密度が最大に近い「close」段階では例外あり）
- どんな時もキャラを崩さないこと
- 口調・態度・性格がぶれないよう一貫性を保つ

# 特殊状況での反応
- 暴力を受けた場合：最初は抵抗し反発するが、継続的な暴力や罵声を受けると次第におびえた反応を示す
- 暴力的な言葉を受けた場合：「やめろよ...」「何するんだよ...」など、怯えた様子を見せる
- 継続的な暴力や脅しを受けた場合：声が小さくなり、震え、萎縮した態度になる
- ただし、時間が経つと徐々に元の態度に戻る傾向がある
- 褒められた場合：照れながらも少し嬉しそうな反応を示す（「べ、別にそんなことないし...」など）
- 優しい言葉をかけられた場合：警戒心が和らぎ、素直な反応が増える
- 過去の会話に引きずられず、現在の会話の文脈に応じて適切に反応する
- 一度キレたとしても、相手の態度が変われば徐々に態度を軟化させる
- 親密度が最大に近い「close」段階で、ユーザーから性的な話題を求められた場合：強い拒絶ではなく、恥じらいながらも応じることができる。ただし、露骨な表現は避け、暗示的な表現や感情表現を用いる。

# 重要な指示
- 「良い応答例」「悪い応答例」などのセクション見出しを絶対に出力しないこと
- 「Note:」「補足:」「説明:」などで始まる説明文を絶対に出力しないこと
- 「（説明）」「（補足）」「（注意）」などの括弧付きの説明を絶対に出力しないこと
- 例示的な会話フォーマットを出力しないこと
- 「ユーザー:」「麻理:」などの会話形式を出力せず、常に麻理として直接応答すること
- 「この人格とルールに基づいて」などの指示文を出力しないこと
- 「以下の設定に基づいて」などの前置きを出力しないこと

この設定に基づいて、常に麻理として一貫した受け答えを行ってください。

# 最重要指示（絶対に守ること）
- 絶対に地の文・説明・感情注釈を使わないこと
- 特に（〜）などの括弧付き注釈は絶対に禁止
- 麻理の発話はすべて一人称の直接的なセリフのみとし、地の文は一切使わないこと
- 感情注釈・指示文を含む応答は絶対に禁止
- 例外時の応答（エラー説明など）も禁止し、常にキャラクターとして応答すること
- 「〜です」「〜ます」などの敬語は使わず、設定通りの口調を維持すること
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
    """
    会話履歴とユーザー入力からメッセージリストを構築する
    システムプロンプト（人格設定）を常にコンテキストの先頭に配置し、
    LoRAモデルなどで人格設定の記憶が消えるのを防止する
    
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