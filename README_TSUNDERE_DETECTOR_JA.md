# ツンデレ感情検出器

このモジュールは麻理AIチャットシステムを強化し、ツンデレ表現や別れの言葉を適切に検出・処理します。キャラクターのツンデレ反応と本当のネガティブ感情を区別し、システムがネガティブな感情ループに陥るのを防ぎます。

## 機能

- **ツンデレ表現の検出**: 「好きなわけじゃない」などのツンデレパターンを識別
- **別れの言葉の分類**: 「じゃあな」などの別れの言葉を文化的文脈とキャラクター一貫性に基づいて分類
- **感情ループ防止**: ネガティブな感情ループを自動的に検出し解消
- **LLMプロンプト強化**: 麻理のツンデレ性格に関する適切な文脈をLLMに提供
- **キャラクター考慮型感情分析**: 麻理のキャラクタープロファイルに基づいて感情分析を調整

## 主要コンポーネント

1. **TsundereSentimentDetector**: ツンデレ表現と別れの言葉を検出するメインクラス
2. **TsundereAwarePromptGenerator**: ツンデレ認識でPromptGeneratorを拡張
3. **統合スクリプト**: app.pyとの統合手順を提供

## 使用方法

### 基本的な使い方

```python
from tsundere_sentiment_detector import TsundereSentimentDetector

# 検出器を作成
detector = TsundereSentimentDetector()

# ツンデレ認識でテキストを分析
result = detector.analyze_with_tsundere_awareness("別にあんたのことが好きなわけじゃないんだからね", "session_id")

# ツンデレ表現が検出されたかチェック
if result["tsundere_analysis"].is_tsundere:
    print("ツンデレ表現を検出しました")
    print(f"解釈提案: {result['tsundere_analysis'].suggested_interpretation}")
    print(f"調整後の感情スコア: {result['final_sentiment_score']}")
    print(f"調整後の親密度変化: {result['final_affection_delta']}")
```

### プロンプト強化

```python
from tsundere_aware_prompt_generator import TsundereAwarePromptGenerator

# プロンプト生成器を作成
prompt_generator = TsundereAwarePromptGenerator("基本システムプロンプト")

# ツンデレ認識で動的プロンプトを生成
dynamic_prompt = prompt_generator.analyze_and_generate_prompt(
    user_input="じゃあな",
    affection_level=50,
    session_id="session_id",
    conversation_history=conversation_history
)
```

### app.pyとの統合

1. PromptGeneratorをTsundereAwarePromptGeneratorに置き換える
2. prompt_generatorの初期化を更新する
3. chat関数をツンデレ分析を使用するように修正する
4. on_submit関数を会話履歴を渡すように更新する

詳細な統合手順については`integrate_tsundere_detector.py`を参照してください。

## ツンデレパターン

検出器は以下のようなツンデレパターンを認識します：

1. **否定的な愛情表現**: 「好きなわけじゃない」などのフレーズ
2. **敵対的な気遣い**: 「うるさい、心配してるんだよ」などのフレーズ
3. **不本意な感謝**: 「別にありがとうとか思ってないからね」などのフレーズ
4. **侮辱を含む愛情表現**: 「バカ...好きだよ」などのフレーズ
5. **ツンデレ別れの言葉**: 「じゃあな」などのフレーズ

## 別れの言葉の処理

検出器は別れの言葉を以下の基準で分類します：

- **タイプ**: カジュアル、フォーマル、行動
- **文化的文脈**: 日本語、英語
- **ツンデレ性**: ツンデレスタイルの別れの言葉かどうか
- **会話終了**: 会話の終了を示すかどうか

## 感情ループ検出

検出器は以下のような感情ループを識別します：

1. **繰り返される別れ**: 短時間に複数の別れの言葉
2. **繰り返されるフレーズ**: 同じフレーズの複数回の繰り返し
3. **ネガティブな感情パターン**: 連続するネガティブなターン

ループが検出されると、システムは適切な介入を行ってループから抜け出します。

## LLMコンテキスト強化

検出器はLLMに以下のような強化されたコンテキストを提供します：

- ツンデレ表現検出結果
- 別れの言葉の分類
- 感情ループ検出とガイダンス
- キャラクター固有の解釈提案

これにより、LLMは麻理のツンデレ性格を考慮したより適切な応答を生成できるようになります。