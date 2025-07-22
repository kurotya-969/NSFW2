"""
修正用のスクリプト：SentimentAnalyzerの問題を修正します
"""

import re

# ファイルを読み込む
with open('sentiment_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# _determine_interaction_typeメソッドを修正
# 特定のキーワードが検出された場合、スコアに関わらず対応するinteraction_typeを返すように変更
old_determine_interaction_type = r'''    def _determine_interaction_type\(self, sentiment_types: List\[SentimentType\], sentiment_score: float\) -> str:
        """
        Determine the overall interaction type based on sentiment analysis
        
        Args:
            sentiment_types: List of detected sentiment types
            sentiment_score: Overall sentiment score
            
        Returns:
            String describing the interaction type
        """
        if SentimentType.SEXUAL in sentiment_types:
            return "sexual"  # 性的な内容が検出された場合、最優先で「sexual」タイプとする
        elif SentimentType.HOSTILE in sentiment_types:
            return "hostile"
        elif SentimentType.APPRECIATIVE in sentiment_types:
            return "appreciative"
        elif SentimentType.CARING in sentiment_types:
            return "caring"
        elif SentimentType.DISMISSIVE in sentiment_types:
            return "dismissive"
        elif sentiment_score > 0.3:
            return "positive"
        elif sentiment_score < -0.3:
            return "negative"
        else:
            return "neutral"'''

new_determine_interaction_type = r'''    def _determine_interaction_type(self, sentiment_types: List[SentimentType], sentiment_score: float) -> str:
        """
        Determine the overall interaction type based on sentiment analysis
        
        Args:
            sentiment_types: List of detected sentiment types
            sentiment_score: Overall sentiment score
            
        Returns:
            String describing the interaction type
        """
        if SentimentType.SEXUAL in sentiment_types:
            return "sexual"  # 性的な内容が検出された場合、最優先で「sexual」タイプとする
        elif SentimentType.HOSTILE in sentiment_types:
            return "hostile"
        elif SentimentType.APPRECIATIVE in sentiment_types:
            return "appreciative"
        elif SentimentType.CARING in sentiment_types:
            return "caring"
        elif SentimentType.DISMISSIVE in sentiment_types:
            return "dismissive"
        elif SentimentType.POSITIVE in sentiment_types:
            # ポジティブなキーワードが検出された場合、スコアに関わらずpositiveを返す
            return "positive"
        elif SentimentType.NEGATIVE in sentiment_types:
            # ネガティブなキーワードが検出された場合、スコアに関わらずnegativeを返す
            return "negative"
        elif sentiment_score > 0.3:
            return "positive"
        elif sentiment_score < -0.3:
            return "negative"
        else:
            return "neutral"'''

# 正規表現を使用して置換
modified_content = re.sub(old_determine_interaction_type, new_determine_interaction_type, content)

# 修正したコードを書き込む
with open('sentiment_analyzer.py', 'w', encoding='utf-8') as f:
    f.write(modified_content)

print("SentimentAnalyzerの修正が完了しました。")