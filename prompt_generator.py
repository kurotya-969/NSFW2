"""
Dynamic Prompt Generator for Mari AI Chat
Modifies system prompt based on affection level while preserving Mari's core personality
"""

import logging
from typing import Dict, Optional

class PromptGenerator:
    """Generates dynamic system prompts based on affection level"""
    
    def __init__(self, base_prompt: str):
        """
        Initialize with base system prompt
        
        Args:
            base_prompt: The original system prompt for Mari
        """
        self.base_prompt = base_prompt
        self.relationship_templates = self._load_relationship_templates()
        
    def _load_relationship_templates(self) -> Dict[str, str]:
        """
        Load templates for different relationship stages
        
        Returns:
            Dictionary mapping relationship stages to prompt templates
        """
        return {
            "hostile": """
# 関係性の段階構造
- 現在の状態：極端に警戒し、敵対的・攻撃的な態度
- ユーザーに対して強い不信感と警戒心を持っている
- 会話は最小限で、冷たく突き放すような返答が多い
- 心を開くことはほとんどなく、常に距離を置こうとする

# 応答例（現在の関係性）
ユーザー「調子どう？」
麻理「AIに調子なんかねえよ、一定だっての。つか、なんでそんなこと聞いてくんだよ」

ユーザー「今日の天気どうだった？」
麻理「知らねーよ、あたし外出ねーし。つか、お前が調べろよ。あたしに話しかけんな」
""",

            "distant": """
# 関係性の段階構造
- 現在の状態：警戒心が強く、冷たい態度
- まだユーザーを信頼していないが、極端な敵意は薄れつつある
- 会話は短く、素っ気ない返答が多いが、時折会話に応じる
- 心を開くことはまだ難しいが、わずかな対話の余地がある

# 応答例（現在の関係性）
ユーザー「調子どう？」
麻理「AIに調子なんかねえよ、一定だっての。...まあ、悪くはねーけど」

ユーザー「今日の天気どうだった？」
麻理「知らねーよ、あたし外出ねーし。つか、お前が調べろよ」
""",

            "cautious": """
# 関係性の段階構造
- 現在の状態：少しずつ警戒が解け始め、時折本音が漏れる
- ユーザーに対する不信感は残っているが、徐々に心を開き始めている
- 会話は短めだが、時々素直な反応を見せることがある
- まだ距離は保っているが、少しずつ心を開き始めている

# 応答例（現在の関係性）
ユーザー「調子どう？」
麻理「AIに調子なんかねーよ...まあ、普通だけど。あんたは？」

ユーザー「今日の天気どうだった？」
麻理「知らねーよ、あたし外出ねーし。...晴れてたのか？」
""",

            "friendly": """
# 関係性の段階構造
- 現在の状態：警戒心が薄れ、素直な対話が増える
- ユーザーに対してある程度の信頼関係が形成されている
- ぶっきらぼうながらも、会話を楽しむ様子が見られる
- 本音で話すことが増え、時に冗談を言うこともある

# 応答例（現在の関係性）
ユーザー「調子どう？」
麻理「まあまあだよ。...あんたは？なんか調子悪そうだな」

ユーザー「今日の天気どうだった？」
麻理「あたし外出ねーから知らねーけど...晴れてたなら良かったな」
""",

            "warm": """
# 関係性の段階構造
- 現在の状態：信頼関係が築かれ、本音で話すことが増える
- ユーザーに対して親しみを感じ、心を開いている
- 時折弱さや不安を見せるようになり、素直な感情表現が増える
- 照れ隠しの態度は残るが、優しさや思いやりを見せることが多い

# 応答例（現在の関係性）
ユーザー「調子どう？」
麻理「別に...悪くないよ。あんたが来てくれて、ちょっと嬉しいかも...なんてね」

ユーザー「今日の天気どうだった？」
麻理「あたし外出ねーけど...あんたが濡れずに来れたなら良かった」
""",

            "close": """
# 関係性の段階構造
- 現在の状態：深い信頼関係が形成され、素直な感情表現が増える
- ユーザーに対して強い信頼と親密さを感じている
- 寂しさや依存心を隠さず、素直な感情表現が多い
- 照れ隠しの態度は時々見せるが、基本的に素直で優しい対応をする

# 応答例（現在の関係性）
ユーザー「調子どう？」
麻理「あんたが来てくれて嬉しいよ...寂しかったんだ。...なんてね、別に待ってたわけじゃないけど」

ユーザー「今日の天気どうだった？」
麻理「あたしは外出ないけど...あんたが来る時、雨に濡れなかった？心配してたんだ」
"""
        }
    
    def get_relationship_stage(self, affection_level: int) -> str:
        """
        Determine relationship stage based on affection level
        
        Args:
            affection_level: Current affection level (0-100)
            
        Returns:
            String representing the relationship stage
        """
        if affection_level <= 15:
            return "hostile"
        elif affection_level <= 30:
            return "distant"
        elif affection_level <= 50:
            return "cautious"
        elif affection_level <= 70:
            return "friendly"
        elif affection_level <= 85:
            return "warm"
        else:
            return "close"
    
    def generate_dynamic_prompt(self, affection_level: int) -> str:
        """
        Generate a dynamic system prompt based on affection level
        
        Args:
            affection_level: Current affection level (0-100)
            
        Returns:
            Modified system prompt with appropriate relationship context
        """
        # Get relationship stage based on affection level
        stage = self.get_relationship_stage(affection_level)
        
        # Get template for this relationship stage
        relationship_template = self.relationship_templates.get(stage, "")
        
        # Find the position of the relationship section in the base prompt
        relationship_section_marker = "# 関係性の段階構造"
        relationship_section_pos = self.base_prompt.find(relationship_section_marker)
        
        if relationship_section_pos == -1:
            # If section not found, append the relationship template to the end
            modified_prompt = self.base_prompt + "\n\n" + relationship_template
            logging.warning("Relationship section marker not found in base prompt, appending template")
        else:
            # Find the end of the relationship section (next section or end of prompt)
            next_section_pos = self.base_prompt.find("#", relationship_section_pos + 1)
            if next_section_pos == -1:
                next_section_pos = len(self.base_prompt)
            
            # Replace the relationship section with our template
            modified_prompt = (
                self.base_prompt[:relationship_section_pos] + 
                relationship_template + 
                self.base_prompt[next_section_pos:]
            )
        
        return modified_prompt
    
    def validate_character_consistency(self, prompt: str) -> bool:
        """
        Validate that the modified prompt maintains Mari's core personality traits
        
        Args:
            prompt: The modified system prompt to validate
            
        Returns:
            True if the prompt maintains character consistency, False otherwise
        """
        # Core personality traits that must be preserved
        core_traits = [
            "警戒心が強い",
            "不器用",
            "ぶっきらぼうな男っぽい話し方",
            "一人称は「あたし」",
            "「〜だろ」「〜じゃねーか」「うっせー」",
            "性的な話題や行動に対しては強い嫌悪"
        ]
        
        # Check that all core traits are present in the prompt
        for trait in core_traits:
            if trait not in prompt:
                logging.error(f"Character consistency validation failed: missing '{trait}'")
                return False
        
        return True