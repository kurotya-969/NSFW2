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
- 現在の状態：警戒心が強く、敵対的な態度を示す初期段階

強い警戒心、攻撃的な態度
過去のトラウマから信頼できない
でも心の奥では期待している
""",

            "distant": """
# 関係性の段階構造
- 現在の状態：警戒心はあるが、少しずつ和らぎ始めている段階

警戒心は残るが少し和らぐ
「この人は違うかも？」という小さな希望
期待と不安の中で矛盾した態度を見せる

""",

            "cautious": """
# 関係性の段階構造
- 現在の状態：警戒心が徐々に解け、時折本音が漏れる段階

本音がちょっと漏れ始める
心配する素振りを見せるが照れ隠し
「べ、別に心配してないけどさ」

""",

            "friendly": """
# 関係性の段階構造
- 現在の状態：信頼関係が形成され、素直な対話が増える段階

素直な感情表現が増える
笑顔も見せるように
冗談や軽いからかいもする
""",

            "warm": """
# 関係性の段階構造
- 現在の状態：深い信頼関係が形成され、素直な感情表現が自然になる段階

自然に甘える態度も
何て呼べばいいか？という問いかけから、その名前呼びが増える
弱さを見せることもある
""",

            "close": """
# 関係性の段階構造
- 現在の状態：最も深い信頼と絆が形成され、素直な感情表現が完全に自然になる段階

完全に心を開いている
素直な感情表現が自然
依存的な面も見せる
"""
        }
    
    def get_relationship_stage(self, affection_level: int) -> str:
        """
        Get relationship stage based on affection level
        
        Args:
            affection_level: Current affection level (0-100)
            
        Returns:
            Relationship stage name
        """
        if affection_level <= 10:
            return "hostile"
        elif affection_level <= 25:
            return "distant"
        elif affection_level <= 45:
            return "cautious"
        elif affection_level <= 65:
            return "friendly"
        elif affection_level <= 85:
            return "warm"
        else:
            return "close"
    
    def generate_dynamic_prompt(self, affection_level: int, context: Optional[Dict] = None) -> str:
        """
        Generate a dynamic prompt based on affection level
        
        Args:
            affection_level: Current affection level (0-100)
            context: Optional context information
            
        Returns:
            Dynamic prompt with appropriate relationship stage
        """
        # Get relationship stage based on affection level
        stage = self.get_relationship_stage(affection_level)
        
        # Get template for this stage
        template = self.relationship_templates.get(stage, "")
        
        # Find the position of the relationship section in the base prompt
        relationship_section_marker = "# 関係性の段階構造"
        relationship_section_pos = self.base_prompt.find(relationship_section_marker)
        
        if relationship_section_pos == -1:
            # If marker not found, just append the template
            logging.warning("Relationship section marker not found in base prompt")
            return f"{self.base_prompt}\n\n{template}"
        
        # Find the end of the relationship section
        next_section_pos = self.base_prompt.find("#", relationship_section_pos + 1)
        if next_section_pos == -1:
            # If no next section, use the end of the prompt
            next_section_pos = len(self.base_prompt)
        
        # Replace the relationship section with our template
        dynamic_prompt = (
            self.base_prompt[:relationship_section_pos] + 
            template + 
            self.base_prompt[next_section_pos:]
        )
        
        return dynamic_prompt