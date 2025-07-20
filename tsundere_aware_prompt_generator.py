"""
Tsundere-Aware Prompt Generator for Mari AI Chat
Enhances prompts with tsundere awareness and sentiment loop guidance
"""

import logging
from typing import Dict, Any, Optional
from prompt_generator import PromptGenerator
from tsundere_sentiment_detector import TsundereSentimentDetector

class TsundereAwarePromptGenerator(PromptGenerator):
    """Extends the PromptGenerator with tsundere awareness"""
    
    def __init__(self, base_prompt: str):
        """
        Initialize the tsundere-aware prompt generator
        
        Args:
            base_prompt: The base system prompt
        """
        super().__init__(base_prompt)
        self.tsundere_detector = TsundereSentimentDetector()
    
    def generate_dynamic_prompt(self, affection_level: int, tsundere_context: Optional[Dict[str, Any]] = None, 
                               user_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a dynamic prompt based on affection level, tsundere context, and user metadata
        
        Args:
            affection_level: Current affection level (0-100)
            tsundere_context: Optional tsundere analysis context
            user_metadata: Optional user metadata for personalization
            
        Returns:
            Dynamic prompt with affection, tsundere awareness, and personalization
        """
        # First, generate the affection-based dynamic prompt using parent method
        affection_prompt = super().generate_dynamic_prompt(affection_level)
        
        # Enhance with tsundere awareness if context is provided
        if tsundere_context:
            affection_prompt = self.tsundere_detector.get_enhanced_prompt(affection_prompt, tsundere_context)
        
        # Enhance with user metadata if provided
        if user_metadata:
            affection_prompt = self._enhance_with_user_metadata(affection_prompt, user_metadata, affection_level)
        
        return affection_prompt
        
    def _enhance_with_user_metadata(self, prompt: str, user_metadata: Dict[str, Any], affection_level: int) -> str:
        """
        Enhance prompt with user metadata
        
        Args:
            prompt: Base prompt to enhance
            user_metadata: User metadata for personalization
            affection_level: Current affection level for context
            
        Returns:
            Enhanced prompt with user metadata
        """
        # Create a user context section
        user_context = "# ユーザーについての情報\n"
        
        # Add nickname if available
        if user_metadata.get("nickname"):
            user_context += f"- ユーザーの名前: {user_metadata['nickname']}\n"
        
        # Add birthday if available
        if user_metadata.get("birthday"):
            user_context += f"- ユーザーの誕生日: {user_metadata['birthday']}\n"
        
        # Add age if available
        if user_metadata.get("age"):
            user_context += f"- ユーザーの年齢: {user_metadata['age']}歳\n"
        
        # Add likes if available
        if user_metadata.get("likes") and len(user_metadata["likes"]) > 0:
            user_context += "- ユーザーの好きなもの:\n"
            for like in user_metadata["likes"][:3]:  # 最大3つまで
                user_context += f"  - {like['item']} (カテゴリ: {like['category']})\n"
        
        # Add dislikes if available
        if user_metadata.get("dislikes") and len(user_metadata["dislikes"]) > 0:
            user_context += "- ユーザーの嫌いなもの:\n"
            for dislike in user_metadata["dislikes"][:3]:  # 最大3つまで
                user_context += f"  - {dislike['item']} (カテゴリ: {dislike['category']})\n"
        
        # Add location if available
        if user_metadata.get("location"):
            user_context += f"- ユーザーの住んでいる場所: {user_metadata['location']}\n"
        
        # Add occupation if available
        if user_metadata.get("occupation"):
            user_context += f"- ユーザーの職業: {user_metadata['occupation']}\n"
        
        # Add guidance on how to use this information based on relationship stage
        stage = self.get_relationship_stage(affection_level)
        
        if stage in ["hostile", "distant"]:
            user_context += "\n上記の情報は覚えていますが、まだ警戒心が強いため、積極的には使わないでください。\n"
        elif stage in ["cautious"]:
            user_context += "\n上記の情報を時々、さりげなく会話に取り入れてください。例: 「そういえば、あんた[好きなもの]が好きだったよな」\n"
        elif stage in ["friendly", "warm"]:
            user_context += "\n上記の情報を自然に会話に取り入れてください。ユーザーの好みや情報を覚えていることを示してください。\n"
        elif stage in ["close"]:
            user_context += "\n上記の情報を積極的に会話に取り入れ、親密な関係性を示してください。ユーザーの好みや情報を大切にしていることを表現してください。\n"
        
        # Add the user context section to the prompt
        # Find a good position to insert - after the main character description but before specific instructions
        insert_marker = "# 話し方の特徴"
        insert_pos = prompt.find(insert_marker)
        
        if insert_pos == -1:
            # If marker not found, just append to the end
            return prompt + "\n\n" + user_context
        
        # Insert the user context before the marker
        return prompt[:insert_pos] + user_context + "\n\n" + prompt[insert_pos:]
    
    def analyze_and_generate_prompt(self, user_input: str, affection_level: int, 
                                  session_id: Optional[str] = None,
                                  conversation_history: Optional[list] = None) -> str:
        """
        Analyze user input and generate an appropriate prompt
        
        Args:
            user_input: The user's message
            affection_level: Current affection level (0-100)
            session_id: Optional session ID for loop detection
            conversation_history: Optional conversation history
            
        Returns:
            Dynamic prompt with tsundere awareness
        """
        # Analyze the user input with tsundere awareness
        analysis_result = self.tsundere_detector.analyze_with_tsundere_awareness(
            user_input, session_id, conversation_history
        )
        
        # Extract tsundere context from analysis
        tsundere_context = analysis_result.get("llm_context", {})
        
        # Generate dynamic prompt with tsundere awareness
        dynamic_prompt = self.generate_dynamic_prompt(affection_level, tsundere_context)
        
        # Log the prompt generation
        logging.debug(f"Generated tsundere-aware prompt for affection level {affection_level}")
        if tsundere_context.get("tsundere_detected"):
            logging.info(f"Tsundere context applied to prompt: {tsundere_context.get('suggested_interpretation')}")
        if tsundere_context.get("sentiment_loop_detected"):
            logging.info(f"Sentiment loop guidance applied to prompt: {tsundere_context.get('suggested_intervention')}")
        
        return dynamic_prompt