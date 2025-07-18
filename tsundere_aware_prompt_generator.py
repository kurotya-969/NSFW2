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
    
    def generate_dynamic_prompt(self, affection_level: int, tsundere_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a dynamic prompt based on affection level and tsundere context
        
        Args:
            affection_level: Current affection level (0-100)
            tsundere_context: Optional tsundere analysis context
            
        Returns:
            Dynamic prompt with affection and tsundere awareness
        """
        # First, generate the affection-based dynamic prompt using parent method
        affection_prompt = super().generate_dynamic_prompt(affection_level)
        
        # If no tsundere context, return the affection-based prompt
        if not tsundere_context:
            return affection_prompt
        
        # Enhance the prompt with tsundere awareness
        enhanced_prompt = self.tsundere_detector.get_enhanced_prompt(affection_prompt, tsundere_context)
        
        return enhanced_prompt
    
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