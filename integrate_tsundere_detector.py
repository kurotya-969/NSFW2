"""
Integration script for the Tsundere Sentiment Detector
This script shows how to integrate the tsundere detector with the existing app.py
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from tsundere_sentiment_detector import TsundereSentimentDetector
from tsundere_aware_prompt_generator import TsundereAwarePromptGenerator
from affection_system import get_session_manager, get_affection_tracker

# Type alias for chat history
ChatHistory = List[Tuple[str, str]]

def integrate_tsundere_detector():
    """
    Instructions for integrating the tsundere detector with app.py
    
    This function provides code snippets and instructions for integrating
    the tsundere detector with the existing app.py file.
    """
    print("=== Tsundere Sentiment Detector Integration Guide ===")
    print("\n1. Replace the PromptGenerator import with TsundereAwarePromptGenerator:")
    print("   FROM: from prompt_generator import PromptGenerator")
    print("   TO:   from tsundere_aware_prompt_generator import TsundereAwarePromptGenerator")
    
    print("\n2. Replace the prompt_generator initialization:")
    print("   FROM: prompt_generator = PromptGenerator(system_prompt)")
    print("   TO:   prompt_generator = TsundereAwarePromptGenerator(system_prompt)")
    
    print("\n3. Modify the chat function to use tsundere analysis:")
    print("""
def chat(user_input: str, system_prompt: str, history: Any = None, session_id: Optional[str] = None) -> Tuple[str, ChatHistory]:
    \"\"\"
    Enhanced chat function with tsundere awareness
    
    Args:
        user_input: The user's message
        system_prompt: Base system prompt
        history: Chat history
        session_id: User session ID for affection tracking
        
    Returns:
        Tuple of (assistant_response, updated_history)
    \"\"\"
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
        
        # Rest of the chat function remains the same...
        # Build messages and make API call
        messages = build_messages(safe_hist, user_input, dynamic_prompt)
        
        # Continue with the existing code...
    """)
    
    print("\n4. Update the on_submit function to pass conversation history:")
    print("""
def on_submit(msg: str, history: ChatHistory, session_id: str = None, relationship_info: dict = None):
    \"\"\"
    Enhanced handle user message submission with tsundere awareness
    
    Args:
        msg: User message
        history: Chat history
        session_id: User session ID for affection tracking
        relationship_info: Current relationship information
        
    Returns:
        Tuple of (empty_input, updated_chatbot, updated_history, session_id, relationship_info)
    \"\"\"
    # Check for stored session ID in browser localStorage or create a new one
    if not session_id and get_session_manager():
        # First try to create a new session
        session_id = get_session_manager().create_new_session()
        logging.info(f"Created new session: {session_id}")
    
    # Convert history to conversation history format for tsundere analysis
    conversation_history = []
    if history:
        for u, a in history:
            conversation_history.append({
                "user": u,
                "assistant": a,
                "timestamp": None  # We don't have timestamps in the UI history
            })
    
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
    """)
    
    print("\n5. Add logging for tsundere detection:")
    print("""
# Add to the logging configuration
logging.getLogger('tsundere_sentiment_detector').setLevel(logging.INFO)
    """)
    
    print("\nFollow these steps to integrate the tsundere detector with your app.py file.")
    print("This will enable the system to properly handle tsundere expressions and farewell phrases.")

if __name__ == "__main__":
    integrate_tsundere_detector()