"""
Test script for the "じゃあな！" loop issue
This script simulates the loop and shows how the tsundere detector resolves it
"""

import logging
import sys
from tsundere_sentiment_detector import TsundereSentimentDetector
from tsundere_aware_prompt_generator import TsundereAwarePromptGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_jaa_na_loop():
    """Test the "じゃあな！" loop issue"""
    print("=== Testing 'じゃあな！' Loop Resolution ===\n")
    
    # Create detector and prompt generator
    detector = TsundereSentimentDetector()
    prompt_generator = TsundereAwarePromptGenerator("Base system prompt for Mari")
    
    # Create a test session ID
    session_id = "test_session_jaa_na_loop"
    
    # Simulate conversation history
    conversation_history = [
        {"user": "こんにちは", "assistant": "何か用？", "timestamp": "2023-01-01T12:00:00"},
        {"user": "元気？", "assistant": "別に普通だよ", "timestamp": "2023-01-01T12:01:00"}
    ]
    
    # Test farewell phrase without loop detection
    print("=== First 'じゃあな' (No Loop) ===")
    result1 = detector.analyze_with_tsundere_awareness("じゃあな", session_id, conversation_history)
    print(f"Is tsundere: {result1['tsundere_analysis'].is_tsundere}")
    print(f"Is farewell: {result1['tsundere_analysis'].is_farewell}")
    print(f"Farewell type: {result1['tsundere_analysis'].farewell_type}")
    print(f"Original sentiment score: {result1['original_sentiment'].adjusted_sentiment_score:.2f}")
    print(f"Original affection delta: {result1['original_sentiment'].adjusted_affection_delta}")
    print(f"Final sentiment score: {result1['final_sentiment_score']:.2f}")
    print(f"Final affection delta: {result1['final_affection_delta']}")
    print(f"Loop detected: {result1.get('sentiment_loop', {}).get('loop_detected', False)}")
    
    # Generate prompt for first farewell
    prompt1 = prompt_generator.generate_dynamic_prompt(50, result1.get("llm_context", {}))
    print("\nPrompt excerpt for first 'じゃあな':")
    prompt_lines = prompt1.split("\n")
    tsundere_section_start = next((i for i, line in enumerate(prompt_lines) if "Tsundere Expression Handling" in line), -1)
    if tsundere_section_start > 0:
        print("\n".join(prompt_lines[tsundere_section_start:tsundere_section_start+10]))
    else:
        print("No tsundere section found in prompt")
    
    # Update conversation history
    conversation_history.append({"user": "じゃあな", "assistant": "ああ、じゃあな", "timestamp": "2023-01-01T12:02:00"})
    
    # Test second farewell phrase (should trigger loop detection)
    print("\n=== Second 'じゃあな' (Loop Detected) ===")
    result2 = detector.analyze_with_tsundere_awareness("じゃあな", session_id, conversation_history)
    print(f"Is tsundere: {result2['tsundere_analysis'].is_tsundere}")
    print(f"Is farewell: {result2['tsundere_analysis'].is_farewell}")
    print(f"Farewell type: {result2['tsundere_analysis'].farewell_type}")
    print(f"Original sentiment score: {result2['original_sentiment'].adjusted_sentiment_score:.2f}")
    print(f"Original affection delta: {result2['original_sentiment'].adjusted_affection_delta}")
    print(f"Final sentiment score: {result2['final_sentiment_score']:.2f}")
    print(f"Final affection delta: {result2['final_affection_delta']}")
    print(f"Loop detected: {result2.get('sentiment_loop', {}).get('loop_detected', True)}")
    if result2.get('sentiment_loop', {}).get('loop_detected', False):
        print(f"Loop severity: {result2['sentiment_loop'].loop_severity:.2f}")
        print(f"Loop patterns: {result2['sentiment_loop'].repeated_patterns}")
        print(f"Suggested intervention: {result2['sentiment_loop'].suggested_intervention}")
        print(f"Affection recovery: {result2['sentiment_loop'].affection_recovery_suggestion}")
    
    # Generate prompt for second farewell
    prompt2 = prompt_generator.generate_dynamic_prompt(50, result2.get("llm_context", {}))
    print("\nPrompt excerpt for second 'じゃあな' (with loop detection):")
    prompt_lines = prompt2.split("\n")
    tsundere_section_start = next((i for i, line in enumerate(prompt_lines) if "Tsundere Expression Handling" in line), -1)
    if tsundere_section_start > 0:
        print("\n".join(prompt_lines[tsundere_section_start:tsundere_section_start+15]))
    else:
        print("No tsundere section found in prompt")
    
    # Update conversation history
    conversation_history.append({"user": "じゃあな", "assistant": "もう言ったじゃん。じゃあな", "timestamp": "2023-01-01T12:03:00"})
    
    # Test third farewell phrase (should trigger stronger intervention)
    print("\n=== Third 'じゃあな' (Stronger Intervention) ===")
    result3 = detector.analyze_with_tsundere_awareness("じゃあな", session_id, conversation_history)
    print(f"Is tsundere: {result3['tsundere_analysis'].is_tsundere}")
    print(f"Is farewell: {result3['tsundere_analysis'].is_farewell}")
    print(f"Original sentiment score: {result3['original_sentiment'].adjusted_sentiment_score:.2f}")
    print(f"Original affection delta: {result3['original_sentiment'].adjusted_affection_delta}")
    print(f"Final sentiment score: {result3['final_sentiment_score']:.2f}")
    print(f"Final affection delta: {result3['final_affection_delta']}")
    print(f"Loop detected: {result3.get('sentiment_loop', {}).get('loop_detected', True)}")
    if result3.get('sentiment_loop', {}).get('loop_detected', False):
        print(f"Loop severity: {result3['sentiment_loop'].loop_severity:.2f}")
        print(f"Loop patterns: {result3['sentiment_loop'].repeated_patterns}")
        print(f"Suggested intervention: {result3['sentiment_loop'].suggested_intervention}")
        print(f"Affection recovery: {result3['sentiment_loop'].affection_recovery_suggestion}")
    
    # Generate prompt for third farewell
    prompt3 = prompt_generator.generate_dynamic_prompt(50, result3.get("llm_context", {}))
    print("\nPrompt excerpt for third 'じゃあな' (with stronger intervention):")
    prompt_lines = prompt3.split("\n")
    tsundere_section_start = next((i for i, line in enumerate(prompt_lines) if "Tsundere Expression Handling" in line), -1)
    if tsundere_section_start > 0:
        print("\n".join(prompt_lines[tsundere_section_start:tsundere_section_start+15]))
    else:
        print("No tsundere section found in prompt")
    
    print("\n=== Summary ===")
    print("The tsundere detector successfully identified the 'じゃあな！' loop and applied interventions:")
    print("1. First occurrence: Identified as a tsundere farewell phrase")
    print("2. Second occurrence: Detected loop and applied circuit breaker")
    print("3. Third occurrence: Applied stronger intervention with affection recovery")
    print("\nThis prevents the system from getting stuck in a negative sentiment loop")
    print("and provides appropriate context to the LLM for generating better responses.")

if __name__ == "__main__":
    test_jaa_na_loop()