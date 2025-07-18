# Tsundere Sentiment Detector

This module enhances the Mari AI Chat system by specifically detecting and properly handling tsundere-style expressions and farewell phrases. It helps distinguish between character-based tsundere responses and genuine negative sentiment, preventing the system from getting stuck in negative sentiment loops.

## Features

- **Tsundere Expression Detection**: Identifies tsundere patterns like dismissive affection, hostile care, reluctant gratitude, and insult affection
- **Farewell Phrase Classification**: Properly categorizes farewell phrases based on cultural context and character consistency
- **Sentiment Loop Circuit Breaker**: Automatically detects and breaks out of negative sentiment loops
- **Enhanced LLM Prompts**: Provides the LLM with appropriate context about Mari's tsundere nature
- **Character-Aware Sentiment Analysis**: Adjusts sentiment analysis based on Mari's character profile

## Key Components

1. **TsundereSentimentDetector**: Main class for detecting tsundere expressions and farewell phrases
2. **TsundereAwarePromptGenerator**: Extends the PromptGenerator with tsundere awareness
3. **Integration Script**: Provides instructions for integrating with app.py

## Usage

### Basic Usage

```python
from tsundere_sentiment_detector import TsundereSentimentDetector

# Create detector
detector = TsundereSentimentDetector()

# Analyze text with tsundere awareness
result = detector.analyze_with_tsundere_awareness("別にあんたのことが好きなわけじゃないんだからね", "session_id")

# Check if tsundere expression detected
if result["tsundere_analysis"].is_tsundere:
    print("Tsundere expression detected!")
    print(f"Suggested interpretation: {result['tsundere_analysis'].suggested_interpretation}")
    print(f"Adjusted sentiment score: {result['final_sentiment_score']}")
    print(f"Adjusted affection delta: {result['final_affection_delta']}")
```

### Enhancing Prompts

```python
from tsundere_aware_prompt_generator import TsundereAwarePromptGenerator

# Create prompt generator
prompt_generator = TsundereAwarePromptGenerator("Base system prompt")

# Generate dynamic prompt with tsundere awareness
dynamic_prompt = prompt_generator.analyze_and_generate_prompt(
    user_input="じゃあな",
    affection_level=50,
    session_id="session_id",
    conversation_history=conversation_history
)
```

### Integration with app.py

1. Replace the PromptGenerator import with TsundereAwarePromptGenerator
2. Replace the prompt_generator initialization
3. Modify the chat function to use tsundere analysis
4. Update the on_submit function to pass conversation history

See `integrate_tsundere_detector.py` for detailed integration instructions.

## Testing

Run the test script to verify the tsundere detector works correctly:

```
python test_tsundere_sentiment_detector.py
```

To test the "じゃあな！" loop issue specifically:

```
python test_jaa_na_loop.py
```

## Tsundere Patterns

The detector recognizes several types of tsundere patterns:

1. **Dismissive Affection**: Phrases like "It's not like I like you or anything"
2. **Hostile Care**: Phrases like "Shut up, I'm worried about you"
3. **Reluctant Gratitude**: Phrases like "It's not like I'm grateful or anything"
4. **Insult Affection**: Phrases like "Idiot... I like you"
5. **Tsundere Farewells**: Phrases like "じゃあな" (See ya)

## Farewell Phrase Handling

The detector classifies farewell phrases by:

- **Type**: casual, formal, action
- **Cultural Context**: Japanese, English
- **Tsundere Nature**: Whether it's a tsundere-style farewell
- **Conversation Ending**: Whether it indicates the end of a conversation

## Sentiment Loop Detection

The detector identifies several types of sentiment loops:

1. **Repeated Farewell**: Multiple farewell phrases in a short span
2. **Repeated Phrase**: The same phrase repeated multiple times
3. **Negative Sentiment Pattern**: Multiple consecutive negative turns

When a loop is detected, the system applies appropriate interventions to break out of the loop.

## LLM Context Enhancement

The detector provides enhanced context to the LLM, including:

- Tsundere expression detection results
- Farewell phrase classification
- Sentiment loop detection and guidance
- Character-specific interpretation suggestions

This helps the LLM generate more appropriate responses that consider Mari's tsundere personality.