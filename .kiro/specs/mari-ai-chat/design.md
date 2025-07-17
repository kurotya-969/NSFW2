# Design Document

## Overview

The Mari AI Chat system is designed as a character-driven conversational AI that maintains consistent personality traits while dynamically adjusting relationship dynamics based on user interactions. The system uses an affection parameter system to track relationship progression and influence response generation while preserving Mari's core character identity.

## Architecture

The system builds upon the existing FastAPI/Gradio architecture while adding the affection system and enhanced personality management:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Gradio Interface│────│  Enhanced Chat   │────│ LM Studio API   │
│   (Existing)    │    │   Controller     │    │   (Existing)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
                       ┌────────▼────────┐      ┌────────▼────────┐
                       │ Affection System │      │ Enhanced System │
                       │    (New)         │      │ Prompt (Modified)│
                       └─────────────────┘      └─────────────────┘
                                │
                                │
                       ┌────────▼────────┐
                       │ Session Storage │
                       │    (New)        │
                       └─────────────────┘

Existing Components to Maintain:
- FastAPI app with /manifest.json endpoint
- Gradio UI with chatbot interface
- LM Studio API integration
- Logging system
- PWA manifest functionality
```

## Components and Interfaces

### 1. Enhanced Chat Controller (Modifies existing `chat` function)
**Purpose:** Extends the existing chat function to integrate affection system

**Integration Points:**
- Replaces the existing `chat()` function in app.py
- Maintains compatibility with existing Gradio interface
- Preserves LM Studio API integration
- Adds affection tracking and system prompt modification

**Key Methods:**
- `enhanced_chat(user_input: str, system_prompt: str, history: Any, session_id: str) -> Tuple[str, ChatHistory]`: Enhanced version of existing chat function
- `get_dynamic_system_prompt(base_prompt: str, affection_level: int) -> str`: Modifies system prompt based on affection
- `analyze_and_update_affection(user_input: str, session_id: str) -> None`: Updates affection based on user input

### 2. Enhanced System Prompt Manager (Modifies existing system_prompt)
**Purpose:** Dynamically adjusts Mari's relationship behavior while maintaining her core personality

**Integration with Existing Character:**
- Preserves Mari's core traits: 警戒心が強い, 不器用, ぶっきらぼうな男っぽい話し方
- Maintains existing speech patterns: 「〜だろ」「〜じゃねーか」「うっせー」
- Keeps personality constraints: 性的な話題への強い拒絶反応

**Key Methods:**
- `get_affection_modified_prompt(base_prompt: str, affection_level: int) -> str`: Modifies existing system prompt
- `get_relationship_context(affection_level: int) -> str`: Adds relationship-specific context
- `validate_character_consistency(prompt: str) -> bool`: Ensures Mari's core traits remain intact

**Affection-Based Modifications:**
- Low (0-30): 極端に警戒し、敵対的・攻撃的な態度 (existing initial state)
- Medium (31-70): 少しずつ棘が抜けてくる (existing middle state)  
- High (71-100): 本音や不安、寂しさなどを漏らす (existing later state)

### 3. Affection System
**Purpose:** Manages relationship dynamics and affection parameter tracking

**Key Methods:**
- `analyze_user_input(message: str) -> int`: Determines affection impact of user input
- `update_affection(delta: int) -> None`: Modifies current affection level
- `get_relationship_stage() -> str`: Returns current relationship category
- `get_affection_modifiers() -> dict`: Provides response tone adjustments

**Affection Levels:**
- Low (0-30): Distant, cautious, formal
- Medium (31-70): Friendly, comfortable, casual
- High (71-100): Warm, close, affectionate

### 4. Response Engine
**Purpose:** Generates contextually appropriate responses combining personality and relationship dynamics

**Key Methods:**
- `generate_response(user_input: str, personality: dict, affection_data: dict) -> str`: Creates Mari's response
- `apply_relationship_tone(base_response: str, affection_level: int) -> str`: Adjusts response tone
- `ensure_natural_flow(response: str, conversation_history: list) -> str`: Maintains conversation coherence

### 5. Persistence Layer
**Purpose:** Handles data storage and retrieval for session continuity

**Key Methods:**
- `save_session_data(affection_level: int, conversation_history: list) -> None`: Persists session state
- `load_session_data() -> dict`: Retrieves previous session data
- `initialize_new_user() -> dict`: Sets up data for first-time users

## Data Models

### User Session
```python
{
    "user_id": str,
    "affection_level": int,  # 0-100
    "conversation_history": list,
    "session_start_time": datetime,
    "last_interaction": datetime
}
```

### Mari Response Context
```python
{
    "base_personality": dict,
    "current_affection": int,
    "relationship_stage": str,
    "tone_modifiers": dict,
    "conversation_context": list
}
```

### Affection Analysis Result
```python
{
    "sentiment_score": float,
    "interaction_type": str,
    "affection_delta": int,
    "confidence": float
}
```

## Error Handling

### Input Validation
- Sanitize user input to prevent injection attacks
- Handle empty or malformed messages gracefully
- Validate affection parameter bounds (0-100)

### System Resilience
- Fallback responses when AI generation fails
- Default personality traits if core data is corrupted
- Session recovery mechanisms for interrupted conversations

### Data Persistence Errors
- Graceful degradation when storage is unavailable
- Backup mechanisms for critical session data
- Error logging for debugging and monitoring

## Testing Strategy

### Unit Testing
- Test individual component methods in isolation
- Verify affection parameter calculations
- Validate personality consistency checks
- Test data persistence operations

### Integration Testing
- Test complete conversation flows
- Verify affection level transitions
- Test session continuity across restarts
- Validate response quality and consistency

### Character Testing
- Verify Mari's personality remains consistent across affection levels
- Test relationship progression feels natural
- Validate speech patterns and vocabulary consistency
- Test edge cases in affection parameter changes

### User Experience Testing
- Test conversation flow and naturalness
- Verify relationship progression feels meaningful
- Test system behavior with various input types
- Validate session persistence works correctly