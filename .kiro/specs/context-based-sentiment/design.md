# Design Document

## Overview

The Context-Based Sentiment Analysis system enhances Mari's affection system by moving beyond simple keyword matching to analyze the emotional context, intensity, and nuance of user messages. This design builds upon the existing sentiment analysis framework while introducing more sophisticated natural language understanding capabilities to create a more authentic relationship progression experience.

## Architecture

The enhanced sentiment analysis system will integrate with the existing architecture while adding new components for contextual analysis:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Existing        │    │  Enhanced        │    │ Existing        │
│ Affection       │────│  Sentiment       │────│ Session         │
│ Tracker         │    │  Analyzer        │    │ Manager         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        
                                │                        
                       ┌────────▼────────┐      
                       │ Context Analysis│      
                       │    Engine       │      
                       └─────────────────┘      
                                │
                                │
                       ┌────────▼────────┐
                       │ Emotion Intensity│
                       │    Detector     │
                       └─────────────────┘
```

## Components and Interfaces

### 1. Enhanced Sentiment Analyzer (Extends existing SentimentAnalyzer)
**Purpose:** Provides improved sentiment analysis with contextual understanding

**Key Methods:**
- `analyze_user_input(user_input: str, conversation_history: list = None) -> SentimentAnalysisResult`: Enhanced version of existing method that considers context
- `detect_emotional_context(text: str) -> dict`: Identifies emotional context beyond keywords
- `analyze_sentiment_intensity(text: str) -> float`: Determines the strength of emotional expression
- `detect_sarcasm_and_irony(text: str) -> float`: Attempts to identify non-literal language

**Integration Points:**
- Maintains the same interface as the existing SentimentAnalyzer
- Returns the same SentimentAnalysisResult structure for compatibility
- Can be used as a drop-in replacement for the current analyzer

### 2. Context Analysis Engine (New Component)
**Purpose:** Analyzes message context and conversation history to improve sentiment understanding

**Key Methods:**
- `analyze_context(text: str, conversation_history: list = None) -> dict`: Extracts contextual information
- `detect_topic_sentiment(text: str) -> dict`: Identifies sentiment related to specific topics
- `analyze_conversation_flow(history: list) -> dict`: Examines patterns in conversation
- `detect_sentiment_shifts(current: dict, previous: dict) -> float`: Identifies changes in sentiment

**Data Structures:**
```python
{
    "dominant_emotion": str,
    "emotion_confidence": float,
    "contextual_modifiers": list,
    "topic_sentiments": dict,
    "conversation_pattern": str
}
```

### 3. Emotion Intensity Detector (New Component)
**Purpose:** Determines the strength and intensity of emotional expressions

**Key Methods:**
- `detect_intensity(text: str) -> float`: Measures emotional intensity on a scale
- `identify_intensifiers(text: str) -> list`: Finds words that amplify emotions
- `detect_emotional_qualifiers(text: str) -> dict`: Identifies phrases that modify emotion strength
- `calculate_intensity_score(base_score: float, modifiers: dict) -> float`: Adjusts score based on intensity

**Intensity Categories:**
- Mild (0.0-0.3): Slight emotional expression
- Moderate (0.31-0.6): Clear but controlled emotion
- Strong (0.61-0.85): Pronounced emotional expression
- Extreme (0.86-1.0): Intense emotional expression

### 4. Sentiment Context Integrator (New Component)
**Purpose:** Combines keyword-based and contextual analysis for final sentiment determination

**Key Methods:**
- `integrate_analyses(keyword_result: dict, context_result: dict) -> dict`: Merges analysis results
- `resolve_contradictions(analyses: list) -> dict`: Handles conflicting sentiment signals
- `calculate_confidence(analyses: list) -> float`: Determines confidence in final analysis
- `apply_confidence_weighting(result: dict, confidence: float) -> dict`: Adjusts impact based on confidence

## Data Models

### Enhanced SentimentAnalysisResult
```python
@dataclass
class SentimentAnalysisResult:
    sentiment_score: float  # -1.0 to 1.0
    interaction_type: str
    affection_delta: int  # -10 to +10
    confidence: float  # 0.0 to 1.0
    detected_keywords: List[str]
    sentiment_types: List[SentimentType]
    # New fields
    emotional_intensity: float  # 0.0 to 1.0
    contextual_analysis: dict
    conversation_pattern: str
```

### ContextualAnalysis
```python
@dataclass
class ContextualAnalysis:
    dominant_emotion: str
    emotion_confidence: float
    contextual_modifiers: List[str]
    detected_topics: List[str]
    topic_sentiments: Dict[str, float]
    sarcasm_probability: float
    irony_probability: float
```

### ConversationPattern
```python
@dataclass
class ConversationPattern:
    pattern_type: str  # "escalating", "de-escalating", "consistent", "fluctuating"
    duration: int  # Number of turns this pattern has persisted
    intensity_trend: float  # Rate of change in intensity
    sentiment_stability: float  # How consistent the sentiment has been
```

## Implementation Approach

### 1. Text Preprocessing
- Normalize text (lowercase, remove extra whitespace)
- Tokenize into words and sentences
- Apply basic NLP preprocessing (stemming, lemmatization)
- Extract key phrases and emotional expressions

### 2. Multi-layered Analysis
- Layer 1: Traditional keyword-based analysis (existing system)
- Layer 2: Contextual analysis (new)
- Layer 3: Intensity detection (new)
- Layer 4: Conversation pattern recognition (new)

### 3. Confidence Scoring
- Calculate confidence based on:
  - Consistency between layers
  - Clarity of emotional signals
  - Presence of ambiguous language
  - Historical accuracy of predictions

### 4. Affection Impact Calculation
- Base impact from keyword analysis
- Adjusted by contextual understanding
- Scaled by emotional intensity
- Weighted by confidence score
- Smoothed based on conversation history

## Error Handling

### Ambiguity Handling
- When multiple contradictory emotions are detected, use confidence scoring to determine dominant emotion
- When confidence is below threshold, reduce affection impact proportionally
- Log ambiguous cases for future improvement

### Fallback Mechanism
- If contextual analysis fails, fall back to keyword-based analysis
- If both fail, default to neutral sentiment with minimal affection impact
- Implement graceful degradation of functionality

### Edge Cases
- Very short messages: Apply more conservative analysis
- Mixed emotions: Identify dominant emotion and secondary emotions
- Unusual language patterns: Flag for reduced confidence

## Testing Strategy

### Unit Testing
- Test individual components in isolation
- Verify correct identification of emotional context
- Test intensity detection with various expressions
- Validate confidence scoring mechanism

### Integration Testing
- Test complete sentiment analysis pipeline
- Verify compatibility with existing affection system
- Test with conversation history integration
- Validate affection impact calculations

### Scenario Testing
- Test with common conversation patterns
- Verify handling of sarcasm and irony
- Test with ambiguous messages
- Validate behavior with emotional intensity variations

### Regression Testing
- Ensure existing functionality remains intact
- Verify backward compatibility with current sessions
- Test performance impact of enhanced analysis