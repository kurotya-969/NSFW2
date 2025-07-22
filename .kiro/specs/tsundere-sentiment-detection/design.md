# Design Document

## Overview

The Tsundere Sentiment Detection system enhances Mari's affection system by specifically identifying and properly handling tsundere-style expressions and farewell phrases. This design builds upon the existing context-based sentiment analysis framework while adding specialized components to recognize character-specific expressions that should not be interpreted as genuine negative sentiment.

## Architecture

The enhanced system will integrate with the existing architecture while adding new components for tsundere detection:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Context-Based   │    │  Tsundere        │    │ Existing        │
│ Sentiment       │────│  Sentiment       │────│ Affection       │
│ Analyzer        │    │  Detector        │    │ Tracker         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        
                                │                        
                       ┌────────▼────────┐      
                       │ Character Profile│      
                       │    Analyzer     │      
                       └─────────────────┘      
                                │
                                │
                       ┌────────▼────────┐
                       │ Sentiment Loop  │
                       │ Circuit Breaker │
                       └─────────────────┘
```

## Components and Interfaces

### 1. Tsundere Sentiment Detector (New Component)
**Purpose:** Identifies tsundere expressions and distinguishes them from genuine negative sentiment

**Key Methods:**
- `detect_tsundere_expressions(text: str) -> TsundereAnalysisResult`: Analyzes text for tsundere patterns
- `classify_farewell_phrases(text: str) -> FarewellAnalysisResult`: Identifies and classifies farewell expressions
- `get_tsundere_confidence(text: str, character_profile: dict) -> float`: Determines confidence in tsundere detection
- `adjust_sentiment_for_tsundere(sentiment_result: ContextualSentimentResult) -> ContextualSentimentResult`: Modifies sentiment analysis based on tsundere detection

**Integration Points:**
- Integrates between the Context-Based Sentiment Analyzer and Affection Tracker
- Receives input from the Character Profile Analyzer
- Provides adjusted sentiment results to the Affection Tracker

### 2. Character Profile Analyzer (New Component)
**Purpose:** Analyzes text against Mari's character profile to identify character-consistent expressions

**Key Methods:**
- `match_character_speech_patterns(text: str, character_profile: dict) -> List[str]`: Identifies matches with character's speech patterns
- `get_character_consistency_score(text: str, character_profile: dict) -> float`: Determines how consistent text is with character profile
- `identify_character_specific_phrases(text: str, character_profile: dict) -> Dict[str, float]`: Maps phrases to character-specific meanings
- `get_character_context_for_llm(sentiment_result: Any, tsundere_result: Any) -> dict`: Generates character context for LLM prompts

**Data Structures:**
```python
{
    "speech_patterns": List[str],
    "character_phrases": Dict[str, Dict],
    "tsundere_indicators": Dict[str, float],
    "farewell_phrases": Dict[str, str],
    "character_traits": Dict[str, float]
}
```

### 3. Sentiment Loop Circuit Breaker (New Component)
**Purpose:** Detects and breaks negative sentiment loops

**Key Methods:**
- `detect_sentiment_loop(conversation_history: List[dict]) -> bool`: Identifies potential sentiment loops
- `get_loop_severity(conversation_history: List[dict]) -> float`: Determines how severe the loop is
- `apply_circuit_breaker(sentiment_result: Any, loop_data: dict) -> Any`: Modifies sentiment to break out of loops
- `get_recovery_recommendations(loop_data: dict) -> dict`: Provides recommendations for recovering from loops

**Loop Detection Criteria:**
- Repeated similar negative phrases
- Consistent negative sentiment across multiple turns
- Decreasing affection levels over multiple interactions
- Presence of farewell phrases followed by continued conversation

### 4. Enhanced Prompt Generator (Extends existing PromptGenerator)
**Purpose:** Incorporates tsundere context into LLM prompts

**Key Methods:**
- `add_tsundere_context(prompt: str, tsundere_data: dict) -> str`: Enhances prompts with tsundere context
- `generate_character_aware_prompt(affection_level: int, tsundere_context: dict) -> str`: Creates prompts with character awareness
- `add_sentiment_loop_guidance(prompt: str, loop_data: dict) -> str`: Adds guidance for breaking sentiment loops
- `get_farewell_handling_instructions(farewell_data: dict) -> str`: Provides instructions for handling farewells

## Data Models

### TsundereAnalysisResult
```python
@dataclass
class TsundereAnalysisResult:
    is_tsundere: bool
    tsundere_confidence: float
    detected_patterns: List[str]
    character_consistency: float
    suggested_interpretation: str
    affection_adjustment: int
    sentiment_adjustment: float
```

### FarewellAnalysisResult
```python
@dataclass
class FarewellAnalysisResult:
    is_farewell: bool
    farewell_type: str  # "casual", "formal", "tsundere", "genuine", etc.
    cultural_context: str
    suggested_interpretation: str
    is_conversation_ending: bool
    confidence: float
```

### CharacterProfileMatch
```python
@dataclass
class CharacterProfileMatch:
    matched_patterns: List[str]
    character_consistency: float
    speech_pattern_matches: Dict[str, float]
    personality_trait_relevance: Dict[str, float]
    suggested_interpretation: str
```

### SentimentLoopData
```python
@dataclass
class SentimentLoopData:
    loop_detected: bool
    loop_severity: float
    repeated_patterns: List[str]
    loop_duration: int  # Number of turns
    suggested_intervention: str
    affection_recovery_suggestion: int
```

## Implementation Approach

### 1. Tsundere Expression Detection
- Create a comprehensive database of tsundere expressions in Japanese and English
- Implement pattern matching for common tsundere phrases
- Develop contextual analysis to distinguish tsundere from genuine negativity
- Create confidence scoring based on character profile consistency

### 2. Farewell Phrase Classification
- Build a database of farewell phrases in Japanese and English
- Categorize farewells by formality, cultural context, and emotional tone
- Implement detection for conversation-ending vs. casual farewells
- Create special handling for tsundere-style farewells like "じゃあな！"

### 3. Character Profile Integration
- Extract Mari's core character traits and speech patterns
- Create a mapping between speech patterns and emotional meanings
- Implement consistency checking between expressions and character profile
- Develop character-aware sentiment adjustment

### 4. Sentiment Loop Prevention
- Implement detection for repeated negative patterns
- Create automatic affection recovery for detected loops
- Develop gradual sentiment normalization for prolonged negative interactions
- Implement topic change detection for sentiment recovery

### 5. LLM Context Enhancement
- Extend prompt generation to include tsundere awareness
- Add character profile context to guide response generation
- Implement sentiment loop guidance in prompts
- Create farewell handling instructions for the LLM

## Character-Specific Considerations

### Mari's Tsundere Characteristics
- Uses rough language as a defense mechanism
- Expresses affection through seemingly dismissive phrases
- Uses farewell phrases that may seem hostile but are character-consistent
- Has established speech patterns that appear negative but aren't genuinely hostile

### Common Tsundere Patterns to Detect
- Apparent dismissal that masks interest
- Rough language combined with helpful actions
- Insults followed by caring behaviors
- Farewells that seem abrupt but are character-consistent
- Statements that contradict emotional subtext

## Error Handling

### Misclassification Handling
- Implement confidence thresholds for tsundere detection
- Create fallback to standard sentiment analysis for low-confidence cases
- Log potential misclassifications for review
- Implement gradual sentiment adjustment to avoid dramatic swings

### Sentiment Loop Recovery
- Detect when system is stuck in negative sentiment loops
- Implement automatic circuit breaker after N turns of negative loops
- Create gradual affection recovery mechanisms
- Provide LLM with context about potential loops

### Edge Cases
- Very short messages with ambiguous sentiment
- Cultural expressions that might be misinterpreted
- Mixed tsundere and genuine negative sentiment
- Sarcasm that might be confused with tsundere expressions

## Testing Strategy

### Unit Testing
- Test individual tsundere pattern detection
- Verify farewell phrase classification
- Test character profile matching
- Validate sentiment loop detection

### Integration Testing
- Test complete sentiment analysis pipeline with tsundere detection
- Verify compatibility with existing affection system
- Test with conversation history integration
- Validate LLM prompt enhancement

### Scenario Testing
- Test with common tsundere conversation patterns
- Verify handling of the "じゃあな！" loop issue
- Test with mixed genuine and tsundere sentiment
- Validate behavior with various farewell phrases

### Regression Testing
- Ensure existing functionality remains intact
- Verify backward compatibility with current sessions
- Test performance impact of enhanced analysis