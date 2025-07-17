"""
Sentiment Fallback Handler for Mari AI Chat
Provides graceful fallback mechanisms for advanced analysis failures
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from sentiment_analyzer import SentimentAnalyzer, SentimentType, SentimentAnalysisResult

@dataclass
class FallbackResult:
    """Result of a fallback operation"""
    success: bool
    result: Optional[SentimentAnalysisResult]
    fallback_level: int  # 0: No fallback, 1: Partial fallback, 2: Complete fallback
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    fallback_strategy: Optional[str] = None

class SentimentFallbackHandler:
    """
    Handles graceful fallbacks for advanced sentiment analysis failures
    Implements progressive fallback strategies from most to least sophisticated
    """
    
    def __init__(self):
        """Initialize the fallback handler"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fallback_stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "fallback_level_counts": {0: 0, 1: 0, 2: 0, 3: 0},
            "error_type_counts": {}
        }
    
    def handle_analysis_error(self, text: str, error: Exception, 
                             partial_result: Optional[Dict] = None,
                             conversation_history: Optional[List[Dict]] = None) -> FallbackResult:
        """
        Handle an error in advanced sentiment analysis
        
        Args:
            text: The original text to analyze
            error: The exception that occurred
            partial_result: Optional partial result from the failed analysis
            conversation_history: Optional conversation history
            
        Returns:
            FallbackResult with fallback analysis
        """
        self.fallback_stats["total_attempts"] += 1
        
        # Get error details
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error type statistics
        if error_type not in self.fallback_stats["error_type_counts"]:
            self.fallback_stats["error_type_counts"][error_type] = 0
        self.fallback_stats["error_type_counts"][error_type] += 1
        
        # Log the error
        logging.error(f"Sentiment analysis error ({error_type}): {error_message}")
        logging.debug(f"Error details: {traceback.format_exc()}")
        
        # Try progressive fallback strategies
        
        # Strategy 1: Try to use partial results if available
        if partial_result:
            try:
                result = self._fallback_with_partial_result(text, partial_result)
                self.fallback_stats["successful_attempts"] += 1
                self.fallback_stats["fallback_level_counts"][1] += 1
                return FallbackResult(
                    success=True,
                    result=result,
                    fallback_level=1,
                    error_message=error_message,
                    error_type=error_type,
                    fallback_strategy="partial_result"
                )
            except Exception as e:
                logging.warning(f"Partial result fallback failed: {str(e)}")
        
        # Strategy 2: Fall back to basic keyword-based analysis
        try:
            result = self.sentiment_analyzer.analyze_user_input(text)
            self.fallback_stats["successful_attempts"] += 1
            self.fallback_stats["fallback_level_counts"][2] += 1
            return FallbackResult(
                success=True,
                result=result,
                fallback_level=2,
                error_message=error_message,
                error_type=error_type,
                fallback_strategy="keyword_based"
            )
        except Exception as e:
            logging.error(f"Keyword-based fallback failed: {str(e)}")
        
        # Strategy 3: Last resort - return neutral sentiment
        neutral_result = SentimentAnalysisResult(
            sentiment_score=0.0,
            interaction_type="neutral",
            affection_delta=0,
            confidence=0.1,  # Very low confidence
            detected_keywords=[],
            sentiment_types=[SentimentType.NEUTRAL]
        )
        
        logging.warning("All fallback strategies failed, returning neutral sentiment")
        return FallbackResult(
            success=False,
            result=neutral_result,
            fallback_level=3,
            error_message=error_message,
            error_type=error_type,
            fallback_strategy="neutral_default"
        )
    
    def _fallback_with_partial_result(self, text: str, partial_result: Dict) -> SentimentAnalysisResult:
        """
        Create a sentiment result using partial analysis data
        
        Args:
            text: The original text
            partial_result: Partial analysis result
            
        Returns:
            SentimentAnalysisResult constructed from partial data
        """
        # Extract whatever useful information we can from the partial result
        raw_sentiment = partial_result.get("raw_sentiment")
        
        if raw_sentiment:
            # We have the raw sentiment analysis, use it as a base
            return SentimentAnalysisResult(
                sentiment_score=raw_sentiment.sentiment_score,
                interaction_type=raw_sentiment.interaction_type,
                affection_delta=raw_sentiment.affection_delta,
                confidence=raw_sentiment.confidence * 0.8,  # Reduce confidence slightly
                detected_keywords=raw_sentiment.detected_keywords,
                sentiment_types=raw_sentiment.sentiment_types
            )
        
        # If we have contextual analysis but not raw sentiment
        contextual_analysis = partial_result.get("contextual_analysis")
        if contextual_analysis:
            # Try to construct a result from contextual analysis
            sentiment_score = self._estimate_sentiment_score_from_emotion(
                contextual_analysis.get("dominant_emotion", "neutral"),
                contextual_analysis.get("emotion_confidence", 0.5)
            )
            
            return SentimentAnalysisResult(
                sentiment_score=sentiment_score,
                interaction_type=self._emotion_to_interaction_type(
                    contextual_analysis.get("dominant_emotion", "neutral")
                ),
                affection_delta=int(sentiment_score * 10),  # Scale to -10 to 10
                confidence=contextual_analysis.get("emotion_confidence", 0.5) * 0.7,  # Reduce confidence
                detected_keywords=[],
                sentiment_types=[self._determine_sentiment_type(sentiment_score)]
            )
        
        # If we don't have enough information, fall back to keyword analysis
        raise ValueError("Insufficient partial result data for fallback")
    
    def _estimate_sentiment_score_from_emotion(self, emotion: str, confidence: float) -> float:
        """
        Estimate a sentiment score from an emotion name
        
        Args:
            emotion: The dominant emotion
            confidence: The confidence in the emotion
            
        Returns:
            Estimated sentiment score (-1.0 to 1.0)
        """
        # Map emotions to sentiment directions
        positive_emotions = ["joy", "trust", "anticipation"]
        negative_emotions = ["sadness", "anger", "fear", "disgust"]
        neutral_emotions = ["surprise", "neutral"]
        
        if emotion in positive_emotions:
            # Scale by confidence: 0.3 to 0.8
            return 0.3 + (confidence * 0.5)
        elif emotion in negative_emotions:
            # Scale by confidence: -0.3 to -0.8
            return -0.3 - (confidence * 0.5)
        else:
            # Neutral emotions get scores closer to 0
            return 0.0
    
    def _emotion_to_interaction_type(self, emotion: str) -> str:
        """
        Convert an emotion to an interaction type
        
        Args:
            emotion: The dominant emotion
            
        Returns:
            Interaction type string
        """
        # Map emotions to interaction types
        emotion_to_interaction = {
            "joy": "positive",
            "trust": "positive",
            "anticipation": "positive",
            "sadness": "negative",
            "anger": "negative",
            "fear": "negative",
            "disgust": "negative",
            "surprise": "neutral",
            "neutral": "neutral"
        }
        
        return emotion_to_interaction.get(emotion, "neutral")
    
    def _determine_sentiment_type(self, sentiment_score: float) -> SentimentType:
        """
        Determine sentiment type from score
        
        Args:
            sentiment_score: The sentiment score
            
        Returns:
            SentimentType enum value
        """
        if sentiment_score > 0.3:
            return SentimentType.POSITIVE
        elif sentiment_score < -0.3:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about fallback operations
        
        Returns:
            Dictionary with fallback statistics
        """
        if self.fallback_stats["total_attempts"] > 0:
            success_rate = (self.fallback_stats["successful_attempts"] / 
                           self.fallback_stats["total_attempts"]) * 100
        else:
            success_rate = 0
            
        return {
            "total_attempts": self.fallback_stats["total_attempts"],
            "successful_attempts": self.fallback_stats["successful_attempts"],
            "success_rate": f"{success_rate:.1f}%",
            "fallback_level_counts": self.fallback_stats["fallback_level_counts"],
            "error_type_counts": self.fallback_stats["error_type_counts"]
        }
    
    def reset_stats(self) -> None:
        """Reset fallback statistics"""
        self.fallback_stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "fallback_level_counts": {0: 0, 1: 0, 2: 0, 3: 0},
            "error_type_counts": {}
        }