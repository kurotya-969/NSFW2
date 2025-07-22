"""
Tests for the Sentiment Fallback Handler
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

from sentiment_analyzer import SentimentAnalyzer, SentimentType, SentimentAnalysisResult
from sentiment_fallback_handler import SentimentFallbackHandler, FallbackResult

class TestSentimentFallbackHandler(unittest.TestCase):
    """Test cases for the SentimentFallbackHandler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.handler = SentimentFallbackHandler()
        
        # Create mock sentiment analyzer
        self.handler.sentiment_analyzer = MagicMock()
    
    def test_initialization(self):
        """Test handler initialization"""
        handler = SentimentFallbackHandler()
        stats = handler.get_fallback_stats()
        self.assertEqual(stats["total_attempts"], 0)
        self.assertEqual(stats["successful_attempts"], 0)
        self.assertEqual(stats["fallback_level_counts"], {0: 0, 1: 0, 2: 0, 3: 0})
        self.assertEqual(stats["error_type_counts"], {})
    
    def test_handle_analysis_error_with_partial_result(self):
        """Test handling error with partial result available"""
        # Create mock raw sentiment result
        mock_raw_sentiment = MagicMock()
        mock_raw_sentiment.sentiment_score = 0.5
        mock_raw_sentiment.interaction_type = "positive"
        mock_raw_sentiment.affection_delta = 5
        mock_raw_sentiment.confidence = 0.7
        mock_raw_sentiment.detected_keywords = ["good", "great"]
        mock_raw_sentiment.sentiment_types = [SentimentType.POSITIVE]
        
        # Create partial result with raw sentiment
        partial_result = {"raw_sentiment": mock_raw_sentiment}
        
        # Call the method
        result = self.handler.handle_analysis_error(
            "This is great!", ValueError("Test error"), partial_result
        )
        
        # Verify the result
        self.assertTrue(result.success)
        self.assertEqual(result.fallback_level, 1)
        self.assertEqual(result.error_type, "ValueError")
        self.assertEqual(result.fallback_strategy, "partial_result")
        
        # Verify the sentiment result
        self.assertEqual(result.result.sentiment_score, 0.5)
        self.assertEqual(result.result.interaction_type, "positive")
        self.assertEqual(result.result.affection_delta, 5)
        self.assertAlmostEqual(result.result.confidence, 0.56, places=2)  # 0.7 * 0.8
        
        # Verify stats were updated
        stats = self.handler.get_fallback_stats()
        self.assertEqual(stats["total_attempts"], 1)
        self.assertEqual(stats["successful_attempts"], 1)
        self.assertEqual(stats["fallback_level_counts"][1], 1)
        self.assertEqual(stats["error_type_counts"]["ValueError"], 1)
    
    def test_handle_analysis_error_with_contextual_analysis(self):
        """Test handling error with only contextual analysis available"""
        # Create mock contextual analysis
        mock_contextual = {
            "dominant_emotion": "joy",
            "emotion_confidence": 0.8
        }
        
        # Create partial result with contextual analysis
        partial_result = {"contextual_analysis": mock_contextual}
        
        # Call the method
        result = self.handler.handle_analysis_error(
            "This is great!", RuntimeError("Test error"), partial_result
        )
        
        # Verify the result
        self.assertTrue(result.success)
        self.assertEqual(result.fallback_level, 1)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertEqual(result.fallback_strategy, "partial_result")
        
        # Verify the sentiment result
        self.assertGreater(result.result.sentiment_score, 0)  # Should be positive for "joy"
        self.assertEqual(result.result.interaction_type, "positive")
        self.assertGreater(result.result.affection_delta, 0)  # Should be positive
        
        # Verify stats were updated
        stats = self.handler.get_fallback_stats()
        self.assertEqual(stats["total_attempts"], 1)
        self.assertEqual(stats["successful_attempts"], 1)
        self.assertEqual(stats["fallback_level_counts"][1], 1)
        self.assertEqual(stats["error_type_counts"]["RuntimeError"], 1)
    
    def test_handle_analysis_error_with_keyword_fallback(self):
        """Test falling back to keyword-based analysis"""
        # Set up mock return value for keyword analysis
        mock_result = SentimentAnalysisResult(
            sentiment_score=0.5,
            interaction_type="positive",
            affection_delta=5,
            confidence=0.7,
            detected_keywords=["good", "great"],
            sentiment_types=[SentimentType.POSITIVE]
        )
        
        self.handler.sentiment_analyzer.analyze_user_input.return_value = mock_result
        
        # Call the method with no partial result
        result = self.handler.handle_analysis_error(
            "This is great!", ValueError("Test error"), None
        )
        
        # Verify the result
        self.assertTrue(result.success)
        self.assertEqual(result.fallback_level, 2)
        self.assertEqual(result.error_type, "ValueError")
        self.assertEqual(result.fallback_strategy, "keyword_based")
        
        # Verify the sentiment result
        self.assertEqual(result.result.sentiment_score, 0.5)
        self.assertEqual(result.result.interaction_type, "positive")
        self.assertEqual(result.result.affection_delta, 5)
        
        # Verify the mock was called correctly
        self.handler.sentiment_analyzer.analyze_user_input.assert_called_once_with("This is great!")
        
        # Verify stats were updated
        stats = self.handler.get_fallback_stats()
        self.assertEqual(stats["total_attempts"], 1)
        self.assertEqual(stats["successful_attempts"], 1)
        self.assertEqual(stats["fallback_level_counts"][2], 1)
        self.assertEqual(stats["error_type_counts"]["ValueError"], 1)
    
    def test_handle_analysis_error_with_all_fallbacks_failing(self):
        """Test when all fallback strategies fail"""
        # Set up mock to raise an exception for keyword analysis
        self.handler.sentiment_analyzer.analyze_user_input.side_effect = Exception("Keyword analysis failed")
        
        # Call the method with no partial result
        result = self.handler.handle_analysis_error(
            "This is great!", ValueError("Test error"), None
        )
        
        # Verify the result
        self.assertFalse(result.success)
        self.assertEqual(result.fallback_level, 3)
        self.assertEqual(result.error_type, "ValueError")
        self.assertEqual(result.fallback_strategy, "neutral_default")
        
        # Verify the sentiment result is neutral
        self.assertEqual(result.result.sentiment_score, 0.0)
        self.assertEqual(result.result.interaction_type, "neutral")
        self.assertEqual(result.result.affection_delta, 0)
        self.assertEqual(result.result.confidence, 0.1)  # Very low confidence
        
        # Verify stats were updated
        stats = self.handler.get_fallback_stats()
        self.assertEqual(stats["total_attempts"], 1)
        self.assertEqual(stats["successful_attempts"], 0)  # Should be 0 for unsuccessful
        self.assertEqual(stats["fallback_level_counts"][3], 0)  # Level 3 not counted in successful attempts
        self.assertEqual(stats["error_type_counts"]["ValueError"], 1)
    
    def test_reset_stats(self):
        """Test resetting statistics"""
        # Generate some stats
        self.handler.handle_analysis_error("Test", ValueError("Test error"), None)
        self.handler.handle_analysis_error("Test", RuntimeError("Test error"), None)
        
        # Verify stats were recorded
        stats = self.handler.get_fallback_stats()
        self.assertEqual(stats["total_attempts"], 2)
        
        # Reset stats
        self.handler.reset_stats()
        
        # Verify stats were reset
        stats = self.handler.get_fallback_stats()
        self.assertEqual(stats["total_attempts"], 0)
        self.assertEqual(stats["successful_attempts"], 0)
        self.assertEqual(stats["error_type_counts"], {})
    
    def test_emotion_to_interaction_type(self):
        """Test converting emotions to interaction types"""
        self.assertEqual(self.handler._emotion_to_interaction_type("joy"), "positive")
        self.assertEqual(self.handler._emotion_to_interaction_type("sadness"), "negative")
        self.assertEqual(self.handler._emotion_to_interaction_type("surprise"), "neutral")
        self.assertEqual(self.handler._emotion_to_interaction_type("unknown"), "neutral")
    
    def test_estimate_sentiment_score_from_emotion(self):
        """Test estimating sentiment scores from emotions"""
        # Test positive emotions
        self.assertGreater(self.handler._estimate_sentiment_score_from_emotion("joy", 0.8), 0)
        
        # Test negative emotions
        self.assertLess(self.handler._estimate_sentiment_score_from_emotion("anger", 0.8), 0)
        
        # Test neutral emotions
        self.assertEqual(self.handler._estimate_sentiment_score_from_emotion("surprise", 0.8), 0.0)
        
        # Test confidence scaling
        joy_high = self.handler._estimate_sentiment_score_from_emotion("joy", 0.9)
        joy_low = self.handler._estimate_sentiment_score_from_emotion("joy", 0.5)
        self.assertGreater(joy_high, joy_low)

if __name__ == "__main__":
    unittest.main()