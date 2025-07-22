"""
Tests for the Enhanced Sentiment Analyzer Adapter
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

from sentiment_analyzer import SentimentAnalyzer, SentimentType, SentimentAnalysisResult
from context_sentiment_detector import ContextSentimentDetector, ContextualSentimentResult
from enhanced_sentiment_adapter import EnhancedSentimentAdapter

class TestEnhancedSentimentAdapter(unittest.TestCase):
    """Test cases for the EnhancedSentimentAdapter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = EnhancedSentimentAdapter()
        
        # Create mock sentiment analyzer and context detector
        self.adapter.sentiment_analyzer = MagicMock()
        self.adapter.context_sentiment_detector = MagicMock()
    
    def test_initialization(self):
        """Test adapter initialization"""
        adapter = EnhancedSentimentAdapter()
        self.assertTrue(adapter.use_enhanced_analysis)
        self.assertIsNone(adapter.last_analysis_result)
        self.assertIsNone(adapter.last_contextual_result)
        
        adapter = EnhancedSentimentAdapter(use_enhanced_analysis=False)
        self.assertFalse(adapter.use_enhanced_analysis)
    
    def test_analyze_with_enhanced_analysis(self):
        """Test analyze_user_input with enhanced analysis enabled"""
        # Set up mock return values
        mock_raw_result = SentimentAnalysisResult(
            sentiment_score=0.5,
            interaction_type="positive",
            affection_delta=5,
            confidence=0.7,
            detected_keywords=["good", "great"],
            sentiment_types=[SentimentType.POSITIVE]
        )
        
        mock_contextual_analysis = MagicMock()
        mock_contextual_analysis.dominant_emotion = "joy"
        mock_contextual_analysis.emotion_confidence = 0.8
        mock_contextual_analysis.sarcasm_probability = 0.1
        mock_contextual_analysis.irony_probability = 0.1
        
        mock_contextual_result = MagicMock()
        mock_contextual_result.raw_sentiment = mock_raw_result
        mock_contextual_result.contextual_analysis = mock_contextual_analysis
        mock_contextual_result.adjusted_sentiment_score = 0.7
        mock_contextual_result.adjusted_affection_delta = 7
        mock_contextual_result.context_confidence = 0.8
        
        self.adapter.context_sentiment_detector.analyze_with_context.return_value = mock_contextual_result
        
        # Call the method
        result = self.adapter.analyze_user_input("This is great!", [])
        
        # Verify the result
        self.assertEqual(result.sentiment_score, 0.7)  # Should use adjusted score
        self.assertEqual(result.affection_delta, 7)    # Should use adjusted delta
        self.assertEqual(result.confidence, 0.8)       # Should use higher confidence
        self.assertEqual(result.interaction_type, "positive")
        self.assertEqual(result.detected_keywords, ["good", "great"])
        
        # Verify the mock was called correctly
        self.adapter.context_sentiment_detector.analyze_with_context.assert_called_once_with("This is great!", [])
        self.adapter.sentiment_analyzer.analyze_user_input.assert_not_called()
    
    def test_analyze_with_original_analysis(self):
        """Test analyze_user_input with original analysis"""
        # Set up mock return value
        mock_result = SentimentAnalysisResult(
            sentiment_score=0.5,
            interaction_type="positive",
            affection_delta=5,
            confidence=0.7,
            detected_keywords=["good", "great"],
            sentiment_types=[SentimentType.POSITIVE]
        )
        
        self.adapter.sentiment_analyzer.analyze_user_input.return_value = mock_result
        
        # Disable enhanced analysis
        self.adapter.use_enhanced_analysis = False
        
        # Call the method
        result = self.adapter.analyze_user_input("This is great!")
        
        # Verify the result
        self.assertEqual(result.sentiment_score, 0.5)
        self.assertEqual(result.affection_delta, 5)
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.interaction_type, "positive")
        self.assertEqual(result.detected_keywords, ["good", "great"])
        
        # Verify the mock was called correctly
        self.adapter.sentiment_analyzer.analyze_user_input.assert_called_once_with("This is great!")
        self.adapter.context_sentiment_detector.analyze_with_context.assert_not_called()
    
    def test_fallback_on_error(self):
        """Test fallback to original analyzer on error"""
        # Set up mock to raise an exception
        self.adapter.context_sentiment_detector.analyze_with_context.side_effect = Exception("Test error")
        
        # Set up mock return value for fallback
        mock_result = SentimentAnalysisResult(
            sentiment_score=0.5,
            interaction_type="positive",
            affection_delta=5,
            confidence=0.7,
            detected_keywords=["good", "great"],
            sentiment_types=[SentimentType.POSITIVE]
        )
        
        # Set up mock fallback handler
        self.adapter.fallback_handler = MagicMock()
        mock_fallback_result = MagicMock()
        mock_fallback_result.success = True
        mock_fallback_result.result = mock_result
        self.adapter.fallback_handler.handle_analysis_error.return_value = mock_fallback_result
        
        # Call the method
        result = self.adapter.analyze_user_input("This is great!")
        
        # Verify the result (should be from fallback handler)
        self.assertEqual(result.sentiment_score, 0.5)
        self.assertEqual(result.affection_delta, 5)
        self.assertEqual(result.confidence, 0.7)
        
        # Verify mocks were called
        self.adapter.context_sentiment_detector.analyze_with_context.assert_called_once()
        self.adapter.fallback_handler.handle_analysis_error.assert_called_once()
    
    def test_toggle_enhanced_analysis(self):
        """Test toggling between enhanced and original analysis"""
        self.adapter.use_enhanced_analysis = True
        self.adapter.toggle_enhanced_analysis(False)
        self.assertFalse(self.adapter.use_enhanced_analysis)
        
        self.adapter.toggle_enhanced_analysis(True)
        self.assertTrue(self.adapter.use_enhanced_analysis)
    
    def test_determine_interaction_type(self):
        """Test determining interaction type from contextual analysis"""
        # Test with special sentiment types
        sentiment_types = [SentimentType.SEXUAL]
        result = self.adapter._determine_interaction_type(0.5, MagicMock(), sentiment_types)
        self.assertEqual(result, "sexual")
        
        sentiment_types = [SentimentType.HOSTILE]
        result = self.adapter._determine_interaction_type(0.5, MagicMock(), sentiment_types)
        self.assertEqual(result, "hostile")
        
        # Test with high sarcasm probability
        contextual_analysis = MagicMock()
        contextual_analysis.sarcasm_probability = 0.8
        contextual_analysis.irony_probability = 0.2
        contextual_analysis.emotion_confidence = 0.5
        
        result = self.adapter._determine_interaction_type(0.5, contextual_analysis, [])
        self.assertEqual(result, "negative")
        
        # Test with dominant emotion and high confidence
        contextual_analysis = MagicMock()
        contextual_analysis.sarcasm_probability = 0.2
        contextual_analysis.irony_probability = 0.2
        contextual_analysis.emotion_confidence = 0.8
        contextual_analysis.dominant_emotion = "joy"
        
        result = self.adapter._determine_interaction_type(0.5, contextual_analysis, [])
        self.assertEqual(result, "positive")
        
        # Test fallback to sentiment score
        contextual_analysis = MagicMock()
        contextual_analysis.sarcasm_probability = 0.2
        contextual_analysis.irony_probability = 0.2
        contextual_analysis.emotion_confidence = 0.5
        
        result = self.adapter._determine_interaction_type(0.5, contextual_analysis, [])
        self.assertEqual(result, "positive")
        
        result = self.adapter._determine_interaction_type(-0.5, contextual_analysis, [])
        self.assertEqual(result, "negative")
        
        result = self.adapter._determine_interaction_type(0.2, contextual_analysis, [])
        self.assertEqual(result, "neutral")
    
    def test_get_sentiment_explanation(self):
        """Test getting sentiment explanation"""
        # Set up mock return values
        self.adapter.sentiment_analyzer.get_sentiment_explanation.return_value = "Original explanation"
        self.adapter.context_sentiment_detector.get_contextual_explanation.return_value = "Contextual explanation"
        
        # Test with enhanced analysis and contextual result
        self.adapter.use_enhanced_analysis = True
        self.adapter.last_contextual_result = MagicMock()
        self.adapter.last_analysis_result = MagicMock()
        
        explanation = self.adapter.get_sentiment_explanation()
        self.assertEqual(explanation, "Contextual explanation")
        
        # Test with original analysis
        self.adapter.use_enhanced_analysis = False
        explanation = self.adapter.get_sentiment_explanation()
        self.assertEqual(explanation, "Original explanation")
        
        # Test with no results
        self.adapter.last_contextual_result = None
        self.adapter.last_analysis_result = None
        explanation = self.adapter.get_sentiment_explanation()
        self.assertEqual(explanation, "No sentiment analysis result available")
    
    def test_get_detailed_analysis(self):
        """Test getting detailed analysis information"""
        # Test with no contextual result
        self.adapter.last_contextual_result = None
        result = self.adapter.get_detailed_analysis()
        self.assertFalse(result["available"])
        
        # Test with contextual result
        mock_contextual_result = MagicMock()
        mock_contextual_result.raw_sentiment.sentiment_score = 0.5
        mock_contextual_result.adjusted_sentiment_score = 0.7
        mock_contextual_result.raw_sentiment.affection_delta = 5
        mock_contextual_result.adjusted_affection_delta = 7
        mock_contextual_result.contextual_analysis.dominant_emotion = "joy"
        mock_contextual_result.contextual_analysis.emotion_confidence = 0.8
        mock_contextual_result.context_confidence = 0.8
        mock_contextual_result.contradictions_detected = False
        mock_contextual_result.context_override_applied = False
        mock_contextual_result.contextual_analysis.sarcasm_probability = 0.1
        mock_contextual_result.contextual_analysis.irony_probability = 0.1
        mock_contextual_result.raw_sentiment.detected_keywords = ["good", "great"]
        mock_contextual_result.contextual_analysis.contextual_modifiers = ["very"]
        mock_contextual_result.intensity_analysis = None
        mock_contextual_result.conversation_pattern = None
        
        self.adapter.last_contextual_result = mock_contextual_result
        
        result = self.adapter.get_detailed_analysis()
        self.assertTrue(result["available"])
        self.assertEqual(result["raw_sentiment_score"], 0.5)
        self.assertEqual(result["adjusted_sentiment_score"], 0.7)
        self.assertEqual(result["dominant_emotion"], "joy")

if __name__ == "__main__":
    unittest.main()