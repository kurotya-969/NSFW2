"""
Test script for context-aware sentiment detector
"""

import unittest
from context_sentiment_detector import ContextSentimentDetector, ContextualSentimentResult
from sentiment_analyzer import SentimentType

class TestContextSentimentDetector(unittest.TestCase):
    """Test cases for the ContextSentimentDetector class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.detector = ContextSentimentDetector()
    
    def test_positive_words_negative_context(self):
        """Test detection of positive words in negative context"""
        # Positive words but negative context
        result = self.detector.analyze_with_context("This is great, I'm so happy it failed completely.")
        
        # Should detect contradiction
        self.assertTrue(result.contradictions_detected)
        # Should adjust sentiment to be more negative
        self.assertLess(result.adjusted_sentiment_score, result.raw_sentiment.sentiment_score)
        # Should reduce affection impact
        self.assertLess(result.adjusted_affection_delta, result.raw_sentiment.affection_delta)
    
    def test_negative_words_positive_context(self):
        """Test detection of negative words in positive context"""
        # Negative words but positive context
        result = self.detector.analyze_with_context("This is terrible, I'm so excited it worked so well!")
        
        # Should detect contradiction
        self.assertTrue(result.contradictions_detected)
        # Should adjust sentiment to be more positive
        self.assertGreater(result.adjusted_sentiment_score, result.raw_sentiment.sentiment_score)
        # Should increase affection impact
        self.assertGreater(result.adjusted_affection_delta, result.raw_sentiment.affection_delta)
    
    def test_negated_positive(self):
        """Test detection of negated positive expressions"""
        # Negated positive expression
        result = self.detector.analyze_with_context("This is not good at all.")
        
        # Should detect contradiction
        self.assertTrue(result.contradictions_detected)
        # Should have negative adjusted sentiment
        self.assertLess(result.adjusted_sentiment_score, 0)
    
    def test_negated_negative(self):
        """Test detection of negated negative expressions"""
        # Negated negative expression
        result = self.detector.analyze_with_context("This is not bad actually.")
        
        # Should detect contradiction
        self.assertTrue(result.contradictions_detected)
        # Should have positive adjusted sentiment
        self.assertGreater(result.adjusted_sentiment_score, 0)
    
    def test_sarcasm_detection(self):
        """Test detection of sarcastic expressions"""
        # Sarcastic expression
        result = self.detector.analyze_with_context("Oh great, another error message. Just what I needed.")
        
        # Should detect contradiction
        self.assertTrue(result.contradictions_detected)
        # Should adjust sentiment to be negative
        self.assertLess(result.adjusted_sentiment_score, 0)
    
    def test_conditional_sentiment(self):
        """Test detection of conditional sentiment"""
        # Conditional positive sentiment
        result = self.detector.analyze_with_context("It would be great if this actually worked.")
        
        # Should detect conditional sentiment
        self.assertTrue(result.contradictions_detected)
        # Should reduce the impact of the sentiment
        self.assertLess(abs(result.adjusted_affection_delta), abs(result.raw_sentiment.affection_delta))
    
    def test_intensity_modifiers(self):
        """Test detection and application of intensity modifiers"""
        # With intensity modifier
        result_with_modifier = self.detector.analyze_with_context("I'm very happy with this.")
        # Without intensity modifier
        result_without_modifier = self.detector.analyze_with_context("I'm happy with this.")
        
        # Should have stronger sentiment with modifier
        self.assertGreater(result_with_modifier.adjusted_sentiment_score, result_without_modifier.adjusted_sentiment_score)
        # Should have stronger affection impact with modifier
        self.assertGreater(result_with_modifier.adjusted_affection_delta, result_without_modifier.adjusted_affection_delta)
    
    def test_japanese_context(self):
        """Test context-aware sentiment detection with Japanese text"""
        # Negated positive in Japanese
        result = self.detector.analyze_with_context("全然良くないです。")
        
        # Should detect contradiction
        self.assertTrue(result.contradictions_detected)
        # Should have negative adjusted sentiment
        self.assertLess(result.adjusted_sentiment_score, 0)
        
        # Sarcasm in Japanese
        result = self.detector.analyze_with_context("素晴らしいですね、また失敗しました。")
        
        # Should detect contradiction
        self.assertTrue(result.contradictions_detected)
        # Should adjust sentiment to be negative
        self.assertLess(result.adjusted_sentiment_score, 0)
    
    def test_no_contradictions(self):
        """Test that non-contradictory text is processed correctly"""
        # Simple positive sentiment
        result = self.detector.analyze_with_context("I'm happy today.")
        
        # Should not detect contradictions
        self.assertFalse(result.contradictions_detected)
        # Should not apply context override
        self.assertFalse(result.context_override_applied)
        # Scores should be similar
        self.assertAlmostEqual(result.adjusted_sentiment_score, result.raw_sentiment.sentiment_score, delta=0.3)
    
    def test_explanation_generation(self):
        """Test generation of human-readable explanations"""
        result = self.detector.analyze_with_context("This is not good at all.")
        explanation = self.detector.get_contextual_explanation(result)
        
        # Should contain key information
        self.assertIn("Raw sentiment score", explanation)
        self.assertIn("Context-adjusted score", explanation)
        self.assertIn("Dominant emotion", explanation)
        self.assertIn("Contradictions detected", explanation)

if __name__ == "__main__":
    unittest.main()