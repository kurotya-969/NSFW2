"""
Unit tests for the Confidence Calculator module
"""

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from confidence_calculator import ConfidenceCalculator, ConfidenceAssessmentResult
from sentiment_analyzer import SentimentAnalysisResult, SentimentType
from context_analyzer import ContextualAnalysis
from conversation_history_analyzer import ConversationPattern

class TestConfidenceCalculator(unittest.TestCase):
    """Test cases for the ConfidenceCalculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = ConfidenceCalculator()
        
        # Create mock objects for testing
        self.mock_sentiment_result = SentimentAnalysisResult(
            sentiment_score=0.5,
            interaction_type="positive",
            affection_delta=5,
            confidence=0.7,
            detected_keywords=["happy", "good"],
            sentiment_types=[SentimentType.POSITIVE]
        )
        
        self.mock_contextual_analysis = MagicMock(spec=ContextualAnalysis)
        self.mock_contextual_analysis.dominant_emotion = "joy"
        self.mock_contextual_analysis.emotion_confidence = 0.8
        self.mock_contextual_analysis.sarcasm_probability = 0.1
        self.mock_contextual_analysis.irony_probability = 0.1
        self.mock_contextual_analysis.contextual_modifiers = ["very"]
        
        self.mock_conversation_pattern = MagicMock(spec=ConversationPattern)
        self.mock_conversation_pattern.pattern_type = "consistent"
        self.mock_conversation_pattern.sentiment_stability = 0.8
        self.mock_conversation_pattern.dominant_emotions = ["joy", "trust"]
    
    def test_calculate_confidence_high_confidence_case(self):
        """Test confidence calculation for a high confidence case"""
        # Clear text with strong positive sentiment
        text = "I am very happy with this. It's really great!"
        
        result = self.calculator.calculate_confidence(
            self.mock_sentiment_result,
            self.mock_contextual_analysis,
            self.mock_conversation_pattern,
            text,
            []  # No contradictions
        )
        
        # Check that confidence is high
        self.assertGreaterEqual(result.overall_confidence, 0.7)
        self.assertEqual(result.keyword_confidence, 0.7)
        self.assertEqual(result.context_confidence, 0.8)
        self.assertGreaterEqual(result.recommended_weight, 0.7)
        
        # Check that there are no significant penalties
        self.assertEqual(len(result.uncertainty_factors), 0)
        self.assertLessEqual(result.ambiguity_score, 0.1)
    
    def test_calculate_confidence_ambiguous_case(self):
        """Test confidence calculation for an ambiguous case"""
        # Text with mixed signals
        text = "I'm kind of happy but also a bit sad about this. I guess it's good and bad."
        
        # Adjust mock objects for this test
        mock_sentiment_result = SentimentAnalysisResult(
            sentiment_score=0.2,
            interaction_type="mixed",
            affection_delta=2,
            confidence=0.5,
            detected_keywords=["happy", "sad", "good", "bad"],
            sentiment_types=[SentimentType.POSITIVE, SentimentType.NEGATIVE]
        )
        
        mock_contextual_analysis = MagicMock(spec=ContextualAnalysis)
        mock_contextual_analysis.dominant_emotion = "mixed"
        mock_contextual_analysis.emotion_confidence = 0.5
        mock_contextual_analysis.sarcasm_probability = 0.2
        mock_contextual_analysis.irony_probability = 0.1
        mock_contextual_analysis.contextual_modifiers = ["kind of", "a bit"]
        
        result = self.calculator.calculate_confidence(
            mock_sentiment_result,
            mock_contextual_analysis,
            None,  # No conversation pattern
            text,
            []  # No contradictions
        )
        
        # Check that confidence is lower
        self.assertLessEqual(result.overall_confidence, 0.6)
        self.assertEqual(result.keyword_confidence, 0.5)
        self.assertEqual(result.context_confidence, 0.5)
        
        # Check that ambiguity is detected
        self.assertGreaterEqual(result.ambiguity_score, 0.2)
        self.assertGreaterEqual(len(result.uncertainty_factors), 1)
        
        # Check that recommended weight is reduced
        self.assertLessEqual(result.recommended_weight, 0.6)
    
    def test_calculate_confidence_contradictory_case(self):
        """Test confidence calculation for a case with contradictions"""
        # Text with contradictions
        text = "This is great but it's also terrible. I love it and I hate it."
        
        # Adjust mock objects for this test
        mock_sentiment_result = SentimentAnalysisResult(
            sentiment_score=0.0,
            interaction_type="neutral",
            affection_delta=0,
            confidence=0.6,
            detected_keywords=["great", "terrible", "love", "hate"],
            sentiment_types=[SentimentType.POSITIVE, SentimentType.NEGATIVE]
        )
        
        mock_contextual_analysis = MagicMock(spec=ContextualAnalysis)
        mock_contextual_analysis.dominant_emotion = "mixed"
        mock_contextual_analysis.emotion_confidence = 0.4
        mock_contextual_analysis.sarcasm_probability = 0.3
        mock_contextual_analysis.irony_probability = 0.2
        mock_contextual_analysis.contextual_modifiers = []
        
        contradictions = ["mixed_signals", "positive_keywords_negative_context"]
        
        result = self.calculator.calculate_confidence(
            mock_sentiment_result,
            mock_contextual_analysis,
            None,  # No conversation pattern
            text,
            contradictions
        )
        
        # Check that confidence is low
        self.assertLessEqual(result.overall_confidence, 0.5)
        
        # Check that contradiction penalty is applied
        self.assertLess(result.confidence_breakdown["contradiction_penalty"], 0)
        
        # Check that recommended weight is reduced
        self.assertLessEqual(result.recommended_weight, 0.5)
    
    def test_calculate_confidence_sarcastic_case(self):
        """Test confidence calculation for a sarcastic case"""
        # Sarcastic text
        text = "Oh great, another error. This is just wonderful."
        
        # Adjust mock objects for this test
        mock_contextual_analysis = MagicMock(spec=ContextualAnalysis)
        mock_contextual_analysis.dominant_emotion = "anger"
        mock_contextual_analysis.emotion_confidence = 0.7
        mock_contextual_analysis.sarcasm_probability = 0.8
        mock_contextual_analysis.irony_probability = 0.7
        mock_contextual_analysis.contextual_modifiers = ["just"]
        
        result = self.calculator.calculate_confidence(
            self.mock_sentiment_result,
            mock_contextual_analysis,
            None,  # No conversation pattern
            text,
            ["sarcastic_positive"]  # Contradiction indicating sarcasm
        )
        
        # Check that sarcasm penalty is applied
        self.assertLess(result.confidence_breakdown["sarcasm_irony_penalty"], 0)
        
        # Check that confidence is reduced
        self.assertLessEqual(result.overall_confidence, 0.6)
        
        # Check that recommended weight is reduced
        self.assertLessEqual(result.recommended_weight, 0.6)
    
    def test_calculate_confidence_uncertain_case(self):
        """Test confidence calculation for an uncertain case"""
        # Uncertain text
        text = "I think maybe it's kind of good? I'm not really sure though."
        
        result = self.calculator.calculate_confidence(
            self.mock_sentiment_result,
            self.mock_contextual_analysis,
            None,  # No conversation pattern
            text,
            []  # No contradictions
        )
        
        # Check that uncertainty is detected
        self.assertGreaterEqual(len(result.uncertainty_factors), 2)
        
        # Check that confidence is reduced
        self.assertLessEqual(result.overall_confidence, 0.7)
    
    def test_identify_ambiguous_cases(self):
        """Test identification of ambiguous cases"""
        # Test various ambiguous texts
        ambiguous_texts = [
            "I'm happy but also sad about this.",
            "It might be good, but I'm not sure.",
            "I kind of like it, but also don't like some parts.",
            "If it worked properly, it would be great.",
            "I have mixed feelings about this."
        ]
        
        for text in ambiguous_texts:
            result = self.calculator.identify_ambiguous_cases(text)
            self.assertTrue(result["is_ambiguous"], f"Failed to identify ambiguity in: {text}")
            self.assertGreaterEqual(result["ambiguity_score"], 0.2)
            self.assertGreaterEqual(len(result["ambiguity_types"]), 1)
        
        # Test non-ambiguous texts
        clear_texts = [
            "I am very happy with this.",
            "This is terrible and I hate it.",
            "Thank you for your help.",
            "This doesn't work at all."
        ]
        
        for text in clear_texts:
            result = self.calculator.identify_ambiguous_cases(text)
            self.assertLessEqual(result["ambiguity_score"], 0.2)
    
    def test_get_confidence_explanation(self):
        """Test generation of confidence explanation"""
        # Create a sample result
        result = ConfidenceAssessmentResult(
            overall_confidence=0.65,
            keyword_confidence=0.7,
            context_confidence=0.6,
            pattern_confidence=0.8,
            ambiguity_score=0.3,
            uncertainty_factors=["hedging_words: maybe", "uncertainty_qualifiers: kind of"],
            confidence_breakdown={
                "keyword_match_strength": 0.7,
                "context_emotion_confidence": 0.6,
                "conversation_pattern_stability": 0.8,
                "contradiction_penalty": -0.1,
                "ambiguity_penalty": -0.3,
                "sarcasm_irony_penalty": -0.1,
                "intensity_clarity": 0.1,
                "mixed_emotion_penalty": -0.1
            },
            recommended_weight=0.65
        )
        
        explanation = self.calculator.get_confidence_explanation(result)
        
        # Check that explanation contains key information
        self.assertIn("Overall confidence: 0.65", explanation)
        self.assertIn("moderate", explanation)
        self.assertIn("Ambiguity detected", explanation)
        self.assertIn("Uncertainty indicators", explanation)
        self.assertIn("Recommended affection impact weight", explanation)

if __name__ == '__main__':
    unittest.main()