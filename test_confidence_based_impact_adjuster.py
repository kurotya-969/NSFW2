"""
Unit tests for the Confidence-Based Impact Adjuster module
"""

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from confidence_based_impact_adjuster import ConfidenceBasedImpactAdjuster, ImpactAdjustmentResult
from confidence_calculator import ConfidenceAssessmentResult
from sentiment_analyzer import SentimentAnalysisResult, SentimentType
from context_analyzer import ContextualAnalysis

class TestConfidenceBasedImpactAdjuster(unittest.TestCase):
    """Test cases for the ConfidenceBasedImpactAdjuster class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adjuster = ConfidenceBasedImpactAdjuster()
        
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
        
        # Mock the confidence calculator to return controlled results
        self.adjuster.confidence_calculator = MagicMock()
    
    def test_adjust_impact_high_confidence(self):
        """Test impact adjustment for high confidence case"""
        # Set up mock confidence assessment
        mock_confidence_assessment = ConfidenceAssessmentResult(
            overall_confidence=0.9,
            keyword_confidence=0.8,
            context_confidence=0.9,
            pattern_confidence=0.9,
            ambiguity_score=0.1,
            uncertainty_factors=[],
            confidence_breakdown={
                "keyword_match_strength": 0.8,
                "context_emotion_confidence": 0.9,
                "conversation_pattern_stability": 0.9,
                "contradiction_penalty": 0.0,
                "ambiguity_penalty": -0.1,
                "sarcasm_irony_penalty": 0.0,
                "intensity_clarity": 0.2,
                "mixed_emotion_penalty": 0.0
            },
            recommended_weight=0.9
        )
        
        self.adjuster.confidence_calculator.calculate_confidence.return_value = mock_confidence_assessment
        
        # Test with positive sentiment
        result = self.adjuster.adjust_impact(
            self.mock_sentiment_result,
            self.mock_contextual_analysis,
            "I am very happy with this. It's really great!"
        )
        
        # For high confidence, impact should be minimally adjusted
        self.assertEqual(result.original_affection_delta, 5)
        self.assertIn(result.adjusted_affection_delta, [4, 5])  # Allow for small rounding differences
        self.assertFalse(result.fallback_applied)
        self.assertGreaterEqual(result.adjustment_factor, 0.8)
        
        # Test with negative sentiment
        negative_sentiment = SentimentAnalysisResult(
            sentiment_score=-0.5,
            interaction_type="negative",
            affection_delta=-5,
            confidence=0.7,
            detected_keywords=["bad", "terrible"],
            sentiment_types=[SentimentType.NEGATIVE]
        )
        
        result = self.adjuster.adjust_impact(
            negative_sentiment,
            self.mock_contextual_analysis,
            "This is really terrible. I hate it."
        )
        
        # For high confidence, impact should be minimally adjusted
        self.assertEqual(result.original_affection_delta, -5)
        self.assertIn(result.adjusted_affection_delta, [-5, -4])  # Allow for small rounding differences
        self.assertFalse(result.fallback_applied)
    
    def test_adjust_impact_moderate_confidence(self):
        """Test impact adjustment for moderate confidence case"""
        # Set up mock confidence assessment
        mock_confidence_assessment = ConfidenceAssessmentResult(
            overall_confidence=0.6,
            keyword_confidence=0.6,
            context_confidence=0.6,
            pattern_confidence=0.6,
            ambiguity_score=0.3,
            uncertainty_factors=["hedging_words: maybe"],
            confidence_breakdown={
                "keyword_match_strength": 0.6,
                "context_emotion_confidence": 0.6,
                "conversation_pattern_stability": 0.6,
                "contradiction_penalty": -0.1,
                "ambiguity_penalty": -0.3,
                "sarcasm_irony_penalty": 0.0,
                "intensity_clarity": 0.1,
                "mixed_emotion_penalty": 0.0
            },
            recommended_weight=0.6
        )
        
        self.adjuster.confidence_calculator.calculate_confidence.return_value = mock_confidence_assessment
        
        # Test with positive sentiment
        result = self.adjuster.adjust_impact(
            self.mock_sentiment_result,
            self.mock_contextual_analysis,
            "I think I'm kind of happy with this. It seems good."
        )
        
        # For moderate confidence, impact should be moderately reduced
        self.assertEqual(result.original_affection_delta, 5)
        self.assertLess(result.adjusted_affection_delta, 5)
        self.assertGreaterEqual(result.adjusted_affection_delta, 2)
        self.assertFalse(result.fallback_applied)
        self.assertLess(result.adjustment_factor, 0.8)
        self.assertGreaterEqual(result.adjustment_factor, 0.4)
    
    def test_adjust_impact_low_confidence(self):
        """Test impact adjustment for low confidence case"""
        # Set up mock confidence assessment
        mock_confidence_assessment = ConfidenceAssessmentResult(
            overall_confidence=0.3,
            keyword_confidence=0.4,
            context_confidence=0.3,
            pattern_confidence=0.3,
            ambiguity_score=0.5,
            uncertainty_factors=["hedging_words: maybe", "uncertainty_qualifiers: kind of", "ambivalent_words: mixed"],
            confidence_breakdown={
                "keyword_match_strength": 0.4,
                "context_emotion_confidence": 0.3,
                "conversation_pattern_stability": 0.3,
                "contradiction_penalty": -0.2,
                "ambiguity_penalty": -0.5,
                "sarcasm_irony_penalty": -0.1,
                "intensity_clarity": 0.0,
                "mixed_emotion_penalty": -0.2
            },
            recommended_weight=0.3
        )
        
        self.adjuster.confidence_calculator.calculate_confidence.return_value = mock_confidence_assessment
        
        # Test with positive sentiment
        result = self.adjuster.adjust_impact(
            self.mock_sentiment_result,
            self.mock_contextual_analysis,
            "I have mixed feelings about this. Maybe it's kind of good but also bad?"
        )
        
        # For low confidence, impact should be significantly reduced
        self.assertEqual(result.original_affection_delta, 5)
        self.assertLessEqual(result.adjusted_affection_delta, 2)
        self.assertFalse(result.fallback_applied)  # Not low enough for fallback
        self.assertLessEqual(result.adjustment_factor, 0.4)
    
    def test_adjust_impact_very_low_confidence(self):
        """Test impact adjustment for very low confidence case"""
        # Set up mock confidence assessment
        mock_confidence_assessment = ConfidenceAssessmentResult(
            overall_confidence=0.2,
            keyword_confidence=0.3,
            context_confidence=0.2,
            pattern_confidence=0.2,
            ambiguity_score=0.7,
            uncertainty_factors=["hedging_words: maybe", "uncertainty_qualifiers: kind of", 
                               "ambivalent_words: mixed", "hedging_words: guess"],
            confidence_breakdown={
                "keyword_match_strength": 0.3,
                "context_emotion_confidence": 0.2,
                "conversation_pattern_stability": 0.2,
                "contradiction_penalty": -0.3,
                "ambiguity_penalty": -0.7,
                "sarcasm_irony_penalty": -0.2,
                "intensity_clarity": 0.0,
                "mixed_emotion_penalty": -0.3
            },
            recommended_weight=0.2
        )
        
        self.adjuster.confidence_calculator.calculate_confidence.return_value = mock_confidence_assessment
        
        # Test with positive sentiment
        result = self.adjuster.adjust_impact(
            self.mock_sentiment_result,
            self.mock_contextual_analysis,
            "I really don't know if this is good or bad. Maybe it's kind of good? But also terrible? I'm so confused."
        )
        
        # For very low confidence, fallback should be applied
        self.assertEqual(result.original_affection_delta, 5)
        self.assertLessEqual(result.adjusted_affection_delta, 1)  # Fallback limits to +/-1
        self.assertTrue(result.fallback_applied)
        self.assertLessEqual(result.adjustment_factor, 0.3)
    
    def test_adjust_impact_with_contradictions(self):
        """Test impact adjustment with contradictions"""
        # Set up mock confidence assessment
        mock_confidence_assessment = ConfidenceAssessmentResult(
            overall_confidence=0.5,
            keyword_confidence=0.6,
            context_confidence=0.5,
            pattern_confidence=0.5,
            ambiguity_score=0.4,
            uncertainty_factors=[],
            confidence_breakdown={
                "keyword_match_strength": 0.6,
                "context_emotion_confidence": 0.5,
                "conversation_pattern_stability": 0.5,
                "contradiction_penalty": -0.3,
                "ambiguity_penalty": -0.4,
                "sarcasm_irony_penalty": 0.0,
                "intensity_clarity": 0.1,
                "mixed_emotion_penalty": 0.0
            },
            recommended_weight=0.5
        )
        
        self.adjuster.confidence_calculator.calculate_confidence.return_value = mock_confidence_assessment
        
        # Test with contradictions
        result = self.adjuster.adjust_impact(
            self.mock_sentiment_result,
            self.mock_contextual_analysis,
            "This is great but also terrible.",
            contradictions=["mixed_signals", "positive_keywords_negative_context"]
        )
        
        # With contradictions, impact should be reduced
        self.assertEqual(result.original_affection_delta, 5)
        self.assertLess(result.adjusted_affection_delta, 4)
        self.assertFalse(result.fallback_applied)
        self.assertIn("Contradictions detected", result.adjustment_reason)
    
    def test_adjust_impact_with_sarcasm(self):
        """Test impact adjustment with sarcasm"""
        # Set up mock confidence assessment
        mock_confidence_assessment = ConfidenceAssessmentResult(
            overall_confidence=0.5,
            keyword_confidence=0.6,
            context_confidence=0.5,
            pattern_confidence=0.5,
            ambiguity_score=0.3,
            uncertainty_factors=[],
            confidence_breakdown={
                "keyword_match_strength": 0.6,
                "context_emotion_confidence": 0.5,
                "conversation_pattern_stability": 0.5,
                "contradiction_penalty": -0.1,
                "ambiguity_penalty": -0.3,
                "sarcasm_irony_penalty": -0.3,
                "intensity_clarity": 0.1,
                "mixed_emotion_penalty": 0.0
            },
            recommended_weight=0.5
        )
        
        self.adjuster.confidence_calculator.calculate_confidence.return_value = mock_confidence_assessment
        
        # Adjust mock contextual analysis for sarcasm
        mock_contextual_analysis = MagicMock(spec=ContextualAnalysis)
        mock_contextual_analysis.dominant_emotion = "anger"
        mock_contextual_analysis.emotion_confidence = 0.6
        mock_contextual_analysis.sarcasm_probability = 0.8
        mock_contextual_analysis.irony_probability = 0.7
        mock_contextual_analysis.contextual_modifiers = ["just"]
        
        # Test with sarcasm
        result = self.adjuster.adjust_impact(
            self.mock_sentiment_result,
            mock_contextual_analysis,
            "Oh great, another error. This is just wonderful.",
            contradictions=["sarcastic_positive"]
        )
        
        # With sarcasm, impact should be reduced
        self.assertEqual(result.original_affection_delta, 5)
        self.assertLess(result.adjusted_affection_delta, 4)
        self.assertFalse(result.fallback_applied)
        self.assertIn("Sarcasm or irony detected", result.adjustment_reason)
    
    def test_get_adjustment_explanation(self):
        """Test generation of adjustment explanation"""
        # Create a sample result
        mock_confidence_assessment = ConfidenceAssessmentResult(
            overall_confidence=0.6,
            keyword_confidence=0.7,
            context_confidence=0.6,
            pattern_confidence=0.5,
            ambiguity_score=0.3,
            uncertainty_factors=["hedging_words: maybe"],
            confidence_breakdown={
                "keyword_match_strength": 0.7,
                "context_emotion_confidence": 0.6,
                "conversation_pattern_stability": 0.5,
                "contradiction_penalty": -0.1,
                "ambiguity_penalty": -0.3,
                "sarcasm_irony_penalty": 0.0,
                "intensity_clarity": 0.1,
                "mixed_emotion_penalty": 0.0
            },
            recommended_weight=0.6
        )
        
        result = ImpactAdjustmentResult(
            original_affection_delta=5,
            adjusted_affection_delta=3,
            confidence_score=0.6,
            adjustment_factor=0.6,
            fallback_applied=False,
            adjustment_reason="Moderate confidence",
            confidence_assessment=mock_confidence_assessment
        )
        
        explanation = self.adjuster.get_adjustment_explanation(result)
        
        # Check that explanation contains key information
        self.assertIn("Original affection impact: +5", explanation)
        self.assertIn("Adjusted affection impact: +3", explanation)
        self.assertIn("Confidence score: 0.60", explanation)
        self.assertIn("Adjustment factor: 0.60", explanation)
        self.assertIn("Adjustment reason: Moderate confidence", explanation)
        self.assertIn("Confidence assessment details", explanation)

if __name__ == '__main__':
    unittest.main()