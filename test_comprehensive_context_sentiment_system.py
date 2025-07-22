"""
Comprehensive Unit Tests for Context-Based Sentiment Analysis System
Tests all components, integration points, edge cases, and boundary conditions
Requirements: All
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Import all components of the context-based sentiment system
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult, SentimentType
from context_analyzer import ContextAnalyzer, ContextualAnalysis
from context_sentiment_detector import ContextSentimentDetector, ContextualSentimentResult
from emotion_intensity_detector import EmotionIntensityDetector, IntensityAnalysisResult
from sarcasm_irony_detector import SarcasmIronyDetector, NonLiteralLanguageResult
from mixed_emotion_handler import MixedEmotionHandler, EmotionCategory, MixedEmotionResult
from confidence_calculator import ConfidenceCalculator, ConfidenceAssessmentResult
from confidence_based_impact_adjuster import ConfidenceBasedImpactAdjuster, ImpactAdjustmentResult
from sentiment_transition_smoother import SentimentTransitionSmoother, SentimentShift
from sentiment_pattern_recognizer import SentimentPatternRecognizer, SentimentPattern
from enhanced_sentiment_adapter import EnhancedSentimentAdapter
from sentiment_fallback_handler import SentimentFallbackHandler, FallbackResult
from conversation_history_analyzer import ConversationHistoryAnalyzer, ConversationPattern

class TestComprehensiveContextSentimentSystem(unittest.TestCase):
    """Comprehensive test cases for the Context-Based Sentiment Analysis System"""
    
    def setUp(self):
        """Set up the test environment"""
        # Initialize all components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        self.context_sentiment_detector = ContextSentimentDetector()
        self.emotion_intensity_detector = EmotionIntensityDetector()
        self.sarcasm_irony_detector = SarcasmIronyDetector()
        self.mixed_emotion_handler = MixedEmotionHandler()
        self.confidence_calculator = ConfidenceCalculator()
        self.impact_adjuster = ConfidenceBasedImpactAdjuster()
        self.transition_smoother = SentimentTransitionSmoother()
        self.pattern_recognizer = SentimentPatternRecognizer()
        self.enhanced_adapter = EnhancedSentimentAdapter()
        self.fallback_handler = SentimentFallbackHandler()
        self.conversation_history_analyzer = ConversationHistoryAnalyzer()
        
        # Sample conversation history for testing
        self.sample_conversation_history = [
            {"user": "Hello, how are you?", "assistant": "What do you want?", 
             "sentiment_score": 0.0, "affection_delta": 0},
            {"user": "I just wanted to talk.", "assistant": "Fine, whatever.", 
             "sentiment_score": -0.2, "affection_delta": -1},
            {"user": "You seem upset.", "assistant": "I'm not upset, just cautious.", 
             "sentiment_score": 0.1, "affection_delta": 1}
        ]
    
    # SECTION 1: COMPONENT INITIALIZATION TESTS
    
    def test_all_components_initialization(self):
        """Test proper initialization of all components"""
        # Verify all components were initialized correctly
        self.assertIsInstance(self.sentiment_analyzer, SentimentAnalyzer)
        self.assertIsInstance(self.context_analyzer, ContextAnalyzer)
        self.assertIsInstance(self.context_sentiment_detector, ContextSentimentDetector)
        self.assertIsInstance(self.emotion_intensity_detector, EmotionIntensityDetector)
        self.assertIsInstance(self.sarcasm_irony_detector, SarcasmIronyDetector)
        self.assertIsInstance(self.mixed_emotion_handler, MixedEmotionHandler)
        self.assertIsInstance(self.confidence_calculator, ConfidenceCalculator)
        self.assertIsInstance(self.impact_adjuster, ConfidenceBasedImpactAdjuster)
        self.assertIsInstance(self.transition_smoother, SentimentTransitionSmoother)
        self.assertIsInstance(self.pattern_recognizer, SentimentPatternRecognizer)
        self.assertIsInstance(self.enhanced_adapter, EnhancedSentimentAdapter)
        self.assertIsInstance(self.fallback_handler, SentimentFallbackHandler)
        self.assertIsInstance(self.conversation_history_analyzer, ConversationHistoryAnalyzer)
    
    # SECTION 2: CONTEXT ANALYZER TESTS
    
    def test_context_analyzer_with_complex_text(self):
        """Test context analyzer with complex text containing multiple emotions"""
        text = "I'm happy about the promotion but worried about the new responsibilities and deadlines."
        
        result = self.context_analyzer.analyze_context(text)
        
        # Verify multiple emotions were detected
        self.assertIsNotNone(result)
        self.assertIn(result.dominant_emotion, ["joy", "fear"])
        self.assertGreater(result.emotion_confidence, 0.5)
        self.assertGreaterEqual(len(result.detected_topics), 1)
        
        # Check for contextual modifiers
        self.assertTrue(any(modifier in result.contextual_modifiers for modifier in ["but", "about", "new"]))
    
    def test_context_analyzer_with_ambiguous_text(self):
        """Test context analyzer with ambiguous text"""
        text = "Well, I guess that's something."
        
        result = self.context_analyzer.analyze_context(text)
        
        # Verify ambiguity is reflected in confidence
        self.assertIsNotNone(result)
        self.assertLessEqual(result.emotion_confidence, 0.7)
    
    def test_context_analyzer_with_empty_text(self):
        """Test context analyzer with empty text"""
        text = ""
        
        result = self.context_analyzer.analyze_context(text)
        
        # Verify default values for empty text
        self.assertIsNotNone(result)
        self.assertEqual(result.dominant_emotion, "neutral")
        self.assertLessEqual(result.emotion_confidence, 0.3)
        self.assertEqual(len(result.detected_topics), 0)
    
    # SECTION 3: CONTEXT SENTIMENT DETECTOR TESTS
    
    def test_context_sentiment_detector_with_contradictions(self):
        """Test context sentiment detector with contradictory text"""
        text = "This is great, I'm so happy it failed completely."
        
        result = self.context_sentiment_detector.analyze_with_context(text)
        
        # Verify contradiction was detected and sentiment adjusted
        self.assertTrue(result.contradictions_detected)
        self.assertNotEqual(result.raw_sentiment.sentiment_score, result.adjusted_sentiment_score)
        self.assertNotEqual(result.raw_sentiment.affection_delta, result.adjusted_affection_delta)
    
    def test_context_sentiment_detector_with_negation(self):
        """Test context sentiment detector with negated expressions"""
        # Test negated positive
        text_pos = "This is not good at all."
        result_pos = self.context_sentiment_detector.analyze_with_context(text_pos)
        
        # Verify negation was detected and sentiment adjusted to negative
        self.assertTrue(result_pos.contradictions_detected)
        self.assertLess(result_pos.adjusted_sentiment_score, 0)
        
        # Test negated negative
        text_neg = "This is not bad actually."
        result_neg = self.context_sentiment_detector.analyze_with_context(text_neg)
        
        # Verify negation was detected and sentiment adjusted to positive
        self.assertTrue(result_neg.contradictions_detected)
        self.assertGreater(result_neg.adjusted_sentiment_score, 0)
    
    def test_context_sentiment_detector_with_conditional_statements(self):
        """Test context sentiment detector with conditional statements"""
        text = "It would be great if this actually worked."
        
        result = self.context_sentiment_detector.analyze_with_context(text)
        
        # Verify conditional sentiment was detected and impact reduced
        self.assertTrue(result.contradictions_detected)
        self.assertLess(abs(result.adjusted_affection_delta), abs(result.raw_sentiment.affection_delta))
    
    # SECTION 4: EMOTION INTENSITY DETECTOR TESTS
    
    def test_emotion_intensity_detector_with_varying_intensities(self):
        """Test emotion intensity detector with varying intensity levels"""
        # Test mild intensity
        mild_text = "I'm slightly interested in this topic."
        mild_result = self.emotion_intensity_detector.detect_intensity(mild_text)
        
        self.assertEqual(mild_result.intensity_category, "mild")
        self.assertLessEqual(mild_result.intensity_score, 0.3)
        
        # Test moderate intensity
        moderate_text = "I'm happy about this news."
        moderate_result = self.emotion_intensity_detector.detect_intensity(moderate_text)
        
        self.assertEqual(moderate_result.intensity_category, "moderate")
        self.assertGreater(moderate_result.intensity_score, 0.3)
        self.assertLessEqual(moderate_result.intensity_score, 0.6)
        
        # Test strong intensity
        strong_text = "I'm very excited about this opportunity!"
        strong_result = self.emotion_intensity_detector.detect_intensity(strong_text)
        
        self.assertEqual(strong_result.intensity_category, "strong")
        self.assertGreater(strong_result.intensity_score, 0.6)
        self.assertLessEqual(strong_result.intensity_score, 0.85)
        
        # Test extreme intensity
        extreme_text = "I'm absolutely THRILLED!!! This is the BEST news EVER!!! 😍😍😍"
        extreme_result = self.emotion_intensity_detector.detect_intensity(extreme_text)
        
        self.assertEqual(extreme_result.intensity_category, "extreme")
        self.assertGreater(extreme_result.intensity_score, 0.85)
    
    def test_emotion_intensity_detector_with_mixed_modifiers(self):
        """Test emotion intensity detector with mixed intensifiers and qualifiers"""
        text = "I'm very slightly happy."
        
        result = self.emotion_intensity_detector.detect_intensity(text)
        
        # Verify intensifiers and qualifiers were detected
        self.assertIn("very", result.intensifiers)
        self.assertIn("slightly", result.qualifiers)
        
        # Verify they balance each other
        self.assertGreater(result.intensity_score, 0.2)  # Not too low due to "slightly"
        self.assertLess(result.intensity_score, 0.7)     # Not too high due to "very"
    
    # SECTION 5: SARCASM AND IRONY DETECTOR TESTS
    
    def test_sarcasm_detector_with_clear_sarcasm(self):
        """Test sarcasm detector with clearly sarcastic text"""
        text = "Yeah right!!! That's TOTALLY how it works! ;)"
        
        result = self.sarcasm_irony_detector.detect_non_literal_language(text)
        
        # Verify sarcasm was detected with high confidence
        self.assertGreaterEqual(result.sarcasm_probability, 0.7)
        self.assertEqual(result.non_literal_type, "sarcasm")
        self.assertGreaterEqual(result.confidence, 0.7)
        self.assertGreaterEqual(len(result.detected_patterns), 1)
        self.assertGreaterEqual(len(result.context_indicators), 1)
    
    def test_sarcasm_detector_with_subtle_irony(self):
        """Test sarcasm detector with subtle irony"""
        text = "Just exactly what I needed today."
        
        result = self.sarcasm_irony_detector.detect_non_literal_language(text)
        
        # Verify irony was detected but with lower confidence
        self.assertGreaterEqual(result.irony_probability, 0.5)
        self.assertEqual(result.non_literal_type, "irony")
        self.assertLessEqual(result.confidence, 0.7)  # Lower confidence for subtle irony
    
    def test_sarcasm_detector_with_non_sarcastic_text(self):
        """Test sarcasm detector with non-sarcastic text"""
        text = "I'm happy with the results of the test."
        
        result = self.sarcasm_irony_detector.detect_non_literal_language(text)
        
        # Verify no sarcasm or irony was detected
        self.assertLess(result.sarcasm_probability, 0.5)
        self.assertLess(result.irony_probability, 0.5)
        self.assertIsNone(result.non_literal_type)
    
    # SECTION 6: MIXED EMOTION HANDLER TESTS
    
    def test_mixed_emotion_handler_with_explicit_mixed_emotions(self):
        """Test mixed emotion handler with explicitly stated mixed emotions"""
        text = "I'm happy but also sad about this situation."
        
        result = self.mixed_emotion_handler.detect_mixed_emotions(text)
        
        # Verify mixed emotions were detected
        self.assertTrue(result.is_mixed)
        self.assertTrue(result.conflicting_emotions)
        self.assertEqual(result.emotion_category, EmotionCategory.AMBIVALENT)
        self.assertGreater(result.emotion_ambivalence, 0.5)
        
        # Check that both emotions are detected
        self.assertIn("joy", result.emotions)
        self.assertIn("sadness", result.emotions)
    
    def test_mixed_emotion_handler_with_single_emotion(self):
        """Test mixed emotion handler with single emotion"""
        text = "I am so happy today!"
        
        result = self.mixed_emotion_handler.detect_mixed_emotions(text)
        
        # Verify single emotion was detected
        self.assertEqual(result.dominant_emotion, "joy")
        self.assertFalse(result.is_mixed)
        self.assertEqual(result.emotion_category, EmotionCategory.POSITIVE)
        self.assertGreater(result.emotion_ratio["positive"], 0.8)
        self.assertLess(result.emotion_complexity, 0.3)
    
    def test_mixed_emotion_handler_affection_impact(self):
        """Test mixed emotion handler's affection impact calculation"""
        # Test with ambivalent emotions
        ambivalent_result = self.mixed_emotion_handler.detect_mixed_emotions("I'm happy but also sad about this.")
        ambivalent_impact = self.mixed_emotion_handler.get_affection_impact(ambivalent_result)
        
        # Impact should be reduced for ambivalent emotions
        self.assertLess(abs(ambivalent_impact["sentiment_score"]), 0.5)
        self.assertLess(abs(ambivalent_impact["affection_delta"]), 3)
        self.assertLess(ambivalent_impact["confidence"], 0.8)
        
        # Test with single strong emotion
        single_result = self.mixed_emotion_handler.detect_mixed_emotions("I'm so happy and excited!")
        single_impact = self.mixed_emotion_handler.get_affection_impact(single_result)
        
        # Impact should be stronger for single clear emotion
        self.assertGreater(single_impact["sentiment_score"], 0.5)
        self.assertGreater(single_impact["affection_delta"], 3)
        self.assertGreater(single_impact["confidence"], 0.7)
    
    # SECTION 7: CONFIDENCE CALCULATOR TESTS
    
    def test_confidence_calculator_with_high_confidence_case(self):
        """Test confidence calculator with high confidence case"""
        # Clear text with strong positive sentiment
        text = "I am very happy with this. It's really great!"
        
        # Create mock objects for testing
        mock_sentiment_result = SentimentAnalysisResult(
            sentiment_score=0.7,
            interaction_type="positive",
            affection_delta=7,
            confidence=0.8,
            detected_keywords=["happy", "great"],
            sentiment_types=[SentimentType.POSITIVE]
        )
        
        mock_contextual_analysis = MagicMock(spec=ContextualAnalysis)
        mock_contextual_analysis.dominant_emotion = "joy"
        mock_contextual_analysis.emotion_confidence = 0.9
        mock_contextual_analysis.sarcasm_probability = 0.1
        mock_contextual_analysis.irony_probability = 0.1
        mock_contextual_analysis.contextual_modifiers = ["very", "really"]
        
        mock_conversation_pattern = MagicMock(spec=ConversationPattern)
        mock_conversation_pattern.pattern_type = "consistent"
        mock_conversation_pattern.sentiment_stability = 0.9
        mock_conversation_pattern.dominant_emotions = ["joy"]
        
        result = self.confidence_calculator.calculate_confidence(
            mock_sentiment_result,
            mock_contextual_analysis,
            mock_conversation_pattern,
            text,
            []  # No contradictions
        )
        
        # Verify high confidence
        self.assertGreaterEqual(result.overall_confidence, 0.8)
        self.assertGreaterEqual(result.recommended_weight, 0.8)
        self.assertEqual(len(result.uncertainty_factors), 0)
    
    def test_confidence_calculator_with_ambiguous_case(self):
        """Test confidence calculator with ambiguous case"""
        # Text with mixed signals
        text = "I'm kind of happy but also a bit sad about this. I guess it's good and bad."
        
        # Create mock objects for testing
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
        mock_contextual_analysis.contextual_modifiers = ["kind of", "a bit", "guess"]
        
        result = self.confidence_calculator.calculate_confidence(
            mock_sentiment_result,
            mock_contextual_analysis,
            None,  # No conversation pattern
            text,
            []  # No contradictions
        )
        
        # Verify lower confidence
        self.assertLessEqual(result.overall_confidence, 0.6)
        self.assertGreaterEqual(result.ambiguity_score, 0.3)
        self.assertGreaterEqual(len(result.uncertainty_factors), 1)
        self.assertLessEqual(result.recommended_weight, 0.6)
    
    # SECTION 8: CONFIDENCE-BASED IMPACT ADJUSTER TESTS
    
    def test_impact_adjuster_with_varying_confidence_levels(self):
        """Test impact adjuster with varying confidence levels"""
        # Create mock sentiment result
        mock_sentiment_result = SentimentAnalysisResult(
            sentiment_score=0.7,
            interaction_type="positive",
            affection_delta=7,
            confidence=0.8,
            detected_keywords=["happy", "great"],
            sentiment_types=[SentimentType.POSITIVE]
        )
        
        mock_contextual_analysis = MagicMock(spec=ContextualAnalysis)
        mock_contextual_analysis.dominant_emotion = "joy"
        mock_contextual_analysis.emotion_confidence = 0.8
        
        # Test with high confidence
        with patch.object(self.impact_adjuster, 'confidence_calculator') as mock_calculator:
            mock_confidence_assessment = MagicMock()
            mock_confidence_assessment.overall_confidence = 0.9
            mock_confidence_assessment.recommended_weight = 0.9
            mock_confidence_assessment.uncertainty_factors = []
            mock_calculator.calculate_confidence.return_value = mock_confidence_assessment
            
            result_high = self.impact_adjuster.adjust_impact(
                mock_sentiment_result,
                mock_contextual_analysis,
                "I am very happy with this!"
            )
            
            # Verify minimal adjustment for high confidence
            self.assertGreaterEqual(result_high.adjusted_affection_delta, 6)
            self.assertFalse(result_high.fallback_applied)
        
        # Test with moderate confidence
        with patch.object(self.impact_adjuster, 'confidence_calculator') as mock_calculator:
            mock_confidence_assessment = MagicMock()
            mock_confidence_assessment.overall_confidence = 0.6
            mock_confidence_assessment.recommended_weight = 0.6
            mock_confidence_assessment.uncertainty_factors = ["hedging_words"]
            mock_calculator.calculate_confidence.return_value = mock_confidence_assessment
            
            result_moderate = self.impact_adjuster.adjust_impact(
                mock_sentiment_result,
                mock_contextual_analysis,
                "I think I'm kind of happy with this."
            )
            
            # Verify moderate adjustment
            self.assertLess(result_moderate.adjusted_affection_delta, 7)
            self.assertGreaterEqual(result_moderate.adjusted_affection_delta, 3)
            self.assertFalse(result_moderate.fallback_applied)
        
        # Test with low confidence
        with patch.object(self.impact_adjuster, 'confidence_calculator') as mock_calculator:
            mock_confidence_assessment = MagicMock()
            mock_confidence_assessment.overall_confidence = 0.2
            mock_confidence_assessment.recommended_weight = 0.2
            mock_confidence_assessment.uncertainty_factors = ["hedging_words", "ambivalent_words", "uncertainty_qualifiers"]
            mock_calculator.calculate_confidence.return_value = mock_confidence_assessment
            
            result_low = self.impact_adjuster.adjust_impact(
                mock_sentiment_result,
                mock_contextual_analysis,
                "I really don't know if this is good or bad. Maybe it's kind of good? But also terrible?"
            )
            
            # Verify significant adjustment or fallback for low confidence
            self.assertLessEqual(result_low.adjusted_affection_delta, 2)
    
    # SECTION 9: SENTIMENT TRANSITION SMOOTHER TESTS
    
    def test_transition_smoother_with_dramatic_shifts(self):
        """Test transition smoother with dramatic sentiment shifts"""
        # Test dramatic shift from positive to negative
        current_sentiment = {"sentiment_score": -0.7, "affection_delta": -7, "dominant_emotion": "anger"}
        previous_sentiment = {"sentiment_score": 0.6, "affection_delta": 6, "dominant_emotion": "joy"}
        conversation_history = [
            {"sentiment_score": 0.5, "affection_delta": 5},
            {"sentiment_score": 0.6, "affection_delta": 6}
        ]
        
        smoothed_score, smoothed_delta, shift = self.transition_smoother.apply_smoothing(
            current_sentiment, previous_sentiment, conversation_history
        )
        
        # Verify smoothing was applied
        self.assertTrue(shift.smoothing_applied)
        self.assertTrue(-0.7 < smoothed_score < 0.6)
        self.assertTrue(-7 < smoothed_delta < 6)
    
    def test_transition_smoother_with_minimal_shifts(self):
        """Test transition smoother with minimal sentiment shifts"""
        # Test minimal shift (no smoothing needed)
        current_sentiment = {"sentiment_score": 0.25, "affection_delta": 2, "dominant_emotion": "joy"}
        previous_sentiment = {"sentiment_score": 0.2, "affection_delta": 2, "dominant_emotion": "joy"}
        
        smoothed_score, smoothed_delta, shift = self.transition_smoother.apply_smoothing(
            current_sentiment, previous_sentiment, []
        )
        
        # Verify no smoothing was applied
        self.assertFalse(shift.smoothing_applied)
        self.assertEqual(smoothed_score, 0.25)
        self.assertEqual(smoothed_delta, 2)
    
    # SECTION 10: SENTIMENT PATTERN RECOGNIZER TESTS
    
    def test_pattern_recognizer_with_consistent_pattern(self):
        """Test pattern recognizer with consistent sentiment pattern"""
        # Create history with consistent positive sentiment
        history = [
            {"sentiment_score": 0.6, "dominant_emotion": "joy", "emotion_confidence": 0.7},
            {"sentiment_score": 0.65, "dominant_emotion": "joy", "emotion_confidence": 0.7},
            {"sentiment_score": 0.62, "dominant_emotion": "joy", "emotion_confidence": 0.8},
            {"sentiment_score": 0.58, "dominant_emotion": "joy", "emotion_confidence": 0.7}
        ]
        
        pattern = self.pattern_recognizer.recognize_pattern(history)
        
        # Verify consistent pattern was detected
        self.assertEqual(pattern.pattern_type, "consistent")
        self.assertEqual(pattern.dominant_emotion, "joy")
        self.assertTrue(pattern.sentiment_stability > 0.8)
        self.assertAlmostEqual(pattern.intensity_trend, 0.0, delta=0.2)
        self.assertTrue(pattern.strengthening_factor > 0.0)
    
    def test_pattern_recognizer_with_escalating_pattern(self):
        """Test pattern recognizer with escalating sentiment pattern"""
        # Create history with escalating positive sentiment
        history = [
            {"sentiment_score": 0.3, "dominant_emotion": "joy", "emotion_confidence": 0.5},
            {"sentiment_score": 0.5, "dominant_emotion": "joy", "emotion_confidence": 0.6},
            {"sentiment_score": 0.7, "dominant_emotion": "joy", "emotion_confidence": 0.7},
            {"sentiment_score": 0.8, "dominant_emotion": "joy", "emotion_confidence": 0.8}
        ]
        
        pattern = self.pattern_recognizer.recognize_pattern(history)
        
        # Verify escalating pattern was detected
        self.assertEqual(pattern.pattern_type, "escalating")
        self.assertEqual(pattern.dominant_emotion, "joy")
        self.assertTrue(pattern.intensity_trend > 0.2)
        self.assertTrue(pattern.strengthening_factor > 0.0)
    
    def test_pattern_recognizer_with_insufficient_data(self):
        """Test pattern recognizer with insufficient data"""
        # Test with empty history
        pattern = self.pattern_recognizer.recognize_pattern([])
        
        # Verify insufficient data pattern
        self.assertEqual(pattern.pattern_type, "insufficient_data")
        self.assertEqual(pattern.duration, 0)
        self.assertEqual(pattern.strengthening_factor, 0.0)
    
    # SECTION 11: ENHANCED SENTIMENT ADAPTER TESTS
    
    def test_enhanced_adapter_with_enhanced_analysis(self):
        """Test enhanced sentiment adapter with enhanced analysis enabled"""
        # Set up mock return values
        mock_raw_result = SentimentAnalysisResult(
            sentiment_score=0.5,
            interaction_type="positive",
            affection_delta=5,
            confidence=0.7,
            detected_keywords=["good", "great"],
            sentiment_types=[SentimentType.POSITIVE]
        )
        
        mock_contextual_result = MagicMock()
        mock_contextual_result.raw_sentiment = mock_raw_result
        mock_contextual_result.adjusted_sentiment_score = 0.7
        mock_contextual_result.adjusted_affection_delta = 7
        mock_contextual_result.context_confidence = 0.8
        
        # Create a new adapter with mocked components
        adapter = EnhancedSentimentAdapter()
        adapter.context_sentiment_detector = MagicMock()
        adapter.context_sentiment_detector.analyze_with_context.return_value = mock_contextual_result
        
        # Call the method
        result = adapter.analyze_user_input("This is great!")
        
        # Verify the result uses enhanced analysis
        self.assertEqual(result.sentiment_score, 0.7)  # Should use adjusted score
        self.assertEqual(result.affection_delta, 7)    # Should use adjusted delta
    
    def test_enhanced_adapter_with_fallback(self):
        """Test enhanced sentiment adapter with fallback on error"""
        # Set up mock to raise an exception
        adapter = EnhancedSentimentAdapter()
        adapter.context_sentiment_detector = MagicMock()
        adapter.context_sentiment_detector.analyze_with_context.side_effect = Exception("Test error")
        
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
        adapter.fallback_handler = MagicMock()
        mock_fallback_result = MagicMock()
        mock_fallback_result.success = True
        mock_fallback_result.result = mock_result
        adapter.fallback_handler.handle_analysis_error.return_value = mock_fallback_result
        
        # Call the method
        result = adapter.analyze_user_input("This is great!")
        
        # Verify the result (should be from fallback handler)
        self.assertEqual(result.sentiment_score, 0.5)
        self.assertEqual(result.affection_delta, 5)
        
        # Verify mocks were called
        adapter.context_sentiment_detector.analyze_with_context.assert_called_once()
        adapter.fallback_handler.handle_analysis_error.assert_called_once()
    
    # SECTION 12: SENTIMENT FALLBACK HANDLER TESTS
    
    def test_fallback_handler_with_partial_result(self):
        """Test fallback handler with partial result available"""
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
        result = self.fallback_handler.handle_analysis_error(
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
    
    def test_fallback_handler_with_all_fallbacks_failing(self):
        """Test fallback handler when all fallback strategies fail"""
        # Set up mock to raise an exception for keyword analysis
        handler = SentimentFallbackHandler()
        handler.sentiment_analyzer = MagicMock()
        handler.sentiment_analyzer.analyze_user_input.side_effect = Exception("Keyword analysis failed")
        
        # Call the method with no partial result
        result = handler.handle_analysis_error(
            "This is great!", ValueError("Test error"), None
        )
        
        # Verify the result
        self.assertFalse(result.success)
        self.assertEqual(result.fallback_level, 3)
        self.assertEqual(result.fallback_strategy, "neutral_default")
        
        # Verify the sentiment result is neutral
        self.assertEqual(result.result.sentiment_score, 0.0)
        self.assertEqual(result.result.interaction_type, "neutral")
        self.assertEqual(result.result.affection_delta, 0)
        self.assertEqual(result.result.confidence, 0.1)  # Very low confidence
    
    # SECTION 13: CONVERSATION HISTORY ANALYZER TESTS
    
    def test_conversation_history_analyzer(self):
        """Test conversation history analyzer"""
        # Create sample conversation history
        history = [
            {"user": "Hello", "assistant": "Hi", "sentiment_score": 0.2, "dominant_emotion": "neutral"},
            {"user": "I'm happy today", "assistant": "That's nice", "sentiment_score": 0.6, "dominant_emotion": "joy"},
            {"user": "I got a promotion", "assistant": "Congratulations", "sentiment_score": 0.7, "dominant_emotion": "joy"},
            {"user": "I'm excited", "assistant": "That's great", "sentiment_score": 0.8, "dominant_emotion": "joy"}
        ]
        
        result = self.conversation_history_analyzer.analyze_conversation_history(history)
        
        # Verify analysis results
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.sentiment_pattern)
        self.assertEqual(result.sentiment_pattern.pattern_type, "escalating")
        self.assertEqual(result.sentiment_pattern.dominant_emotion, "joy")
        self.assertGreater(result.sentiment_stability, 0.5)
        self.assertGreater(result.average_sentiment, 0.5)
    
    # SECTION 14: EDGE CASES AND BOUNDARY CONDITIONS
    
    def test_empty_input_handling(self):
        """Test handling of empty input across all components"""
        empty_text = ""
        
        # Test context analyzer with empty input
        context_result = self.context_analyzer.analyze_context(empty_text)
        self.assertEqual(context_result.dominant_emotion, "neutral")
        self.assertLessEqual(context_result.emotion_confidence, 0.3)
        
        # Test emotion intensity detector with empty input
        intensity_result = self.emotion_intensity_detector.detect_intensity(empty_text)
        self.assertEqual(intensity_result.intensity_category, "mild")
        self.assertLessEqual(intensity_result.intensity_score, 0.1)
        
        # Test sarcasm detector with empty input
        sarcasm_result = self.sarcasm_irony_detector.detect_non_literal_language(empty_text)
        self.assertLess(sarcasm_result.sarcasm_probability, 0.5)
        self.assertLess(sarcasm_result.irony_probability, 0.5)
        self.assertIsNone(sarcasm_result.non_literal_type)
        
        # Test mixed emotion handler with empty input
        emotion_result = self.mixed_emotion_handler.detect_mixed_emotions(empty_text)
        self.assertEqual(emotion_result.dominant_emotion, "neutral")
        self.assertFalse(emotion_result.is_mixed)
    
    def test_extremely_long_input_handling(self):
        """Test handling of extremely long input"""
        # Create a very long input text
        long_text = "I am feeling " + "very " * 100 + "happy today."
        
        # Test context analyzer with long input
        context_result = self.context_analyzer.analyze_context(long_text)
        self.assertIsNotNone(context_result)
        
        # Test emotion intensity detector with long input
        intensity_result = self.emotion_intensity_detector.detect_intensity(long_text)
        self.assertIsNotNone(intensity_result)
        self.assertEqual(intensity_result.intensity_category, "extreme")  # Should detect extreme intensity
        
        # Test sarcasm detector with long input
        sarcasm_result = self.sarcasm_irony_detector.detect_non_literal_language(long_text)
        self.assertIsNotNone(sarcasm_result)
    
    def test_special_characters_handling(self):
        """Test handling of special characters and emojis"""
        special_text = "I'm so happy!!! 😊😊😊 This is AMAZING!!! #blessed"
        
        # Test context analyzer with special characters
        context_result = self.context_analyzer.analyze_context(special_text)
        self.assertEqual(context_result.dominant_emotion, "joy")
        
        # Test emotion intensity detector with special characters
        intensity_result = self.emotion_intensity_detector.detect_intensity(special_text)
        self.assertIn(intensity_result.intensity_category, ["strong", "extreme"])
        
        # Test sarcasm detector with special characters
        sarcasm_result = self.sarcasm_irony_detector.detect_non_literal_language(special_text)
        self.assertIn("emoji_indicators", sarcasm_result.context_indicators)
    
    # SECTION 15: INTEGRATION TESTS
    
    def test_full_sentiment_analysis_pipeline(self):
        """Test the full sentiment analysis pipeline with various inputs"""
        # Test with positive text
        positive_text = "I'm really happy with this. It's excellent!"
        positive_result = self.enhanced_adapter.analyze_user_input(positive_text)
        
        self.assertGreater(positive_result.sentiment_score, 0)
        self.assertGreater(positive_result.affection_delta, 0)
        
        # Test with negative text
        negative_text = "This is terrible. I'm very disappointed."
        negative_result = self.enhanced_adapter.analyze_user_input(negative_text)
        
        self.assertLess(negative_result.sentiment_score, 0)
        self.assertLess(negative_result.affection_delta, 0)
        
        # Test with sarcastic text
        sarcastic_text = "Oh great, another error. This is just wonderful."
        sarcastic_result = self.enhanced_adapter.analyze_user_input(sarcastic_text)
        
        # Should detect sarcasm and adjust sentiment accordingly
        self.assertNotEqual(sarcastic_result.sentiment_score, 0.0)
        
        # Test with mixed emotions
        mixed_text = "I'm happy about the promotion but worried about the new responsibilities."
        mixed_result = self.enhanced_adapter.analyze_user_input(mixed_text)
        
        # Should detect mixed emotions and have moderate impact
        self.assertLess(abs(mixed_result.affection_delta), 5)
    
    def test_sentiment_analysis_with_conversation_history(self):
        """Test sentiment analysis with conversation history integration"""
        # Create conversation history
        history = [
            {"user": "Hello", "assistant": "Hi", "sentiment_score": 0.2, "dominant_emotion": "neutral"},
            {"user": "I'm happy today", "assistant": "That's nice", "sentiment_score": 0.6, "dominant_emotion": "joy"},
            {"user": "I got a promotion", "assistant": "Congratulations", "sentiment_score": 0.7, "dominant_emotion": "joy"}
        ]
        
        # Test with consistent sentiment
        consistent_text = "I'm still feeling great about everything!"
        consistent_result = self.enhanced_adapter.analyze_user_input(consistent_text, history)
        
        # Should strengthen impact due to consistent pattern
        self.assertGreater(consistent_result.affection_delta, 0)
        
        # Test with dramatic shift
        shift_text = "Actually, I just got fired. This is terrible."
        shift_result = self.enhanced_adapter.analyze_user_input(shift_text, history)
        
        # Should smooth the transition
        self.assertLess(shift_result.affection_delta, 0)
        self.assertGreater(shift_result.affection_delta, -10)  # Not too extreme

if __name__ == "__main__":
    unittest.main()