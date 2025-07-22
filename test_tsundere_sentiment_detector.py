"""
Unit tests for the Tsundere Sentiment Detector
"""

import unittest
from unittest.mock import patch, MagicMock
from tsundere_sentiment_detector import TsundereSentimentDetector, TsundereAnalysisResult, SentimentLoopData

class TestTsundereSentimentDetector(unittest.TestCase):
    """Test cases for the TsundereSentimentDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = TsundereSentimentDetector()
    
    def test_detect_tsundere_expressions(self):
        """Test detection of tsundere expressions"""
        # Test dismissive affection pattern
        result = self.detector.detect_tsundere_expressions("別にあんたのことが好きなわけじゃないんだからね")
        self.assertTrue(result.is_tsundere)
        self.assertIn("tsundere:dismissive_affection", result.detected_patterns)
        self.assertGreater(result.tsundere_confidence, 0.5)
        
        # Test hostile care pattern
        result = self.detector.detect_tsundere_expressions("うるさいな、心配してるんだよ")
        self.assertTrue(result.is_tsundere)
        self.assertIn("tsundere:hostile_care", result.detected_patterns)
        self.assertGreater(result.tsundere_confidence, 0.5)
        
        # Test reluctant gratitude pattern
        result = self.detector.detect_tsundere_expressions("別にありがとうとか思ってないからね")
        self.assertTrue(result.is_tsundere)
        self.assertIn("tsundere:reluctant_gratitude", result.detected_patterns)
        self.assertGreater(result.tsundere_confidence, 0.5)
        
        # Test insult affection pattern
        result = self.detector.detect_tsundere_expressions("バカ…好きだよ")
        self.assertTrue(result.is_tsundere)
        self.assertIn("tsundere:insult_affection", result.detected_patterns)
        self.assertGreater(result.tsundere_confidence, 0.5)
        
        # Test non-tsundere expression
        result = self.detector.detect_tsundere_expressions("今日はいい天気ですね")
        self.assertFalse(result.is_tsundere)
        self.assertEqual(len(result.detected_patterns), 0)
        self.assertLess(result.tsundere_confidence, 0.5)
    
    def test_classify_farewell_phrases(self):
        """Test classification of farewell phrases"""
        # Test Japanese casual farewell
        result = self.detector.classify_farewell_phrases("じゃあな")
        self.assertTrue(result.is_farewell)
        self.assertEqual(result.farewell_type, "casual")
        self.assertEqual(result.cultural_context, "japanese")
        self.assertTrue(result.is_conversation_ending)
        self.assertTrue(result.is_tsundere)
        
        # Test another Japanese casual farewell
        result = self.detector.classify_farewell_phrases("またな")
        self.assertTrue(result.is_farewell)
        self.assertEqual(result.farewell_type, "casual")
        self.assertEqual(result.cultural_context, "japanese")
        self.assertTrue(result.is_conversation_ending)
        self.assertTrue(result.is_tsundere)
        
        # Test Japanese formal farewell
        result = self.detector.classify_farewell_phrases("さようなら")
        self.assertTrue(result.is_farewell)
        self.assertEqual(result.farewell_type, "formal")
        self.assertEqual(result.cultural_context, "japanese")
        self.assertTrue(result.is_conversation_ending)
        self.assertFalse(result.is_tsundere)
        
        # Test English casual farewell
        result = self.detector.classify_farewell_phrases("see ya")
        self.assertTrue(result.is_farewell)
        self.assertEqual(result.farewell_type, "casual")
        self.assertEqual(result.cultural_context, "english")
        self.assertTrue(result.is_conversation_ending)
        self.assertFalse(result.is_tsundere)
        
        # Test non-farewell phrase
        result = self.detector.classify_farewell_phrases("こんにちは")
        self.assertFalse(result.is_farewell)
        self.assertIsNone(result.farewell_type)
        self.assertIsNone(result.cultural_context)
        self.assertFalse(result.is_conversation_ending)
        self.assertFalse(result.is_tsundere)
    
    @patch('tsundere_sentiment_detector.ContextSentimentDetector')
    def test_detect_sentiment_loop(self, mock_context_detector):
        """Test detection of sentiment loops"""
        # Set up mock
        mock_context_detector_instance = MagicMock()
        mock_context_detector.return_value = mock_context_detector_instance
        
        # Create detector with mock
        detector = TsundereSentimentDetector()
        
        # Test farewell loop detection
        result = detector.detect_sentiment_loop("test_session", "じゃあな")
        self.assertFalse(result.loop_detected)  # First occurrence shouldn't trigger loop detection
        
        result = detector.detect_sentiment_loop("test_session", "じゃあな")
        self.assertTrue(result.loop_detected)  # Second occurrence should trigger loop detection
        self.assertIn("repeated_farewell", result.repeated_patterns)
        self.assertGreater(result.loop_severity, 0.5)
        self.assertEqual(result.loop_duration, 2)
        
        # Test repeated phrase detection
        detector = TsundereSentimentDetector()  # Reset detector
        
        result = detector.detect_sentiment_loop("test_session2", "うるさい")
        self.assertFalse(result.loop_detected)
        
        result = detector.detect_sentiment_loop("test_session2", "うるさい")
        self.assertFalse(result.loop_detected)
        
        result = detector.detect_sentiment_loop("test_session2", "うるさい")
        self.assertTrue(result.loop_detected)
        self.assertIn("repeated_phrase", result.repeated_patterns)
        self.assertGreater(result.loop_severity, 0.5)
        self.assertEqual(result.loop_duration, 3)
    
    @patch('tsundere_sentiment_detector.ContextSentimentDetector')
    def test_analyze_with_tsundere_awareness(self, mock_context_detector):
        """Test tsundere-aware sentiment analysis"""
        # Set up mock
        mock_context_detector_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.adjusted_sentiment_score = -0.5
        mock_result.adjusted_affection_delta = -5
        mock_context_detector_instance.analyze_with_context.return_value = mock_result
        mock_context_detector.return_value = mock_context_detector_instance
        
        # Create detector with mock
        detector = TsundereSentimentDetector()
        detector.context_sentiment_detector = mock_context_detector_instance
        
        # Test tsundere expression adjustment
        result = detector.analyze_with_tsundere_awareness("別にあんたのことが好きなわけじゃないんだからね", "test_session")
        self.assertGreater(result["final_sentiment_score"], -0.5)  # Should be adjusted positively
        self.assertGreater(result["final_affection_delta"], -5)  # Should be adjusted positively
        
        # Test farewell phrase adjustment
        result = detector.analyze_with_tsundere_awareness("じゃあな", "test_session")
        self.assertGreater(result["final_sentiment_score"], -0.5)  # Should be adjusted positively
        self.assertGreaterEqual(result["final_affection_delta"], -5)  # Should be adjusted positively
        
        # Test sentiment loop circuit breaker
        # First, set up the loop detection
        detector.sentiment_loop_history["test_session2"] = {
            "negative_turns": 3,
            "farewell_count": 0,
            "repeated_phrases": {"うるさい": 3},
            "last_phrases": ["うるさい", "うるさい", "うるさい"],
            "intervention_applied": False
        }
        
        result = detector.analyze_with_tsundere_awareness("うるさい", "test_session2")
        self.assertGreater(result["final_sentiment_score"], -0.5)  # Should be adjusted positively
        self.assertGreater(result["final_affection_delta"], -5)  # Should be adjusted positively
    
    def test_get_enhanced_prompt(self):
        """Test enhancement of LLM prompts with tsundere awareness"""
        base_prompt = "Base system prompt"
        
        # Test with tsundere farewell context
        tsundere_context = {
            "tsundere_detected": True,
            "tsundere_confidence": 0.8,
            "is_farewell": True,
            "farewell_type": "casual",
            "is_conversation_ending": True,
            "farewell_guidance": "This is a casual tsundere-style farewell phrase."
        }
        
        enhanced_prompt = self.detector.get_enhanced_prompt(base_prompt, tsundere_context)
        self.assertIn("Base system prompt", enhanced_prompt)
        self.assertIn("Tsundere Expression Handling", enhanced_prompt)
        self.assertIn("Farewell Phrase Detected", enhanced_prompt)
        
        # Test with sentiment loop context
        tsundere_context = {
            "tsundere_detected": True,
            "tsundere_confidence": 0.8,
            "sentiment_loop_detected": True,
            "loop_severity": 0.8,
            "loop_guidance": "The conversation appears to be stuck in a farewell loop.",
            "suggested_intervention": "reset_farewell_context"
        }
        
        enhanced_prompt = self.detector.get_enhanced_prompt(base_prompt, tsundere_context)
        self.assertIn("Base system prompt", enhanced_prompt)
        self.assertIn("Tsundere Expression Handling", enhanced_prompt)
        self.assertIn("Sentiment Loop Detected", enhanced_prompt)
        self.assertIn("Avoid interpreting farewell phrases as genuine", enhanced_prompt)
        
        # Test with non-tsundere context
        tsundere_context = {
            "tsundere_detected": False
        }
        
        enhanced_prompt = self.detector.get_enhanced_prompt(base_prompt, tsundere_context)
        self.assertEqual(enhanced_prompt, base_prompt)  # Should return unchanged

if __name__ == '__main__':
    unittest.main()