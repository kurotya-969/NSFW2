"""
Unit tests for the sarcasm and irony detector
"""

import unittest
from sarcasm_irony_detector import SarcasmIronyDetector, NonLiteralLanguageResult

class TestSarcasmIronyDetector(unittest.TestCase):
    """Test cases for the SarcasmIronyDetector class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.detector = SarcasmIronyDetector()
    
    def test_sarcasm_detection(self):
        """Test detection of sarcastic statements"""
        # Test exaggerated positive sarcasm
        result = self.detector.detect_non_literal_language("That's so awesome but it completely failed")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        self.assertEqual(result.non_literal_type, "sarcasm")
        self.assertIn("sarcasm:exaggerated_positive", result.detected_patterns)
        
        # Test mock agreement sarcasm
        result = self.detector.detect_non_literal_language("Yeah, sure, whatever you say")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        self.assertEqual(result.non_literal_type, "sarcasm")
        
        # Test rhetorical questions
        result = self.detector.detect_non_literal_language("Could you be any more obvious?")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        self.assertEqual(result.non_literal_type, "sarcasm")
        
        # Test obvious falsehood
        result = self.detector.detect_non_literal_language("Because that's totally how things work")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        self.assertEqual(result.non_literal_type, "sarcasm")
        
        # Test hyperbole sarcasm (new pattern)
        result = self.detector.detect_non_literal_language("This is literally the worst day ever in the history of mankind")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        self.assertEqual(result.non_literal_type, "sarcasm")
        self.assertIn("sarcasm:hyperbole", result.detected_patterns)
    
    def test_irony_detection(self):
        """Test detection of ironic statements"""
        # Test situational irony
        result = self.detector.detect_non_literal_language("Just exactly what I needed today")
        self.assertGreaterEqual(result.irony_probability, 0.5)
        self.assertEqual(result.non_literal_type, "irony")
        self.assertIn("irony:situational_irony", result.detected_patterns)
        
        # Test dramatic irony
        result = self.detector.detect_non_literal_language("If only they knew what was coming")
        self.assertGreaterEqual(result.irony_probability, 0.5)
        self.assertEqual(result.non_literal_type, "irony")
        
        # Test verbal irony
        result = self.detector.detect_non_literal_language("How nice of you to show up an hour late")
        self.assertGreaterEqual(result.irony_probability, 0.5)
        self.assertEqual(result.non_literal_type, "irony")
        
        # Test contrary statements
        result = self.detector.detect_non_literal_language("Great job breaking the build")
        self.assertGreaterEqual(result.irony_probability, 0.5)
        self.assertEqual(result.non_literal_type, "irony")
        
        # Test understated irony (new pattern)
        result = self.detector.detect_non_literal_language("Just a minor inconvenience that caused catastrophic failure")
        self.assertGreaterEqual(result.irony_probability, 0.5)
        self.assertEqual(result.non_literal_type, "irony")
        self.assertIn("irony:understated_irony", result.detected_patterns)
    
    def test_context_indicators(self):
        """Test detection of contextual indicators"""
        # Test punctuation indicators
        result = self.detector.detect_non_literal_language("Sure thing!!!")
        self.assertIn("punctuation", result.context_indicators)
        
        # Test formatting indicators
        result = self.detector.detect_non_literal_language("That was AMAZING")
        self.assertIn("formatting", result.context_indicators)
        
        # Test emoji indicators
        result = self.detector.detect_non_literal_language("Great job ;)")
        self.assertIn("emoji_indicators", result.context_indicators)
        
        # Test phrase indicators
        result = self.detector.detect_non_literal_language("It's great, if you know what I mean")
        self.assertIn("phrase_indicators", result.context_indicators)
        
        # Test tone markers (new indicator)
        result = self.detector.detect_non_literal_language("I love this feature /s")
        self.assertIn("tone_markers", result.context_indicators)
        
        # Test multiple indicators
        result = self.detector.detect_non_literal_language("SURE!!! That's \"great\" work 🙄")
        self.assertGreaterEqual(len(result.context_indicators), 3)
    
    def test_mixed_non_literal_language(self):
        """Test detection of mixed sarcasm and irony"""
        result = self.detector.detect_non_literal_language("Oh wow, perfect timing! Just what I needed when everything is going so well!")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        self.assertGreaterEqual(result.irony_probability, 0.5)
        self.assertEqual(result.non_literal_type, "mixed")
    
    def test_japanese_non_literal_language(self):
        """Test detection of Japanese sarcasm and irony"""
        # Test Japanese sarcasm
        result = self.detector.detect_non_literal_language("素晴らしいですね、また失敗しました")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        self.assertEqual(result.non_literal_type, "sarcasm")
        
        # Test Japanese irony
        result = self.detector.detect_non_literal_language("なんて素敵な失敗でしょう")
        self.assertGreaterEqual(result.irony_probability, 0.5)
        self.assertEqual(result.non_literal_type, "irony")
        
        # Test Japanese with explicit marker
        result = self.detector.detect_non_literal_language("素晴らしい出来栄えですね（皮肉です）")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        self.assertIn("tone_markers", result.context_indicators)
    
    def test_confidence_scoring(self):
        """Test confidence scoring for non-literal language detection"""
        # High confidence case (multiple indicators)
        result = self.detector.detect_non_literal_language("Yeah right!!! That's TOTALLY how it works! ;)")
        self.assertGreaterEqual(result.confidence, 0.7)
        self.assertIsNotNone(result.confidence_factors)
        
        # Medium confidence case (some indicators)
        result = self.detector.detect_non_literal_language("Great job breaking the build")
        self.assertGreaterEqual(result.confidence, 0.4)
        self.assertLessEqual(result.confidence, 0.7)
        
        # Low confidence case (few indicators)
        result = self.detector.detect_non_literal_language("Just what I needed")
        self.assertLessEqual(result.confidence, 0.5)
        
        # Very short text (should have lower confidence)
        result = self.detector.detect_non_literal_language("Great.")
        self.assertLess(result.confidence, 0.4)
        self.assertTrue(result.confidence_factors["length_penalty"] < 0)
        
        # Borderline case (close to threshold)
        # Create a message that will score just around the threshold
        result = self.detector.detect_non_literal_language("Well that's nice I guess")
        if 0.4 <= result.sarcasm_probability <= 0.6:
            # If it's borderline, confidence should be reduced
            threshold_distance = min(
                abs(result.sarcasm_probability - 0.5),
                abs(result.irony_probability - 0.5)
            )
            self.assertLess(threshold_distance, 0.1)
            self.assertLess(result.confidence, 0.6)
    
    def test_non_sarcastic_statements(self):
        """Test that non-sarcastic statements are correctly identified"""
        result = self.detector.detect_non_literal_language("I'm happy with the results")
        self.assertLess(result.sarcasm_probability, 0.5)
        self.assertLess(result.irony_probability, 0.5)
        self.assertIsNone(result.non_literal_type)
        
        result = self.detector.detect_non_literal_language("The weather is nice today")
        self.assertLess(result.sarcasm_probability, 0.5)
        self.assertLess(result.irony_probability, 0.5)
        self.assertIsNone(result.non_literal_type)
        
        # Test longer non-sarcastic text
        result = self.detector.detect_non_literal_language(
            "I really enjoyed the movie we watched yesterday. The plot was interesting and the acting was good."
        )
        self.assertLess(result.sarcasm_probability, 0.5)
        self.assertLess(result.irony_probability, 0.5)
        self.assertIsNone(result.non_literal_type)
    
    def test_contextual_analysis_integration(self):
        """Test integration with contextual analysis"""
        # Test with sentiment contradiction context
        context = {"sentiment_contradiction": True}
        result = self.detector.detect_non_literal_language("That's great", context)
        self.assertGreater(result.sarcasm_probability, 0.0)
        self.assertGreater(result.irony_probability, 0.0)
        
        # Test with sarcasm history context
        context = {"sarcasm_history": 2}
        result = self.detector.detect_non_literal_language("Perfect", context)
        self.assertGreater(result.sarcasm_probability, 0.0)
        
        # Test with sentiment mismatch context (new context type)
        context = {"sentiment_mismatch": True}
        result = self.detector.detect_non_literal_language("I'm so happy right now", context)
        self.assertGreater(result.sarcasm_probability, 0.0)
        
        # Test with topic shift context (new context type)
        context = {"topic_shift": True}
        result = self.detector.detect_non_literal_language("Well that's just perfect", context)
        self.assertGreater(result.sarcasm_probability, 0.0)
    
    def test_contradiction_patterns(self):
        """Test detection of contradiction patterns that might signal sarcasm/irony"""
        # Test sentiment contradiction
        result = self.detector.detect_non_literal_language("I'm so happy I could cry from sadness")
        self.assertIn("contradiction:sentiment_contradiction", result.detected_patterns)
        
        # Test expectation contradiction
        result = self.detector.detect_non_literal_language("I expected it to work but I'm shocked it failed")
        self.assertIn("contradiction:expectation_contradiction", result.detected_patterns)
        
        # Test value contradiction
        result = self.detector.detect_non_literal_language("This is an important feature that's completely worthless")
        self.assertIn("contradiction:value_contradiction", result.detected_patterns)
    
    def test_mixed_emotions_detection(self):
        """Test detection of mixed emotions in non-literal language"""
        # Test with text containing mixed emotions
        result = self.detector.detect_non_literal_language("I'm so happy but also sad about this disaster")
        self.assertIsNotNone(result.mixed_emotions)
        if result.mixed_emotions:
            self.assertIn("joy", result.mixed_emotions)
            self.assertIn("sadness", result.mixed_emotions)
        
        # Test with common mixed emotion phrases
        result = self.detector.detect_non_literal_language("I have such a love-hate relationship with this feature")
        self.assertIsNotNone(result.mixed_emotions)
        if result.mixed_emotions:
            self.assertIn("joy", result.mixed_emotions)
            self.assertIn("anger", result.mixed_emotions)
        
        # Test with laughing and crying
        result = self.detector.detect_non_literal_language("I'm laughing and crying at how bad this is")
        self.assertIsNotNone(result.mixed_emotions)
        if result.mixed_emotions:
            self.assertIn("joy", result.mixed_emotions)
            self.assertIn("sadness", result.mixed_emotions)
        
        # Test with no mixed emotions
        result = self.detector.detect_non_literal_language("This is just plain bad")
        if result.mixed_emotions:
            self.assertLessEqual(len(result.mixed_emotions), 1)
    
    def test_ambiguity_scoring(self):
        """Test ambiguity scoring for non-literal language detection"""
        # Test with highly ambiguous case (both sarcasm and irony, mixed emotions)
        result = self.detector.detect_non_literal_language(
            "I'm so happy yet disappointed with this amazing disaster of a feature"
        )
        self.assertGreater(result.ambiguity_score, 0.0)
        
        # Test with borderline case (close to threshold)
        # Create a message that will likely score around the threshold
        result = self.detector.detect_non_literal_language("Well that's nice I guess")
        threshold_distance = min(
            abs(result.sarcasm_probability - 0.5),
            abs(result.irony_probability - 0.5)
        )
        if threshold_distance < 0.1:
            self.assertGreater(result.ambiguity_score, 0.0)
        
        # Test with competing non-literal types
        result = self.detector.detect_non_literal_language(
            "Just what I needed, a perfect disaster. Great job breaking everything."
        )
        if result.sarcasm_probability >= 0.4 and result.irony_probability >= 0.4:
            self.assertGreater(result.ambiguity_score, 0.0)
        
        # Test with clear-cut case (should have low ambiguity)
        result = self.detector.detect_non_literal_language("Yeah right, that's TOTALLY how it works /s")
        if result.sarcasm_probability > 0.7 and result.irony_probability < 0.3:
            self.assertLess(result.ambiguity_score, 0.3)
    
    def test_conversation_context_impact(self):
        """Test the impact of conversation context on non-literal language detection"""
        # Test with no context
        result = self.detector.detect_non_literal_language("That's great")
        self.assertEqual(result.conversation_context_impact, 0.0)
        
        # Test with sentiment contradiction context
        context = {"sentiment_contradiction": True}
        result = self.detector.detect_non_literal_language("That's great", context)
        self.assertGreater(result.conversation_context_impact, 0.0)
        
        # Test with multiple context factors
        context = {
            "sentiment_contradiction": True,
            "sarcasm_history": 2,
            "sentiment_mismatch": True
        }
        result = self.detector.detect_non_literal_language("Perfect", context)
        self.assertGreater(result.conversation_context_impact, 0.2)
        
        # Test with all context factors
        context = {
            "sentiment_contradiction": True,
            "sarcasm_history": 3,
            "sentiment_mismatch": True,
            "topic_shift": True
        }
        result = self.detector.detect_non_literal_language("Awesome", context)
        self.assertGreater(result.conversation_context_impact, 0.5)
    
    def test_confidence_explanation(self):
        """Test the detailed confidence explanation functionality"""
        # Test with a high confidence case
        result = self.detector.detect_non_literal_language("Yeah right!!! That's TOTALLY how it works! ;)")
        explanation = self.detector.get_confidence_explanation(result)
        
        # Check that explanation contains expected keys
        self.assertIn("overall_confidence", explanation)
        self.assertIn("factors", explanation)
        self.assertIn("threshold_analysis", explanation)
        
        # Check that factors contain expected components
        factors = explanation["factors"]
        self.assertIn("pattern_matches", factors)
        self.assertIn("context_indicators", factors)
        self.assertIn("special_cases", factors)
        
        # Check that threshold analysis contains expected components
        threshold = explanation["threshold_analysis"]
        self.assertIn("sarcasm_distance_from_threshold", threshold)
        self.assertIn("irony_distance_from_threshold", threshold)
        self.assertIn("borderline_case", threshold)
        
        # Test with a borderline case
        result = self.detector.detect_non_literal_language("Well that's nice")
        explanation = self.detector.get_confidence_explanation(result)
        
        # If it's close to the threshold, it should be marked as borderline
        if min(abs(result.sarcasm_probability - 0.5), abs(result.irony_probability - 0.5)) < 0.1:
            self.assertTrue(explanation["threshold_analysis"]["borderline_case"])
    
    def test_get_explanation(self):
        """Test the human-readable explanation functionality"""
        # Test with detected non-literal language
        result = self.detector.detect_non_literal_language("Yeah right!!! That's TOTALLY how it works! ;)")
        explanation = self.detector.get_explanation(result)
        
        # Check that explanation contains key information
        self.assertIn("Detected", explanation)
        self.assertIn("confidence", explanation)
        self.assertIn("Sarcasm probability", explanation)
        self.assertIn("Irony probability", explanation)
        
        if result.detected_patterns:
            self.assertIn("Detected patterns", explanation)
        
        if result.context_indicators:
            self.assertIn("Context indicators", explanation)
        
        # Test with no detected non-literal language
        result = self.detector.detect_non_literal_language("The weather is nice today")
        explanation = self.detector.get_explanation(result)
        self.assertIn("No significant non-literal language detected", explanation)
    
    def test_edge_cases(self):
        """Test edge cases for sarcasm and irony detection"""
        # Test empty string
        result = self.detector.detect_non_literal_language("")
        self.assertLess(result.confidence, 0.4)  # Should have low confidence
        
        # Test very short text
        result = self.detector.detect_non_literal_language("Sure.")
        self.assertTrue(result.confidence_factors["length_penalty"] < 0)
        
        # Test with only emoji
        result = self.detector.detect_non_literal_language("🙄")
        self.assertIn("emoji_indicators", result.context_indicators)
        
        # Test with mixed languages
        result = self.detector.detect_non_literal_language("That's so great 素晴らしい but it failed 失敗")
        self.assertGreaterEqual(result.sarcasm_probability, 0.5)
        
        # Test with unusual formatting
        result = self.detector.detect_non_literal_language("t H a T   w A s   *so*   ~amazing~")
        self.assertIn("formatting", result.context_indicators)

if __name__ == "__main__":
    unittest.main()