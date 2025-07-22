"""
Test script for context analyzer
"""

import unittest
from context_analyzer import ContextAnalyzer, ContextualAnalysis

class TestContextAnalyzer(unittest.TestCase):
    """Test cases for the ContextAnalyzer class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.analyzer = ContextAnalyzer()
    
    def test_text_preprocessing(self):
        """Test that text preprocessing normalizes input correctly"""
        # Test with mixed case and extra whitespace
        text = "  This is A TEST with MIXED case  "
        normalized = self.analyzer._preprocess_text(text)
        self.assertEqual(normalized, "this is a test with mixed case")
        
        # Test with special characters
        text = "Hello! How are you? I'm fine."
        normalized = self.analyzer._preprocess_text(text)
        self.assertEqual(normalized, "hello how are you i'm fine")
        
        # Test with Japanese text
        text = "こんにちは、元気ですか？"
        normalized = self.analyzer._preprocess_text(text)
        self.assertEqual(normalized, "こんにちは 元気ですか")
    
    def test_extract_emotional_context(self):
        """Test extraction of emotional context beyond keywords"""
        # Test with clear emotional content
        result = self.analyzer.detect_emotional_context("I am so happy today!")
        self.assertIn("joy", result)
        self.assertGreater(result["joy"], 0.5)
        
        # Test with mixed emotions
        result = self.analyzer.detect_emotional_context("I'm happy but also worried about tomorrow")
        self.assertIn("joy", result)
        self.assertIn("fear", result)
        
        # Test with Japanese text
        result = self.analyzer.detect_emotional_context("とても嬉しいです！")
        self.assertIn("joy", result)
        self.assertGreater(result["joy"], 0.5)
        
        # Test with neutral text
        result = self.analyzer.detect_emotional_context("The sky is blue")
        self.assertEqual(result.get("neutral", 0.5), 0.5)
    
    def test_analyze_context(self):
        """Test the full context analysis functionality"""
        # Test with positive emotional context
        result = self.analyzer.analyze_context("I'm really happy to talk with you today!")
        self.assertEqual(result.dominant_emotion, "joy")
        self.assertGreater(result.emotion_confidence, 0.5)
        self.assertIn("really", result.contextual_modifiers)
        
        # Test with negative emotional context
        result = self.analyzer.analyze_context("I'm feeling very sad and disappointed")
        self.assertEqual(result.dominant_emotion, "sadness")
        self.assertGreater(result.emotion_confidence, 0.5)
        self.assertIn("very", result.contextual_modifiers)
        
        # Test with Japanese text
        result = self.analyzer.analyze_context("アニメについてとても楽しく話せて嬉しいです")
        self.assertEqual(result.dominant_emotion, "joy")
        self.assertIn("anime", result.detected_topics)
        self.assertIn("とても", result.contextual_modifiers)
    
    def test_detect_topics(self):
        """Test topic detection in text"""
        # Test with anime topic
        topics = self.analyzer._detect_topics("I love watching anime and reading manga")
        self.assertIn("anime", topics)
        
        # Test with food topic
        topics = self.analyzer._detect_topics("This ramen restaurant has delicious food")
        self.assertIn("food", topics)
        
        # Test with multiple topics
        topics = self.analyzer._detect_topics("I use my smartphone to read manga and find good restaurants")
        self.assertIn("anime", topics)
        self.assertIn("food", topics)
        self.assertIn("technology", topics)
        
        # Test with Japanese text
        topics = self.analyzer._detect_topics("新しいアニメを見ながら美味しいラーメンを食べました")
        self.assertIn("anime", topics)
        self.assertIn("food", topics)
    
    def test_contextual_modifiers(self):
        """Test detection of contextual modifiers"""
        # Test with intensifiers
        modifiers = self.analyzer._extract_contextual_modifiers("I am very happy")
        self.assertIn("very", modifiers)
        
        # Test with diminishers
        modifiers = self.analyzer._extract_contextual_modifiers("I am somewhat concerned")
        self.assertIn("somewhat", modifiers)
        
        # Test with negators
        modifiers = self.analyzer._extract_contextual_modifiers("I am not happy")
        self.assertIn("not", modifiers)
        
        # Test with Japanese modifiers
        modifiers = self.analyzer._extract_contextual_modifiers("とても嬉しいです")
        self.assertIn("とても", modifiers)
        
        modifiers = self.analyzer._extract_contextual_modifiers("全然嬉しくない")
        self.assertIn("ない", modifiers)
    
    def test_non_literal_explanation(self):
        """Test the non-literal language explanation functionality"""
        # Test with sarcastic text
        analysis = self.analyzer.analyze_context("Yeah right!!! That's TOTALLY how it works! ;)")
        explanation = self.analyzer.get_non_literal_explanation(analysis)
        
        # Check that explanation contains expected keys
        self.assertIn("sarcasm_detected", explanation)
        self.assertIn("irony_detected", explanation)
        self.assertIn("confidence", explanation)
        self.assertIn("confidence_level", explanation)
        self.assertIn("explanation", explanation)
        self.assertIn("affection_impact", explanation)
        self.assertIn("recommendations", explanation)
        
        # Check that sarcasm is detected
        self.assertTrue(explanation["sarcasm_detected"])
        
        # Check that confidence level is set
        self.assertIn(explanation["confidence_level"], ["low", "medium", "high"])
        
        # Check that recommendations are provided
        self.assertIsInstance(explanation["recommendations"], list)
        
        # Test with non-sarcastic text
        analysis = self.analyzer.analyze_context("I'm happy with the results")
        explanation = self.analyzer.get_non_literal_explanation(analysis)
        
        # Check that sarcasm is not detected
        self.assertFalse(explanation["sarcasm_detected"])
        
        # Test with ironic text
        analysis = self.analyzer.analyze_context("Just exactly what I needed today")
        explanation = self.analyzer.get_non_literal_explanation(analysis)
        
        # Check that irony is detected
        self.assertTrue(explanation["irony_detected"])
        
        # Test with mixed non-literal language
        analysis = self.analyzer.analyze_context("Oh wow, perfect timing! Just what I needed when everything is going so well!")
        explanation = self.analyzer.get_non_literal_explanation(analysis)
        
        # Check that both sarcasm and irony are detected
        self.assertTrue(explanation["sarcasm_detected"])
        self.assertTrue(explanation["irony_detected"])
        self.assertEqual(explanation["non_literal_type"], "mixed")

if __name__ == "__main__":
    unittest.main()