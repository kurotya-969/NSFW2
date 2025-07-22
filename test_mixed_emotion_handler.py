"""
Unit tests for the Mixed Emotion Handler
"""

import unittest
from mixed_emotion_handler import MixedEmotionHandler, EmotionCategory, MixedEmotionResult

class TestMixedEmotionHandler(unittest.TestCase):
    """Test cases for the Mixed Emotion Handler"""
    
    def setUp(self):
        """Set up the test environment"""
        self.handler = MixedEmotionHandler()
    
    def test_single_emotion_detection(self):
        """Test detection of a single emotion"""
        # Test with a simple happy message
        result = self.handler.detect_mixed_emotions("I am so happy today!")
        
        self.assertEqual(result.dominant_emotion, "joy")
        self.assertFalse(result.is_mixed)
        self.assertEqual(result.emotion_category, EmotionCategory.POSITIVE)
        self.assertGreater(result.emotion_ratio["positive"], 0.8)
        self.assertLess(result.emotion_complexity, 0.3)
        
        # Test with a simple sad message
        result = self.handler.detect_mixed_emotions("I feel so sad and depressed.")
        
        self.assertEqual(result.dominant_emotion, "sadness")
        self.assertFalse(result.is_mixed)
        self.assertEqual(result.emotion_category, EmotionCategory.NEGATIVE)
        self.assertGreater(result.emotion_ratio["negative"], 0.8)
    
    def test_explicit_mixed_emotions(self):
        """Test detection of explicitly stated mixed emotions"""
        # Test with explicit mixed emotions
        result = self.handler.detect_mixed_emotions("I'm happy but also sad about this situation.")
        
        self.assertTrue(result.is_mixed)
        self.assertTrue(result.conflicting_emotions)
        self.assertEqual(result.emotion_category, EmotionCategory.AMBIVALENT)
        self.assertGreater(result.emotion_ambivalence, 0.5)
        
        # Check that both emotions are detected
        self.assertIn("joy", result.emotions)
        self.assertIn("sadness", result.emotions)
        
        # Test with bittersweet emotions
        result = self.handler.detect_mixed_emotions("It's a bittersweet moment for me.")
        
        self.assertTrue(result.is_mixed)
        self.assertTrue(result.conflicting_emotions)
    
    def test_japanese_mixed_emotions(self):
        """Test detection of mixed emotions in Japanese text"""
        # Test with Japanese mixed emotions
        result = self.handler.detect_mixed_emotions("嬉しいけど悲しい気持ちです。")
        
        self.assertTrue(result.is_mixed)
        self.assertTrue(result.conflicting_emotions)
        self.assertEqual(result.emotion_category, EmotionCategory.AMBIVALENT)
        
        # Test with Japanese complex feelings
        result = self.handler.detect_mixed_emotions("この状況について複雑な気持ちです。")
        
        self.assertTrue(result.is_mixed)
    
    def test_emotion_transition(self):
        """Test detection of emotional transitions"""
        # Test with emotional transition
        result = self.handler.detect_mixed_emotions("I started feeling happy about the news, but then I got angry when I realized the implications.")
        
        self.assertTrue(result.is_mixed)
        self.assertIn("joy", result.emotions)
        self.assertIn("anger", result.emotions)
    
    def test_multiple_non_conflicting_emotions(self):
        """Test detection of multiple non-conflicting emotions"""
        # Test with multiple positive emotions
        result = self.handler.detect_mixed_emotions("I'm happy and excited about the upcoming event!")
        
        self.assertTrue(result.is_mixed)  # Multiple emotions, but not conflicting
        self.assertFalse(result.conflicting_emotions)
        self.assertEqual(result.emotion_category, EmotionCategory.POSITIVE)
        
        # Both emotions should be detected
        self.assertIn("joy", result.emotions)
        self.assertIn("anticipation", result.emotions)
    
    def test_emotion_complexity(self):
        """Test calculation of emotion complexity"""
        # Test with simple emotion
        result = self.handler.detect_mixed_emotions("I'm happy.")
        self.assertLess(result.emotion_complexity, 0.3)
        
        # Test with moderately complex emotions
        result = self.handler.detect_mixed_emotions("I'm happy, excited, and looking forward to the event!")
        self.assertGreater(result.emotion_complexity, 0.3)
        
        # Test with very complex emotions
        result = self.handler.detect_mixed_emotions("I'm happy, excited, but also nervous, anxious, and a bit sad about leaving.")
        self.assertGreater(result.emotion_complexity, 0.7)
    
    def test_emotion_ambivalence(self):
        """Test calculation of emotion ambivalence"""
        # Test with non-ambivalent emotions
        result = self.handler.detect_mixed_emotions("I'm happy and excited!")
        self.assertLess(result.emotion_ambivalence, 0.3)
        
        # Test with slightly ambivalent emotions
        result = self.handler.detect_mixed_emotions("I'm mostly happy but a little worried.")
        self.assertGreater(result.emotion_ambivalence, 0.2)
        
        # Test with highly ambivalent emotions
        result = self.handler.detect_mixed_emotions("I'm equally happy and sad about this situation.")
        self.assertGreater(result.emotion_ambivalence, 0.7)
    
    def test_affection_impact(self):
        """Test calculation of affection impact"""
        # Test with positive emotions
        result = self.handler.detect_mixed_emotions("I'm so happy and excited!")
        impact = self.handler.get_affection_impact(result)
        
        self.assertGreater(impact["sentiment_score"], 0)
        self.assertGreater(impact["affection_delta"], 0)
        self.assertGreater(impact["confidence"], 0.7)
        
        # Test with negative emotions
        result = self.handler.detect_mixed_emotions("I'm so sad and angry!")
        impact = self.handler.get_affection_impact(result)
        
        self.assertLess(impact["sentiment_score"], 0)
        self.assertLess(impact["affection_delta"], 0)
        self.assertGreater(impact["confidence"], 0.7)
        
        # Test with ambivalent emotions
        result = self.handler.detect_mixed_emotions("I'm happy but also sad about this.")
        impact = self.handler.get_affection_impact(result)
        
        # Impact should be reduced for ambivalent emotions
        self.assertLess(abs(impact["sentiment_score"]), 0.5)
        self.assertLess(abs(impact["affection_delta"]), 3)
        self.assertLess(impact["confidence"], 0.8)
    
    def test_emotion_phrases_detection(self):
        """Test detection of emotion phrases"""
        # Test with multiple emotion phrases
        result = self.handler.detect_mixed_emotions("I'm happy about the promotion but worried about the new responsibilities.")
        
        # Check that phrases are detected
        self.assertGreaterEqual(len(result.detected_emotion_phrases), 2)
        
        # Check that the correct emotions are associated with phrases
        phrases_dict = dict(result.detected_emotion_phrases)
        self.assertIn("happy", phrases_dict)
        self.assertEqual(phrases_dict["happy"], "joy")
        self.assertIn("worried", phrases_dict)
        self.assertEqual(phrases_dict["worried"], "fear")
    
    def test_explanation_generation(self):
        """Test generation of human-readable explanations"""
        # Test with mixed emotions
        result = self.handler.detect_mixed_emotions("I'm happy but also sad about this situation.")
        explanation = self.handler.get_explanation(result)
        
        # Check that the explanation contains key information
        self.assertIn("Mixed emotions", explanation)
        self.assertIn("joy", explanation)
        self.assertIn("sadness", explanation)
        self.assertIn("ambivalent", explanation.lower())
        
        # Test with single emotion
        result = self.handler.detect_mixed_emotions("I'm very happy today!")
        explanation = self.handler.get_explanation(result)
        
        # Check that the explanation contains key information
        self.assertIn("Single dominant emotion", explanation)
        self.assertIn("joy", explanation)
        self.assertIn("positive", explanation.lower())

if __name__ == "__main__":
    unittest.main()