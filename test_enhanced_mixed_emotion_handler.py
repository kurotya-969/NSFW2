"""
Enhanced unit tests for the Mixed Emotion Handler
Tests the improved implementation for identifying and weighing multiple emotions
"""

import unittest
from mixed_emotion_handler import MixedEmotionHandler, EmotionCategory, MixedEmotionResult

class TestEnhancedMixedEmotionHandler(unittest.TestCase):
    """Enhanced test cases for the Mixed Emotion Handler"""
    
    def setUp(self):
        """Set up the test environment"""
        self.handler = MixedEmotionHandler()
    
    def test_emotion_weighing(self):
        """Test weighing of multiple emotions in a message"""
        # Test with multiple emotions with different weights
        result = self.handler.detect_mixed_emotions("I'm extremely happy but also a bit worried about the new job.")
        
        # Check that both emotions are detected
        self.assertIn("joy", result.emotions)
        self.assertIn("fear", result.emotions)
        
        # Check that joy has higher weight due to "extremely" modifier
        self.assertGreater(result.emotions["joy"], result.emotions["fear"])
        
        # Test with balanced emotions
        result = self.handler.detect_mixed_emotions("I'm equally happy and sad about this situation.")
        
        # Check that both emotions have similar weights
        self.assertIn("joy", result.emotions)
        self.assertIn("sadness", result.emotions)
        self.assertAlmostEqual(result.emotions["joy"], result.emotions["sadness"], delta=0.2)
    
    def test_negation_handling(self):
        """Test handling of negated emotions"""
        # Test with negated positive emotion
        result = self.handler.detect_mixed_emotions("I'm not happy about this decision.")
        
        # Check that negation reduces the positive emotion score
        if "joy" in result.emotions:
            self.assertLess(result.emotions["joy"], 0.5)
        
        # Check that negation might add some negative emotion
        self.assertGreater(result.emotion_ratio["negative"], 0.3)
        
        # Test with negated negative emotion
        result = self.handler.detect_mixed_emotions("I'm not sad about leaving, I'm excited for the new opportunity.")
        
        # Check that positive emotion dominates
        self.assertEqual(result.emotion_category, EmotionCategory.POSITIVE)
        self.assertGreater(result.emotion_ratio["positive"], result.emotion_ratio["negative"])
    
    def test_emotion_intensity_detection(self):
        """Test detection of emotion intensity"""
        # Test with high intensity emotion
        result = self.handler.detect_mixed_emotions("I'm extremely happy about the promotion!")
        
        # Check that joy has high score
        self.assertGreater(result.emotions["joy"], 0.7)
        
        # Test with low intensity emotion
        result = self.handler.detect_mixed_emotions("I'm a bit worried about the meeting.")
        
        # Check that fear has moderate score
        if "fear" in result.emotions:
            self.assertLess(result.emotions["fear"], 0.7)
    
    def test_dominant_emotion_determination(self):
        """Test determination of dominant emotional tone"""
        # Test with clear dominant emotion
        result = self.handler.detect_mixed_emotions("I'm extremely happy and a little bit excited.")
        
        self.assertEqual(result.dominant_emotion, "joy")
        self.assertEqual(result.secondary_emotion, "anticipation")
        self.assertGreater(result.emotion_confidence, 0.7)
        
        # Test with competing emotions
        result = self.handler.detect_mixed_emotions("I'm happy but also quite worried.")
        
        # Either joy or fear could be dominant, but confidence should be lower
        self.assertIn(result.dominant_emotion, ["joy", "fear"])
        self.assertIsNotNone(result.secondary_emotion)
        self.assertLess(result.emotion_confidence, 0.9)
        
        # Test with multiple similar emotions
        result = self.handler.detect_mixed_emotions("I'm happy, excited, and looking forward to the event!")
        
        # All positive emotions, so confidence should still be decent
        self.assertIn(result.dominant_emotion, ["joy", "anticipation"])
        self.assertGreater(result.emotion_confidence, 0.6)
    
    def test_complex_mixed_emotions(self):
        """Test handling of complex mixed emotions"""
        # Test with multiple conflicting emotions
        result = self.handler.detect_mixed_emotions(
            "I'm happy about the promotion but sad about leaving my team, and also anxious about the new responsibilities."
        )
        
        # Should detect mixed emotions
        self.assertTrue(result.is_mixed)
        self.assertTrue(result.conflicting_emotions)
        
        # Should have high complexity
        self.assertGreater(result.emotion_complexity, 0.5)
        
        # Should have moderate ambivalence
        self.assertGreater(result.emotion_ambivalence, 0.3)
        
        # Should detect at least 3 emotions
        self.assertGreaterEqual(len([e for e, s in result.emotions.items() if s > 0.1]), 3)
    
    def test_subtle_mixed_emotions(self):
        """Test detection of subtle mixed emotions"""
        # Test with subtle conflicting emotions
        result = self.handler.detect_mixed_emotions(
            "The news made me smile, though I can't help feeling a sense of loss."
        )
        
        # Should detect mixed emotions despite lack of explicit emotion words
        self.assertTrue(result.is_mixed)
        
        # Should have moderate confidence
        self.assertLess(result.emotion_confidence, 0.9)
    
    def test_emotion_category_determination(self):
        """Test determination of overall emotion category"""
        # Test with predominantly positive emotions
        result = self.handler.detect_mixed_emotions("I'm happy and excited, with just a touch of nervousness.")
        self.assertEqual(result.emotion_category, EmotionCategory.POSITIVE)
        
        # Test with predominantly negative emotions
        result = self.handler.detect_mixed_emotions("I'm sad and worried, though I try to stay hopeful.")
        self.assertEqual(result.emotion_category, EmotionCategory.NEGATIVE)
        
        # Test with balanced positive and negative emotions
        result = self.handler.detect_mixed_emotions("I'm equally happy and sad about this situation.")
        self.assertEqual(result.emotion_category, EmotionCategory.AMBIVALENT)
        
        # Test with neutral emotions
        result = self.handler.detect_mixed_emotions("I'm feeling calm and neutral about this.")
        self.assertEqual(result.emotion_category, EmotionCategory.NEUTRAL)
    
    def test_affection_impact_for_mixed_emotions(self):
        """Test affection impact calculation for mixed emotions"""
        # Test with mixed but predominantly positive emotions
        result = self.handler.detect_mixed_emotions("I'm very happy though slightly nervous.")
        impact = self.handler.get_affection_impact(result)
        
        # Should have positive but reduced impact
        self.assertGreater(impact["sentiment_score"], 0)
        self.assertGreater(impact["affection_delta"], 0)
        
        # Test with mixed but predominantly negative emotions
        result = self.handler.detect_mixed_emotions("I'm very sad though slightly hopeful.")
        impact = self.handler.get_affection_impact(result)
        
        # Should have negative but reduced impact
        self.assertLess(impact["sentiment_score"], 0)
        self.assertLess(impact["affection_delta"], 0)
        
        # Test with highly ambivalent emotions
        result = self.handler.detect_mixed_emotions("I'm equally happy and sad about this.")
        impact = self.handler.get_affection_impact(result)
        
        # Should have minimal impact due to ambivalence
        self.assertAlmostEqual(impact["sentiment_score"], 0, delta=0.3)
        self.assertIn(impact["affection_delta"], [-1, 0, 1])  # Should be close to zero
    
    def test_contextual_modifiers(self):
        """Test detection and application of contextual modifiers"""
        # Test with intensifiers
        result = self.handler.detect_mixed_emotions("I'm extremely happy about the news.")
        self.assertIn("extremely", result.contextual_modifiers["intensifier"])
        
        # Test with diminishers
        result = self.handler.detect_mixed_emotions("I'm a bit worried about the situation.")
        self.assertIn("a bit", result.contextual_modifiers["diminisher"])
        
        # Test with negators
        result = self.handler.detect_mixed_emotions("I'm not happy about this decision.")
        self.assertIn("not", result.contextual_modifiers["negator"])
        
        # Test with uncertainty markers
        result = self.handler.detect_mixed_emotions("I'm maybe a little excited about it.")
        self.assertIn("maybe", result.contextual_modifiers["uncertainty"])
        
        # Test with certainty markers
        result = self.handler.detect_mixed_emotions("I'm definitely happy about this.")
        self.assertIn("definitely", result.contextual_modifiers["certainty"])
    
    def test_emotion_weights(self):
        """Test calculation of emotion weights based on context"""
        # Test with intensifiers
        result = self.handler.detect_mixed_emotions("I'm extremely happy about the news.")
        self.assertGreater(result.emotion_weights["joy"], result.emotions["joy"])
        
        # Test with diminishers
        result = self.handler.detect_mixed_emotions("I'm a bit worried about the situation.")
        if "fear" in result.emotions and "fear" in result.emotion_weights:
            self.assertLess(result.emotion_weights["fear"], result.emotions["fear"] * 1.1)  # Allow for normalization effects
    
    def test_edge_cases(self):
        """Test edge cases for mixed emotion handling"""
        # Test with empty text
        result = self.handler.detect_mixed_emotions("")
        self.assertEqual(result.emotion_category, EmotionCategory.NEUTRAL)
        
        # Test with text containing no emotion words
        result = self.handler.detect_mixed_emotions("The sky is blue and the grass is green.")
        self.assertEqual(result.emotion_category, EmotionCategory.NEUTRAL)
        
        # Test with text containing only intensifiers but no emotion words
        result = self.handler.detect_mixed_emotions("Very extremely incredibly.")
        self.assertEqual(result.emotion_category, EmotionCategory.NEUTRAL)
        
        # Test with contradictory statements
        result = self.handler.detect_mixed_emotions("I'm happy. I'm sad.")
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

if __name__ == "__main__":
    unittest.main()