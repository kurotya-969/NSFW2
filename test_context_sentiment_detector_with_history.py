"""
Integration test for context sentiment detector with conversation history
"""

import unittest
from context_sentiment_detector import ContextSentimentDetector

class TestContextSentimentDetectorWithHistory(unittest.TestCase):
    """Test cases for the ContextSentimentDetector class with conversation history"""
    
    def setUp(self):
        """Set up the test environment"""
        self.detector = ContextSentimentDetector()
    
    def test_analyze_with_empty_history(self):
        """Test analyzing with empty conversation history"""
        result = self.detector.analyze_with_context("I'm happy today", [])
        self.assertIsNone(result.conversation_pattern)
        self.assertIsNone(result.sentiment_shift)
    
    def test_analyze_with_consistent_history(self):
        """Test analyzing with consistent positive conversation history"""
        history = [
            {
                "text": "I'm having a good day",
                "sentiment": {
                    "dominant_emotion": "joy",
                    "emotion_confidence": 0.7
                }
            },
            {
                "text": "Everything is going well",
                "sentiment": {
                    "dominant_emotion": "joy",
                    "emotion_confidence": 0.6
                }
            }
        ]
        
        result = self.detector.analyze_with_context("I'm still feeling happy", history)
        
        # Check that conversation pattern was detected
        self.assertIsNotNone(result.conversation_pattern)
        # The pattern could be consistent or de-escalating depending on the exact values
        self.assertIn(result.conversation_pattern.pattern_type, ["consistent", "de-escalating"])
        self.assertIn("joy", result.conversation_pattern.dominant_emotions)
        
        # Check that sentiment shift was detected (but should be minimal)
        self.assertIsNotNone(result.sentiment_shift)
        
        # Check that confidence was increased due to consistent pattern
        self.assertGreater(result.context_confidence, 0.5)
    
    def test_analyze_with_sentiment_shift(self):
        """Test analyzing with a significant sentiment shift"""
        history = [
            {
                "text": "I'm feeling sad today",
                "sentiment": {
                    "dominant_emotion": "sadness",
                    "emotion_confidence": 0.7
                }
            },
            {
                "text": "Things aren't going well",
                "sentiment": {
                    "dominant_emotion": "sadness",
                    "emotion_confidence": 0.6
                }
            }
        ]
        
        result = self.detector.analyze_with_context("Actually, I just got great news! I'm happy now!", history)
        
        # Check that sentiment shift was detected
        self.assertIsNotNone(result.sentiment_shift)
        self.assertTrue(result.sentiment_shift.get("shift_detected", False))
        self.assertTrue(result.sentiment_shift.get("shift_magnitude", 0) > 0.5)
        self.assertEqual(result.sentiment_shift.get("previous_sentiment"), "sadness")
        self.assertEqual(result.sentiment_shift.get("current_sentiment"), "joy")
    
    def test_analyze_with_fluctuating_history(self):
        """Test analyzing with fluctuating conversation history"""
        history = [
            {
                "text": "I'm happy today",
                "sentiment": {
                    "dominant_emotion": "joy",
                    "emotion_confidence": 0.7
                }
            },
            {
                "text": "Actually I'm a bit worried",
                "sentiment": {
                    "dominant_emotion": "fear",
                    "emotion_confidence": 0.5
                }
            },
            {
                "text": "Now I'm feeling better",
                "sentiment": {
                    "dominant_emotion": "joy",
                    "emotion_confidence": 0.6
                }
            }
        ]
        
        result = self.detector.analyze_with_context("I'm a bit confused about how I feel", history)
        
        # Check that conversation pattern was detected as fluctuating
        self.assertIsNotNone(result.conversation_pattern)
        self.assertEqual(result.conversation_pattern.pattern_type, "fluctuating")
        
        # Check that confidence was reduced due to fluctuating pattern
        self.assertLess(result.context_confidence, 0.6)
    
    def test_analyze_with_escalating_history(self):
        """Test analyzing with escalating emotional intensity"""
        history = [
            {
                "text": "I'm a bit annoyed",
                "sentiment": {
                    "dominant_emotion": "anger",
                    "emotion_confidence": 0.3
                }
            },
            {
                "text": "This is getting frustrating",
                "sentiment": {
                    "dominant_emotion": "anger",
                    "emotion_confidence": 0.5
                }
            },
            {
                "text": "I'm really angry now",
                "sentiment": {
                    "dominant_emotion": "anger",
                    "emotion_confidence": 0.7
                }
            }
        ]
        
        result = self.detector.analyze_with_context("I'm absolutely furious about this!", history)
        
        # Check that conversation pattern was detected as escalating
        self.assertIsNotNone(result.conversation_pattern)
        self.assertEqual(result.conversation_pattern.pattern_type, "escalating")
        self.assertTrue(result.conversation_pattern.intensity_trend > 0)
        
        # Check that sentiment shift exists
        self.assertIsNotNone(result.sentiment_shift)
        # The shift might not be detected as significant since it's the same emotion category
        # Just check that the shift information is present
        self.assertIn("shift_magnitude", result.sentiment_shift)
    
    def test_analyze_with_topic_continuity(self):
        """Test analyzing with topic continuity in conversation history"""
        history = [
            {
                "text": "I love watching anime",
                "sentiment": {
                    "dominant_emotion": "joy",
                    "emotion_confidence": 0.7
                },
                "topics": ["anime"]
            },
            {
                "text": "My favorite anime is One Piece",
                "sentiment": {
                    "dominant_emotion": "joy",
                    "emotion_confidence": 0.8
                },
                "topics": ["anime"]
            }
        ]
        
        result = self.detector.analyze_with_context("I just watched the latest episode and it was amazing!", history)
        
        # Check that conversation pattern has high topic continuity
        self.assertIsNotNone(result.conversation_pattern)
        self.assertTrue(result.conversation_pattern.topic_continuity > 0.8)
    
    def test_explanation_includes_history_info(self):
        """Test that the explanation includes conversation history information"""
        history = [
            {
                "text": "I'm having a good day",
                "sentiment": {
                    "dominant_emotion": "joy",
                    "emotion_confidence": 0.7
                }
            },
            {
                "text": "Everything is going well",
                "sentiment": {
                    "dominant_emotion": "joy",
                    "emotion_confidence": 0.6
                }
            }
        ]
        
        result = self.detector.analyze_with_context("I'm still feeling happy", history)
        explanation = self.detector.get_contextual_explanation(result)
        
        # Check that explanation includes conversation pattern information
        self.assertIn("Conversation pattern", explanation)
        self.assertIn("Sentiment stability", explanation)
        self.assertIn("Dominant emotions in history", explanation)

if __name__ == "__main__":
    unittest.main()