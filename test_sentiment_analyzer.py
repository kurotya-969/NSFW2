"""
Test script for sentiment analyzer
"""

import unittest
from sentiment_analyzer import SentimentAnalyzer, SentimentType, SentimentAnalysisResult

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for the SentimentAnalyzer class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.analyzer = SentimentAnalyzer()
    
    def test_empty_input(self):
        """Test that empty input returns neutral sentiment"""
        result = self.analyzer.analyze_user_input("")
        self.assertEqual(result.sentiment_score, 0.0)
        self.assertEqual(result.interaction_type, "neutral")
        self.assertEqual(result.affection_delta, 0)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(result.detected_keywords), 0)
        self.assertEqual(result.sentiment_types, [SentimentType.NEUTRAL])
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        result = self.analyzer.analyze_user_input("ありがとう、とても素晴らしいです！")
        self.assertGreater(result.sentiment_score, 0)
        self.assertGreater(result.affection_delta, 0)
        self.assertIn("ありがとう", result.detected_keywords)
        self.assertIn(SentimentType.POSITIVE, result.sentiment_types)
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        result = self.analyzer.analyze_user_input("うるさい、バカ")
        self.assertLess(result.sentiment_score, 0)
        self.assertLess(result.affection_delta, 0)
        self.assertTrue(any(kw in result.detected_keywords for kw in ["うるさい", "バカ"]))
        self.assertIn(SentimentType.NEGATIVE, result.sentiment_types)
    
    def test_caring_sentiment(self):
        """Test caring sentiment detection"""
        result = self.analyzer.analyze_user_input("大丈夫？心配してるよ")
        self.assertGreater(result.sentiment_score, 0)
        self.assertGreater(result.affection_delta, 0)
        self.assertTrue(any(kw in result.detected_keywords for kw in ["大丈夫", "心配"]))
        self.assertIn(SentimentType.CARING, result.sentiment_types)
    
    def test_dismissive_sentiment(self):
        """Test dismissive sentiment detection"""
        result = self.analyzer.analyze_user_input("どうでもいい、知らない")
        self.assertLess(result.sentiment_score, 0)
        self.assertLess(result.affection_delta, 0)
        self.assertTrue(any(kw in result.detected_keywords for kw in ["どうでもいい", "知らない"]))
        self.assertIn(SentimentType.DISMISSIVE, result.sentiment_types)
    
    def test_appreciative_sentiment(self):
        """Test appreciative sentiment detection"""
        result = self.analyzer.analyze_user_input("本当に助かった、感謝します")
        self.assertGreater(result.sentiment_score, 0)
        self.assertGreater(result.affection_delta, 0)
        self.assertTrue(any(kw in result.detected_keywords for kw in ["助かった", "感謝"]))
        self.assertIn(SentimentType.APPRECIATIVE, result.sentiment_types)
    
    def test_hostile_sentiment(self):
        """Test hostile sentiment detection"""
        result = self.analyzer.analyze_user_input("ふざけるな、てめー")
        self.assertLess(result.sentiment_score, 0)
        self.assertLess(result.affection_delta, 0)
        self.assertTrue(any(kw in result.detected_keywords for kw in ["ふざけるな", "てめー"]))
        self.assertIn(SentimentType.HOSTILE, result.sentiment_types)
    
    def test_mixed_sentiment(self):
        """Test mixed sentiment detection"""
        result = self.analyzer.analyze_user_input("ありがとう、でもちょっとうざい")
        # Should detect both positive and negative
        self.assertIn(SentimentType.POSITIVE, result.sentiment_types)
        self.assertIn(SentimentType.NEGATIVE, result.sentiment_types)
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        result = self.analyzer.analyze_user_input("今日は晴れています")
        self.assertAlmostEqual(result.sentiment_score, 0.0, delta=0.3)
        self.assertEqual(result.interaction_type, "neutral")
    
    def test_affection_delta_bounds(self):
        """Test that affection delta is properly bounded"""
        # Create a very negative message with many negative keywords
        very_negative = " ".join(["うざい", "きもい", "バカ", "死ね", "黙れ", "hate", "stupid", "shut up"])
        result = self.analyzer.analyze_user_input(very_negative)
        self.assertGreaterEqual(result.affection_delta, -10)  # Should not go below -10
        
        # Create a very positive message with many positive keywords
        very_positive = " ".join(["ありがとう", "すごい", "いいね", "よかった", "うれしい", "thank", "great", "awesome", "love"])
        result = self.analyzer.analyze_user_input(very_positive)
        self.assertLessEqual(result.affection_delta, 10)  # Should not go above 10

if __name__ == "__main__":
    unittest.main()