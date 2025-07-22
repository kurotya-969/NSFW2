"""
Test script for intensity-based affection scaling
Tests the integration of emotion intensity detection with affection system
"""

import unittest
from context_sentiment_detector_updated import ContextSentimentDetector
from emotion_intensity_detector import EmotionIntensityDetector, IntensityAnalysisResult

class TestIntensityBasedAffectionScaling(unittest.TestCase):
    """Test cases for intensity-based affection scaling"""
    
    def setUp(self):
        """Set up the test environment"""
        self.detector = ContextSentimentDetector()
        self.intensity_detector = EmotionIntensityDetector()
    
    def test_basic_intensity_scaling(self):
        """Test that intensity affects affection delta"""
        # Test with mild intensity
        result_mild = self.detector.analyze_with_context("I'm slightly happy about this.")
        
        # Test with strong intensity
        result_strong = self.detector.analyze_with_context("I'm extremely happy about this!!!")
        
        # Strong intensity should have greater affection impact than mild
        self.assertGreater(result_strong.adjusted_affection_delta, result_mild.adjusted_affection_delta)
        self.assertGreater(result_strong.adjusted_sentiment_score, result_mild.adjusted_sentiment_score)
        
        # Verify intensity categories
        self.assertEqual(result_mild.intensity_analysis.intensity_category, "mild")
        self.assertIn(result_strong.intensity_analysis.intensity_category, ["strong", "extreme"])
    
    def test_negative_intensity_scaling(self):
        """Test intensity scaling with negative sentiment"""
        # Test with mild negative intensity
        result_mild = self.detector.analyze_with_context("I'm a bit disappointed.")
        
        # Test with strong negative intensity
        result_strong = self.detector.analyze_with_context("I'm absolutely FURIOUS about this!!!")
        
        # Strong intensity should have greater negative affection impact
        self.assertLess(result_strong.adjusted_affection_delta, result_mild.adjusted_affection_delta)
        self.assertLess(result_strong.adjusted_sentiment_score, result_mild.adjusted_sentiment_score)
    
    def test_extreme_intensity_bonus(self):
        """Test that extreme intensity gets an additional affection bonus"""
        # Create a message with extreme emotional intensity
        result_extreme = self.detector.analyze_with_context("I'm ABSOLUTELY THRILLED!!! This is the BEST thing EVER!!! 😍😍😍")
        
        # Verify it's categorized as extreme
        self.assertEqual(result_extreme.intensity_analysis.intensity_category, "extreme")
        
        # Verify the affection delta is significant
        self.assertGreaterEqual(result_extreme.adjusted_affection_delta, 5)
        
        # Test extreme negative intensity
        result_extreme_neg = self.detector.analyze_with_context("I'm ABSOLUTELY FURIOUS!!! This is the WORST thing EVER!!! 😡😡😡")
        
        # Verify the negative affection delta is significant
        self.assertLessEqual(result_extreme_neg.adjusted_affection_delta, -5)
    
    def test_confidence_weighting(self):
        """Test that confidence affects intensity scaling"""
        # Create a message with clear intensity markers (high confidence)
        result_high_conf = self.detector.analyze_with_context("I'm extremely happy and excited about this wonderful news!!!")
        
        # Create a message with fewer intensity markers (lower confidence)
        result_low_conf = self.detector.analyze_with_context("I'm happy about this.")
        
        # Higher confidence should lead to stronger scaling effect
        self.assertGreater(result_high_conf.intensity_analysis.confidence, result_low_conf.intensity_analysis.confidence)
        
        # Calculate the ratio of affection delta to raw sentiment score
        # This ratio should be higher for the high confidence result
        if result_high_conf.raw_sentiment.sentiment_score != 0 and result_low_conf.raw_sentiment.sentiment_score != 0:
            high_conf_ratio = abs(result_high_conf.adjusted_affection_delta / result_high_conf.raw_sentiment.affection_delta)
            low_conf_ratio = abs(result_low_conf.adjusted_affection_delta / result_low_conf.raw_sentiment.affection_delta)
            
            # The scaling effect should be stronger with higher confidence
            self.assertGreaterEqual(high_conf_ratio, low_conf_ratio)
    
    def test_mixed_signals_handling(self):
        """Test handling of mixed intensifiers and qualifiers"""
        # Text with mixed signals (both intensifier and qualifier)
        result_mixed = self.detector.analyze_with_context("I'm very slightly happy about this.")
        
        # Text with just intensifier
        result_intensifier = self.detector.analyze_with_context("I'm very happy about this.")
        
        # Text with just qualifier
        result_qualifier = self.detector.analyze_with_context("I'm slightly happy about this.")
        
        # Mixed signals should be between the two extremes
        self.assertLess(result_mixed.adjusted_affection_delta, result_intensifier.adjusted_affection_delta)
        self.assertGreater(result_mixed.adjusted_affection_delta, result_qualifier.adjusted_affection_delta)
        
        # Verify that intensifiers and qualifiers were detected
        self.assertIn("very", result_mixed.intensity_analysis.intensifiers)
        self.assertIn("slightly", result_mixed.intensity_analysis.qualifiers)
    
    def test_japanese_intensity_scaling(self):
        """Test intensity scaling with Japanese text"""
        # Test with mild Japanese intensity
        result_mild = self.detector.analyze_with_context("ちょっと嬉しいです。")
        
        # Test with strong Japanese intensity
        result_strong = self.detector.analyze_with_context("とても非常に嬉しいです！！！")
        
        # Strong intensity should have greater affection impact
        self.assertGreater(result_strong.adjusted_affection_delta, result_mild.adjusted_affection_delta)
        
        # Verify Japanese intensifiers and qualifiers were detected
        self.assertIn("ちょっと", result_mild.intensity_analysis.qualifiers)
        self.assertIn("とても", result_strong.intensity_analysis.intensifiers)
    
    def test_intensity_bounds(self):
        """Test that intensity scaling respects affection bounds"""
        # Test with extremely positive text
        result_extreme_pos = self.detector.analyze_with_context(
            "I'm absolutely THRILLED and OVERJOYED!!! This is the MOST AMAZING thing EVER!!! "
            "I LOVE it SO MUCH!!! 😍😍😍❤️❤️❤️ THANK YOU!!!"
        )
        
        # Test with extremely negative text
        result_extreme_neg = self.detector.analyze_with_context(
            "I'm absolutely FURIOUS and DISGUSTED!!! This is the WORST thing EVER!!! "
            "I HATE it SO MUCH!!! 😡😡😡 This is TERRIBLE!!!"
        )
        
        # Verify bounds are respected
        self.assertLessEqual(result_extreme_pos.adjusted_affection_delta, 10)
        self.assertGreaterEqual(result_extreme_neg.adjusted_affection_delta, -10)
        
        self.assertLessEqual(result_extreme_pos.adjusted_sentiment_score, 1.0)
        self.assertGreaterEqual(result_extreme_neg.adjusted_sentiment_score, -1.0)

if __name__ == "__main__":
    unittest.main()