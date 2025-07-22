"""
Test script for emotion intensity detector
"""

import unittest
from emotion_intensity_detector import EmotionIntensityDetector, IntensityAnalysisResult

class TestEmotionIntensityDetector(unittest.TestCase):
    """Test cases for the EmotionIntensityDetector class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.detector = EmotionIntensityDetector()
    
    def test_basic_intensity_detection(self):
        """Test basic intensity detection"""
        # Test with neutral text
        result = self.detector.detect_intensity("This is a normal sentence.")
        self.assertLessEqual(result.intensity_score, 0.4)
        self.assertEqual(result.intensity_category, "mild")
        
        # Test with emotional text
        result = self.detector.detect_intensity("I'm happy about this.")
        self.assertGreater(result.intensity_score, 0.4)
        self.assertIn(result.intensity_category, ["moderate", "strong"])
    
    def test_intensifier_detection(self):
        """Test detection of intensifiers"""
        # Without intensifier
        result_without = self.detector.detect_intensity("I'm happy about this.")
        
        # With intensifier
        result_with = self.detector.detect_intensity("I'm very happy about this.")
        
        # Intensity should be higher with intensifier
        self.assertGreater(result_with.intensity_score, result_without.intensity_score)
        self.assertIn("very", result_with.intensifiers)
        
        # Test with multiple intensifiers
        result_multiple = self.detector.detect_intensity("I'm really extremely happy about this.")
        self.assertGreater(result_multiple.intensity_score, result_with.intensity_score)
        self.assertGreaterEqual(len(result_multiple.intensifiers), 2)
    
    def test_qualifier_detection(self):
        """Test detection of qualifiers"""
        # Without qualifier
        result_without = self.detector.detect_intensity("I'm angry about this.")
        
        # With qualifier
        result_with = self.detector.detect_intensity("I'm a bit angry about this.")
        
        # Intensity should be lower with qualifier
        self.assertLess(result_with.intensity_score, result_without.intensity_score)
        self.assertIn("a bit", result_with.qualifiers)
        
        # Test with multiple qualifiers
        result_multiple = self.detector.detect_intensity("I'm somewhat slightly angry about this.")
        self.assertLess(result_multiple.intensity_score, result_with.intensity_score)
        self.assertGreaterEqual(len(result_multiple.qualifiers), 2)
    
    def test_pattern_detection(self):
        """Test detection of intensity patterns"""
        # Without patterns
        result_without = self.detector.detect_intensity("I'm happy about this.")
        
        # With exclamation patterns
        result_with_exclamation = self.detector.detect_intensity("I'm happy about this!!!")
        self.assertGreater(result_with_exclamation.intensity_score, result_without.intensity_score)
        
        # With capitalization patterns
        result_with_caps = self.detector.detect_intensity("I'm HAPPY about this.")
        self.assertGreater(result_with_caps.intensity_score, result_without.intensity_score)
        
        # With emoji patterns
        result_with_emoji = self.detector.detect_intensity("I'm happy about this 😁")
        self.assertGreater(result_with_emoji.intensity_score, result_without.intensity_score)
    
    def test_intensity_categories(self):
        """Test intensity category classification"""
        # Test mild intensity
        result_mild = self.detector.detect_intensity("I'm slightly interested in this topic.")
        self.assertEqual(result_mild.intensity_category, "mild")
        self.assertLessEqual(result_mild.intensity_score, 0.3)
        
        # Test moderate intensity
        result_moderate = self.detector.detect_intensity("I'm happy about this news.")
        self.assertEqual(result_moderate.intensity_category, "moderate")
        self.assertGreater(result_moderate.intensity_score, 0.3)
        self.assertLessEqual(result_moderate.intensity_score, 0.6)
        
        # Test strong intensity
        result_strong = self.detector.detect_intensity("I'm excited about this opportunity.")
        self.assertIn(result_strong.intensity_category, ["strong", "moderate"])
        self.assertGreater(result_strong.intensity_score, 0.4)
        self.assertLessEqual(result_strong.intensity_score, 0.85)
        
        # Test extreme intensity
        result_extreme = self.detector.detect_intensity("I'm absolutely THRILLED!!! This is the BEST news EVER!!! 😍😍😍")
        self.assertEqual(result_extreme.intensity_category, "extreme")
        self.assertGreater(result_extreme.intensity_score, 0.85)
    
    def test_japanese_text(self):
        """Test intensity detection with Japanese text"""
        # Test with Japanese text
        result = self.detector.detect_intensity("とても嬉しいです！")
        self.assertGreater(result.intensity_score, 0.6)
        self.assertIn("とても", result.intensifiers)
        
        # Test with Japanese qualifiers
        result = self.detector.detect_intensity("ちょっと悲しいです。")
        self.assertIn("ちょっと", result.qualifiers)
        self.assertIn(result.intensity_category, ["mild", "moderate"])
    
    def test_confidence_calculation(self):
        """Test confidence calculation in intensity analysis"""
        # Simple text should have moderate confidence
        result_simple = self.detector.detect_intensity("I'm happy.")
        self.assertGreaterEqual(result_simple.confidence, 0.5)
        
        # Text with intensifiers should have higher confidence
        result_with_intensifiers = self.detector.detect_intensity("I'm extremely happy.")
        self.assertGreater(result_with_intensifiers.confidence, result_simple.confidence)
        
        # Text with patterns should have higher confidence
        result_with_patterns = self.detector.detect_intensity("I'm happy!!!")
        self.assertGreater(result_with_patterns.confidence, result_simple.confidence)
        
        # Complex text with multiple indicators should have highest confidence
        result_complex = self.detector.detect_intensity("I'm extremely HAPPY!!! This is the BEST day EVER!!! 😍")
        self.assertGreater(result_complex.confidence, 0.7)
    
    def test_mixed_modifiers(self):
        """Test with mixed intensifiers and qualifiers"""
        # Mixed signals should balance each other
        result = self.detector.detect_intensity("I'm very slightly happy.")
        
        # Compare with just intensifier
        result_intensifier = self.detector.detect_intensity("I'm very happy.")
        
        # Compare with just qualifier
        result_qualifier = self.detector.detect_intensity("I'm slightly happy.")
        
        # Mixed should be between the two
        self.assertLess(result.intensity_score, result_intensifier.intensity_score)
        self.assertGreater(result.intensity_score, result_qualifier.intensity_score)

if __name__ == "__main__":
    unittest.main()