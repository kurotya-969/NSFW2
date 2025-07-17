"""
Emotion Intensity Detector for Context-Based Sentiment Analysis
Analyzes the strength and intensity of emotional expressions in text
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class IntensityAnalysisResult:
    """Result of emotion intensity analysis"""
    intensity_score: float  # 0.0 to 1.0
    intensity_category: str  # "mild", "moderate", "strong", "extreme"
    intensifiers: List[str]  # Words that amplify emotions
    qualifiers: List[str]  # Words that modify emotion strength
    confidence: float  # 0.0 to 1.0

class EmotionIntensityDetector:
    """Detects the strength and intensity of emotional expressions in text"""
    
    def __init__(self):
        """Initialize the emotion intensity detector"""
        self.intensifiers = self._load_intensifiers()
        self.qualifiers = self._load_qualifiers()
        self.emotion_indicators = self._load_emotion_indicators()
        self.intensity_patterns = self._load_intensity_patterns()
    
    def _load_intensifiers(self) -> Dict[str, float]:
        """
        Load words that intensify emotions with their multiplier values
        
        Returns:
            Dictionary mapping intensifier words to their multiplier values
        """
        return {
            # English intensifiers
            "very": 1.5,
            "really": 1.5,
            "extremely": 1.8,
            "incredibly": 1.7,
            "absolutely": 1.8,
            "completely": 1.6,
            "totally": 1.6,
            "utterly": 1.7,
            "so": 1.4,
            "too": 1.3,
            "deeply": 1.5,
            "highly": 1.5,
            "intensely": 1.7,
            "terribly": 1.6,
            "awfully": 1.5,
            "exceptionally": 1.6,
            "particularly": 1.4,
            "especially": 1.5,
            "remarkably": 1.5,
            "truly": 1.4,
            
            # Japanese intensifiers
            "とても": 1.5,
            "非常に": 1.8,
            "すごく": 1.6,
            "かなり": 1.4,
            "めちゃ": 1.6,
            "めっちゃ": 1.7,
            "超": 1.7,
            "激": 1.8,
            "すごい": 1.5,
            "ものすごく": 1.7,
            "相当": 1.5,
            "本当に": 1.4,
            "マジで": 1.5,
            "完全に": 1.6,
            "全く": 1.5,
            "絶対に": 1.6
        }
    
    def _load_qualifiers(self) -> Dict[str, float]:
        """
        Load words that qualify or diminish emotions with their multiplier values
        
        Returns:
            Dictionary mapping qualifier words to their multiplier values
        """
        return {
            # English qualifiers (diminishers)
            "somewhat": 0.7,
            "slightly": 0.6,
            "a bit": 0.6,
            "a little": 0.6,
            "kind of": 0.7,
            "sort of": 0.7,
            "rather": 0.8,
            "fairly": 0.8,
            "pretty": 0.8,
            "moderately": 0.8,
            "relatively": 0.7,
            "mildly": 0.6,
            "partially": 0.7,
            "barely": 0.4,
            "hardly": 0.4,
            "scarcely": 0.4,
            "almost": 0.8,
            "nearly": 0.8,
            
            # Japanese qualifiers (diminishers)
            "少し": 0.6,
            "ちょっと": 0.6,
            "やや": 0.8,
            "多少": 0.7,
            "若干": 0.7,
            "わずかに": 0.5,
            "ほんの": 0.6,
            "それほど": 0.7,
            "そこまで": 0.7,
            "まあまあ": 0.8,
            "なんとなく": 0.7,
            "どちらかといえば": 0.8
        }
    
    def _load_emotion_indicators(self) -> Dict[str, float]:
        """
        Load words that indicate emotional content with their base intensity values
        
        Returns:
            Dictionary mapping emotion words to their base intensity values
        """
        return {
            # English emotion indicators - positive
            "happy": 0.5,
            "glad": 0.4,
            "delighted": 0.6,
            "thrilled": 0.7,
            "excited": 0.6,
            "overjoyed": 0.8,
            "ecstatic": 0.8,
            "pleased": 0.4,
            "content": 0.3,
            "satisfied": 0.4,
            "grateful": 0.5,
            "thankful": 0.5,
            "love": 0.7,
            "adore": 0.7,
            "like": 0.4,
            "enjoy": 0.5,
            "appreciate": 0.5,
            
            # English emotion indicators - negative
            "sad": 0.6,
            "unhappy": 0.5,
            "depressed": 0.7,
            "miserable": 0.8,
            "devastated": 0.9,
            "heartbroken": 0.9,
            "disappointed": 0.6,
            "upset": 0.6,
            "angry": 0.7,
            "furious": 0.9,
            "enraged": 0.9,
            "annoyed": 0.5,
            "irritated": 0.6,
            "frustrated": 0.7,
            "afraid": 0.6,
            "scared": 0.7,
            "terrified": 0.9,
            "worried": 0.6,
            "anxious": 0.7,
            "hate": 0.8,
            "dislike": 0.6,
            "disgusted": 0.7,
            
            # Japanese emotion indicators - positive
            "嬉しい": 0.6,
            "楽しい": 0.6,
            "幸せ": 0.7,
            "喜び": 0.6,
            "満足": 0.5,
            "安心": 0.5,
            "好き": 0.6,
            "大好き": 0.8,
            "愛": 0.8,
            "感謝": 0.6,
            "ありがとう": 0.5,
            
            # Japanese emotion indicators - negative
            "悲しい": 0.6,
            "寂しい": 0.6,
            "辛い": 0.7,
            "苦しい": 0.7,
            "切ない": 0.6,
            "落ち込む": 0.6,
            "怒り": 0.7,
            "腹立つ": 0.7,
            "イライラ": 0.6,
            "ムカつく": 0.7,
            "不安": 0.6,
            "心配": 0.6,
            "怖い": 0.7,
            "恐怖": 0.8,
            "嫌い": 0.7,
            "憎い": 0.8,
            "嫌悪": 0.7
        }
    
    def _load_intensity_patterns(self) -> Dict[str, float]:
        """
        Load patterns that indicate emotional intensity
        
        Returns:
            Dictionary mapping intensity patterns to their intensity values
        """
        return {
            # Exclamation patterns
            r"!{2,}": 0.2,  # Multiple exclamation marks
            r"\?{2,}": 0.1,  # Multiple question marks
            r"[A-Z]{3,}": 0.2,  # ALL CAPS (3+ letters)
            
            # Repetition patterns
            r"(\w+)\1{2,}": 0.2,  # Word repetition (e.g., "very very very")
            r"\.{3,}": 0.1,  # Ellipsis
            
            # Emphasis patterns
            r"\*\w+\*": 0.1,  # *emphasized* text
            r"_\w+_": 0.1,  # _emphasized_ text
            
            # Emoji patterns (simplified)
            r"[😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙😚🙂🤗🤩🥳]": 0.2,  # Positive emojis
            r"[😔😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬😱😨😰😥😓]": 0.2,  # Negative emojis
            r"[❤️💕💓💗💖💘💝💟💌]": 0.2,  # Heart emojis
        }
    
    def detect_intensity(self, text: str) -> IntensityAnalysisResult:
        """
        Detect the emotional intensity in text
        
        Args:
            text: The text to analyze
            
        Returns:
            IntensityAnalysisResult with details about the emotional intensity
        """
        # Normalize text to lowercase for analysis
        normalized_text = text.lower()
        
        # Identify intensifiers and qualifiers
        intensifiers, intensifier_score = self._identify_intensifiers(normalized_text)
        qualifiers, qualifier_score = self._identify_qualifiers(normalized_text)
        
        # Identify base emotional content
        base_intensity = self._detect_base_intensity(normalized_text)
        
        # Identify intensity patterns
        pattern_intensity = self._detect_intensity_patterns(text)  # Use original text for case sensitivity
        
        # Calculate final intensity score
        intensity_score = self._calculate_intensity_score(
            base_intensity, 
            intensifier_score, 
            qualifier_score, 
            pattern_intensity
        )
        
        # Determine intensity category
        intensity_category = self._determine_intensity_category(intensity_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            base_intensity, 
            len(intensifiers), 
            len(qualifiers), 
            pattern_intensity
        )
        
        return IntensityAnalysisResult(
            intensity_score=intensity_score,
            intensity_category=intensity_category,
            intensifiers=intensifiers,
            qualifiers=qualifiers,
            confidence=confidence
        )
    
    def _identify_intensifiers(self, text: str) -> Tuple[List[str], float]:
        """
        Identify intensifier words in text
        
        Args:
            text: The text to analyze
            
        Returns:
            Tuple of (found_intensifiers, total_intensifier_score)
        """
        found_intensifiers = []
        total_score = 1.0  # Start with neutral multiplier
        
        for intensifier, value in self.intensifiers.items():
            if intensifier in text:
                found_intensifiers.append(intensifier)
                # Multiply scores for multiple intensifiers
                total_score *= value
        
        # Cap the total score to avoid extreme values from multiple intensifiers
        total_score = min(2.5, total_score)
        
        return found_intensifiers, total_score
    
    def _identify_qualifiers(self, text: str) -> Tuple[List[str], float]:
        """
        Identify qualifier words in text
        
        Args:
            text: The text to analyze
            
        Returns:
            Tuple of (found_qualifiers, total_qualifier_score)
        """
        found_qualifiers = []
        total_score = 1.0  # Start with neutral multiplier
        
        for qualifier, value in self.qualifiers.items():
            if qualifier in text:
                found_qualifiers.append(qualifier)
                # Multiply scores for multiple qualifiers
                total_score *= value
        
        # Cap the total score to avoid extreme values from multiple qualifiers
        total_score = max(0.3, total_score)
        
        return found_qualifiers, total_score
    
    def _detect_base_intensity(self, text: str) -> float:
        """
        Detect the base emotional intensity in text
        
        Args:
            text: The text to analyze
            
        Returns:
            Float representing the base intensity (0.0 to 1.0)
        """
        # Find all emotion indicators in the text
        max_intensity = 0.0
        total_intensity = 0.0
        count = 0
        
        for indicator, value in self.emotion_indicators.items():
            if indicator in text:
                max_intensity = max(max_intensity, value)
                total_intensity += value
                count += 1
        
        # If no emotion indicators found, return low base intensity
        if count == 0:
            return 0.3
        
        # Calculate weighted average: 70% max intensity, 30% average intensity
        avg_intensity = total_intensity / count if count > 0 else 0.0
        base_intensity = (max_intensity * 0.7) + (avg_intensity * 0.3)
        
        return base_intensity
    
    def _detect_intensity_patterns(self, text: str) -> float:
        """
        Detect patterns that indicate emotional intensity
        
        Args:
            text: The text to analyze (original case preserved)
            
        Returns:
            Float representing the pattern-based intensity boost (0.0 to 0.5)
        """
        total_boost = 0.0
        
        for pattern, boost in self.intensity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                # Add boost for each match, with diminishing returns
                total_boost += boost * min(3, len(matches)) / 3
        
        # Cap the total boost
        return min(0.5, total_boost)
    
    def _calculate_intensity_score(self, base_intensity: float, intensifier_score: float, 
                                 qualifier_score: float, pattern_intensity: float) -> float:
        """
        Calculate the final intensity score
        
        Args:
            base_intensity: Base emotional intensity
            intensifier_score: Score from intensifiers
            qualifier_score: Score from qualifiers
            pattern_intensity: Intensity from patterns
            
        Returns:
            Float representing the final intensity score (0.0 to 1.0)
        """
        # Apply intensifiers and qualifiers to base intensity
        modified_intensity = base_intensity * intensifier_score * qualifier_score
        
        # Add pattern-based intensity (with reduced impact to avoid overscoring)
        final_intensity = modified_intensity + (pattern_intensity * 0.7)
        
        # Ensure the result is within bounds
        return max(0.0, min(1.0, final_intensity))
    
    def _determine_intensity_category(self, intensity_score: float) -> str:
        """
        Determine the intensity category based on the score
        
        Args:
            intensity_score: The calculated intensity score
            
        Returns:
            String representing the intensity category
        """
        if intensity_score <= 0.3:
            return "mild"
        elif intensity_score <= 0.6:
            return "moderate"
        elif intensity_score <= 0.85:
            return "strong"
        else:
            return "extreme"
    
    def _calculate_confidence(self, base_intensity: float, num_intensifiers: int, 
                            num_qualifiers: int, pattern_intensity: float) -> float:
        """
        Calculate confidence in the intensity analysis
        
        Args:
            base_intensity: Base emotional intensity
            num_intensifiers: Number of intensifiers found
            num_qualifiers: Number of qualifiers found
            pattern_intensity: Intensity from patterns
            
        Returns:
            Float representing confidence (0.0 to 1.0)
        """
        # Start with moderate confidence
        confidence = 0.5
        
        # More emotional content increases confidence
        if base_intensity > 0.5:
            confidence += 0.1
        
        # More modifiers increases confidence
        modifier_count = num_intensifiers + num_qualifiers
        if modifier_count > 0:
            confidence += min(0.2, modifier_count * 0.05)
        
        # More patterns increases confidence
        if pattern_intensity > 0:
            confidence += min(0.2, pattern_intensity * 0.4)
        
        # Ensure the result is within bounds
        return max(0.0, min(1.0, confidence))