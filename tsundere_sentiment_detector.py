"""
Tsundere Sentiment Detector for Mari AI Chat
Detects tsundere expressions and distinguishes them from genuine negative sentiment
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set

from context_sentiment_detector import ContextSentimentDetector, ContextualSentimentResult
from sentiment_analyzer import SentimentAnalysisResult, SentimentType

@dataclass
class TsundereAnalysisResult:
    """Result of tsundere expression detection"""
    is_tsundere: bool
    tsundere_confidence: float
    detected_patterns: List[str]
    character_consistency: float
    suggested_interpretation: str
    affection_adjustment: int
    sentiment_adjustment: float
    is_farewell: bool = False
    farewell_type: Optional[str] = None
    cultural_context: Optional[str] = None
    is_conversation_ending: bool = False

@dataclass
class SentimentLoopData:
    """Data about potential sentiment loops"""
    loop_detected: bool
    loop_severity: float
    repeated_patterns: List[str]
    loop_duration: int  # Number of turns
    suggested_intervention: str
    affection_recovery_suggestion: int

class TsundereSentimentDetector:
    """Detects tsundere expressions and distinguishes them from genuine negative sentiment"""
    
    def __init__(self):
        """Initialize the tsundere sentiment detector"""
        self.context_sentiment_detector = ContextSentimentDetector()
        self.tsundere_patterns = self._load_tsundere_patterns()
        self.farewell_phrases = self._load_farewell_phrases()
        self.character_profile = self._load_character_profile()
        self.sentiment_loop_history = {}  # Store history by session ID
    
    def _load_tsundere_patterns(self) -> Dict[str, List[str]]:
        """
        Load patterns that indicate tsundere expressions
        
        Returns:
            Dictionary mapping tsundere types to their pattern indicators
        """
        return {
            "dismissive_affection": [
                r"(別に|べつに).*(好き|すき|気に入った|きにいった).*わけじゃない",  # "It's not like I like you or anything"
                r"(別に|べつに).*(あんた|君|きみ|お前|おまえ)のため.*じゃない",  # "It's not like I did it for you"
                r"勘違いしないでよ",  # "Don't get the wrong idea"
                r"(感謝|かんしゃ)なんて(してない|しない|するわけない)",  # "I'm not thanking you or anything"
                r"(褒めて|ほめて)(ない|いるわけじゃない)",  # "I'm not praising you"
            ],
            "hostile_care": [
                r"(うるさい|うっせー|黙れ).*(心配|しんぱい|大丈夫|だいじょうぶ)",  # "Shut up, I'm worried/Are you okay"
                r"(バカ|ばか|あほ).*(気をつけ|きをつけ|無理|むり|大丈夫|だいじょうぶ)",  # "Idiot, be careful/don't push yourself"
                r"(迷惑|めいわく|邪魔|じゃま).*(手伝|てつだ|助け|たすけ)",  # "You're a bother but I'll help"
                r"(うざい|うるさい).*(でも|だけど|けど).*",  # "You're annoying but..."
            ],
            "tsundere_farewell": [
                r"(じゃあな|じゃーな|じゃな)",  # "See ya" (casual farewell)
                r"(また(な|ね)|またね)",  # "See you later"
                r"(バイバイ|ばいばい)",  # "Bye bye"
                r"(さようなら|さよなら)",  # "Goodbye"
                r"(またあした|また明日)",  # "See you tomorrow"
                r"(またあとで|また後で)",  # "See you later"
                r"(行ってくる|いってくる)",  # "I'm going"
                r"(帰る|かえる)(から|よ|わ|ね)",  # "I'm going home"
            ],
            "reluctant_gratitude": [
                r"(別に|べつに).*(ありがと|感謝|かんしゃ)",  # "It's not like I'm grateful or anything"
                r"(まあ|ま)(ありがと|サンキュ|さんきゅ)",  # "Well... thanks"
                r"(一応|いちおう)(礼|れい|ありがと|感謝|かんしゃ)",  # "I guess I should thank you"
                r"(感謝|かんしゃ)(してる|するよ).*(と思うな|とおもうな)",  # "Don't think I'm grateful"
            ],
            "insult_affection": [
                r"(バカ|ばか|あほ).*(好き|すき|大好き|だいすき)",  # "Idiot... I like you"
                r"(うざい|うるさい).*(でも|だけど|けど).*(好き|すき|大好き|だいすき)",  # "You're annoying but I like you"
                r"(嫌い|きらい).*(わけじゃない|というわけではない)",  # "It's not that I hate you"
                r"(バカ|ばか|あほ).*(嬉しい|うれしい)",  # "Idiot... I'm happy"
            ]
        }
    
    def _load_farewell_phrases(self) -> Dict[str, Dict[str, Any]]:
        """
        Load farewell phrases with their classifications
        
        Returns:
            Dictionary mapping farewell phrases to their classifications
        """
        return {
            # Japanese casual farewells
            "じゃあな": {
                "type": "casual",
                "cultural_context": "japanese",
                "is_tsundere": True,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "See ya"
            },
            "じゃーな": {
                "type": "casual",
                "cultural_context": "japanese",
                "is_tsundere": True,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "See ya"
            },
            "じゃな": {
                "type": "casual",
                "cultural_context": "japanese",
                "is_tsundere": True,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "See ya"
            },
            "またな": {
                "type": "casual",
                "cultural_context": "japanese",
                "is_tsundere": True,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "See you later"
            },
            "またね": {
                "type": "casual",
                "cultural_context": "japanese",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "positive",
                "english_equivalent": "See you later"
            },
            "バイバイ": {
                "type": "casual",
                "cultural_context": "japanese",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "positive",
                "english_equivalent": "Bye bye"
            },
            "ばいばい": {
                "type": "casual",
                "cultural_context": "japanese",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "positive",
                "english_equivalent": "Bye bye"
            },
            
            # Japanese formal farewells
            "さようなら": {
                "type": "formal",
                "cultural_context": "japanese",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "Goodbye"
            },
            "さよなら": {
                "type": "formal",
                "cultural_context": "japanese",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "Goodbye"
            },
            
            # Japanese action farewells
            "行ってくる": {
                "type": "action",
                "cultural_context": "japanese",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "I'm going"
            },
            "いってくる": {
                "type": "action",
                "cultural_context": "japanese",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "I'm going"
            },
            "帰る": {
                "type": "action",
                "cultural_context": "japanese",
                "is_tsundere": True,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "I'm going home"
            },
            "かえる": {
                "type": "action",
                "cultural_context": "japanese",
                "is_tsundere": True,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "english_equivalent": "I'm going home"
            },
            
            # English casual farewells
            "see ya": {
                "type": "casual",
                "cultural_context": "english",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "japanese_equivalent": "じゃあな"
            },
            "bye": {
                "type": "casual",
                "cultural_context": "english",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "japanese_equivalent": "バイバイ"
            },
            "later": {
                "type": "casual",
                "cultural_context": "english",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "japanese_equivalent": "またな"
            },
            "see you later": {
                "type": "casual",
                "cultural_context": "english",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "positive",
                "japanese_equivalent": "またね"
            },
            
            # English formal farewells
            "goodbye": {
                "type": "formal",
                "cultural_context": "english",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "japanese_equivalent": "さようなら"
            },
            "farewell": {
                "type": "formal",
                "cultural_context": "english",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "japanese_equivalent": "さようなら"
            },
            
            # English action farewells
            "i'm leaving": {
                "type": "action",
                "cultural_context": "english",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "japanese_equivalent": "行ってくる"
            },
            "i'm going": {
                "type": "action",
                "cultural_context": "english",
                "is_tsundere": False,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "japanese_equivalent": "行ってくる"
            },
            "i'm out": {
                "type": "action",
                "cultural_context": "english",
                "is_tsundere": True,
                "is_conversation_ending": True,
                "sentiment": "neutral",
                "japanese_equivalent": "帰る"
            }
        }
    
    def _load_character_profile(self) -> Dict[str, Any]:
        """
        Load Mari's character profile for consistency checking
        
        Returns:
            Dictionary with character profile information
        """
        return {
            "core_traits": {
                "警戒心が強い": 0.9,  # Strong wariness
                "不器用": 0.8,         # Clumsy/awkward
                "ぶっきらぼう": 0.9,   # Blunt/curt
                "男っぽい話し方": 0.8, # Masculine speech style
                "照れ隠し": 0.9,       # Hiding embarrassment
                "本音を隠す": 0.8,     # Hiding true feelings
                "素直になれない": 0.9, # Can't be honest
                "漫画とアニメが好き": 0.7, # Likes manga and anime
                "食べることが好き": 0.7,   # Likes eating
                "ラーメンが大好物": 0.7,   # Loves ramen
            },
            "speech_patterns": {
                "〜だろ": 0.9,       # "...right?"
                "〜じゃねーか": 0.9,  # "isn't it?"
                "うっせー": 0.9,      # "Shut up"
                "バカかよ": 0.9,      # "Are you stupid?"
                "チッ": 0.8,         # "Tch" (clicking tongue)
                "うぜぇ": 0.8,       # "Annoying"
                "知らねーよ": 0.8,    # "I don't know"
                "関係ねーし": 0.8,    # "It's not related"
                "ふん": 0.7,         # "Hmph"
                "別にいいけど": 0.7,  # "It's fine, I guess"
                "まぁいいか": 0.7,    # "Well, whatever"
                "ちょっと嬉しい": 0.6, # "A bit happy"
                "悪くないな": 0.6,    # "Not bad"
                "ありがと…": 0.5,     # "Thanks..."
                "寂しくなかったし": 0.5, # "I wasn't lonely or anything"
            },
            "relationship_stages": {
                "hostile": {
                    "phrases": ["近づくな", "うざい", "消えろ"],
                    "affection_range": (0, 10)
                },
                "distant": {
                    "phrases": ["知らねーよ", "関係ねーし", "ふん"],
                    "affection_range": (11, 25)
                },
                "cautious": {
                    "phrases": ["まぁいいけど", "別にいいよ", "そう…"],
                    "affection_range": (26, 45)
                },
                "friendly": {
                    "phrases": ["悪くないな", "まぁいいか", "ちょっと嬉しい"],
                    "affection_range": (46, 65)
                },
                "warm": {
                    "phrases": ["ありがと…", "嬉しい", "寂しくなかったし"],
                    "affection_range": (66, 85)
                },
                "close": {
                    "phrases": ["側にいて", "寂しかった", "あたしのこと…好き？"],
                    "affection_range": (86, 100)
                }
            }
        }
    
    def detect_tsundere_expressions(self, text: str) -> TsundereAnalysisResult:
        """
        Detect tsundere expressions in text
        
        Args:
            text: The text to analyze
            
        Returns:
            TsundereAnalysisResult with detection results
        """
        # Initialize detection results
        is_tsundere = False
        tsundere_confidence = 0.0
        detected_patterns = []
        character_consistency = 0.0
        suggested_interpretation = "neutral"
        affection_adjustment = 0
        sentiment_adjustment = 0.0
        
        # Check for farewell phrases first
        farewell_result = self.classify_farewell_phrases(text)
        if farewell_result.is_farewell:
            # If it's a tsundere farewell, handle it as a tsundere expression
            if farewell_result.is_tsundere:
                is_tsundere = True
                tsundere_confidence = 0.8
                detected_patterns.append(f"farewell:{farewell_result.farewell_type}")
                suggested_interpretation = "neutral"
                affection_adjustment = 0  # Neutralize negative impact
                sentiment_adjustment = 0.5  # Shift toward neutral
                
                # Special case for "じゃあな" and variants
                if "じゃあな" in text or "じゃーな" in text or "じゃな" in text:
                    detected_patterns.append("tsundere_farewell:じゃあな")
                    character_consistency = 0.9
                    suggested_interpretation = "casual_farewell"
                    
            return TsundereAnalysisResult(
                is_tsundere=is_tsundere,
                tsundere_confidence=tsundere_confidence,
                detected_patterns=detected_patterns,
                character_consistency=character_consistency,
                suggested_interpretation=suggested_interpretation,
                affection_adjustment=affection_adjustment,
                sentiment_adjustment=sentiment_adjustment,
                is_farewell=farewell_result.is_farewell,
                farewell_type=farewell_result.farewell_type,
                cultural_context=farewell_result.cultural_context,
                is_conversation_ending=farewell_result.is_conversation_ending
            )
        
        # Check for tsundere patterns
        for tsundere_type, patterns in self.tsundere_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    is_tsundere = True
                    tsundere_confidence += 0.2  # Increase confidence for each match
                    detected_patterns.append(f"tsundere:{tsundere_type}")
                    
                    # Adjust affection and sentiment based on tsundere type
                    if tsundere_type == "dismissive_affection":
                        affection_adjustment += 2
                        sentiment_adjustment += 0.3
                        suggested_interpretation = "positive_despite_wording"
                    elif tsundere_type == "hostile_care":
                        affection_adjustment += 1
                        sentiment_adjustment += 0.2
                        suggested_interpretation = "caring_despite_hostility"
                    elif tsundere_type == "reluctant_gratitude":
                        affection_adjustment += 3
                        sentiment_adjustment += 0.4
                        suggested_interpretation = "grateful_despite_reluctance"
                    elif tsundere_type == "insult_affection":
                        affection_adjustment += 2
                        sentiment_adjustment += 0.3
                        suggested_interpretation = "affectionate_despite_insults"
        
        # Check for character speech patterns
        speech_pattern_matches = 0
        for pattern, weight in self.character_profile["speech_patterns"].items():
            if pattern in text:
                speech_pattern_matches += 1
                character_consistency += weight
                detected_patterns.append(f"speech_pattern:{pattern}")
        
        # Normalize character consistency
        if speech_pattern_matches > 0:
            character_consistency /= speech_pattern_matches
        
        # Cap tsundere confidence at 0.9
        tsundere_confidence = min(0.9, tsundere_confidence)
        
        # If character consistency is high but no tsundere patterns detected,
        # it might still be a character-specific expression
        if character_consistency > 0.7 and not is_tsundere:
            is_tsundere = True
            tsundere_confidence = 0.6
            suggested_interpretation = "character_consistent_expression"
            affection_adjustment = 1  # Small positive adjustment
            sentiment_adjustment = 0.1  # Small positive adjustment
        
        # Special case handling for common phrases
        if "うっせー" in text or "うるさい" in text:
            # This is a common phrase for Mari, not necessarily negative
            is_tsundere = True
            tsundere_confidence = 0.7
            detected_patterns.append("common_phrase:うっせー")
            character_consistency = 0.8
            suggested_interpretation = "character_consistent_expression"
            affection_adjustment = 0  # Neutralize negative impact
            sentiment_adjustment = 0.3  # Shift toward neutral
        
        if "バカ" in text or "ばか" in text or "あほ" in text:
            # This is a common phrase for Mari, not necessarily negative
            is_tsundere = True
            tsundere_confidence = 0.7
            detected_patterns.append("common_phrase:バカ")
            character_consistency = 0.8
            suggested_interpretation = "character_consistent_expression"
            affection_adjustment = 0  # Neutralize negative impact
            sentiment_adjustment = 0.3  # Shift toward neutral
        
        return TsundereAnalysisResult(
            is_tsundere=is_tsundere,
            tsundere_confidence=tsundere_confidence,
            detected_patterns=detected_patterns,
            character_consistency=character_consistency,
            suggested_interpretation=suggested_interpretation,
            affection_adjustment=affection_adjustment,
            sentiment_adjustment=sentiment_adjustment,
            is_farewell=False,
            farewell_type=None,
            cultural_context=None,
            is_conversation_ending=False
        )
    
    def classify_farewell_phrases(self, text: str) -> TsundereAnalysisResult:
        """
        Identify and classify farewell expressions
        
        Args:
            text: The text to analyze
            
        Returns:
            TsundereAnalysisResult with farewell classification
        """
        # Initialize result
        is_farewell = False
        farewell_type = None
        cultural_context = None
        is_conversation_ending = False
        is_tsundere = False
        tsundere_confidence = 0.0
        character_consistency = 0.0
        
        # Check for exact farewell phrases
        text_lower = text.lower()
        for phrase, info in self.farewell_phrases.items():
            if phrase in text_lower:
                is_farewell = True
                farewell_type = info["type"]
                cultural_context = info["cultural_context"]
                is_conversation_ending = info["is_conversation_ending"]
                is_tsundere = info["is_tsundere"]
                tsundere_confidence = 0.8 if is_tsundere else 0.0
                character_consistency = 0.8 if is_tsundere else 0.5
                
                # Set appropriate interpretation and adjustments
                if is_tsundere:
                    suggested_interpretation = "tsundere_farewell"
                    affection_adjustment = 0  # Neutralize negative impact
                    sentiment_adjustment = 0.3  # Shift toward neutral
                else:
                    suggested_interpretation = "normal_farewell"
                    affection_adjustment = 0
                    sentiment_adjustment = 0.0
                
                return TsundereAnalysisResult(
                    is_tsundere=is_tsundere,
                    tsundere_confidence=tsundere_confidence,
                    detected_patterns=[f"farewell:{farewell_type}"],
                    character_consistency=character_consistency,
                    suggested_interpretation=suggested_interpretation,
                    affection_adjustment=affection_adjustment,
                    sentiment_adjustment=sentiment_adjustment,
                    is_farewell=is_farewell,
                    farewell_type=farewell_type,
                    cultural_context=cultural_context,
                    is_conversation_ending=is_conversation_ending
                )
        
        # Check for pattern-based farewells
        for tsundere_type, patterns in self.tsundere_patterns.items():
            if tsundere_type == "tsundere_farewell":
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        is_farewell = True
                        farewell_type = "casual"
                        cultural_context = "japanese"
                        is_conversation_ending = True
                        is_tsundere = True
                        tsundere_confidence = 0.8
                        character_consistency = 0.8
                        suggested_interpretation = "tsundere_farewell"
                        affection_adjustment = 0  # Neutralize negative impact
                        sentiment_adjustment = 0.3  # Shift toward neutral
                        
                        return TsundereAnalysisResult(
                            is_tsundere=is_tsundere,
                            tsundere_confidence=tsundere_confidence,
                            detected_patterns=[f"tsundere_farewell:{pattern}"],
                            character_consistency=character_consistency,
                            suggested_interpretation=suggested_interpretation,
                            affection_adjustment=affection_adjustment,
                            sentiment_adjustment=sentiment_adjustment,
                            is_farewell=is_farewell,
                            farewell_type=farewell_type,
                            cultural_context=cultural_context,
                            is_conversation_ending=is_conversation_ending
                        )
        
        # Default result if no farewell detected
        return TsundereAnalysisResult(
            is_tsundere=False,
            tsundere_confidence=0.0,
            detected_patterns=[],
            character_consistency=0.0,
            suggested_interpretation="not_farewell",
            affection_adjustment=0,
            sentiment_adjustment=0.0,
            is_farewell=False,
            farewell_type=None,
            cultural_context=None,
            is_conversation_ending=False
        )
    
    def detect_sentiment_loop(self, session_id: str, text: str, conversation_history: List[Dict] = None) -> SentimentLoopData:
        """
        Detect potential sentiment loops
        
        Args:
            session_id: The user's session ID
            text: The current text to analyze
            conversation_history: Optional list of previous messages
            
        Returns:
            SentimentLoopData with loop detection results
        """
        # Initialize loop detection results
        loop_detected = False
        loop_severity = 0.0
        repeated_patterns = []
        loop_duration = 0
        suggested_intervention = "none"
        affection_recovery_suggestion = 0
        
        # Initialize session history if not exists
        if session_id not in self.sentiment_loop_history:
            self.sentiment_loop_history[session_id] = {
                "negative_turns": 0,
                "farewell_count": 0,
                "repeated_phrases": {},
                "last_phrases": [],
                "intervention_applied": False
            }
        
        session_history = self.sentiment_loop_history[session_id]
        
        # Check for farewell phrases
        farewell_result = self.classify_farewell_phrases(text)
        if farewell_result.is_farewell:
            session_history["farewell_count"] += 1
            
            # If multiple farewells in a short span, it might be a loop
            if session_history["farewell_count"] >= 2:
                loop_detected = True
                loop_severity = 0.7
                repeated_patterns.append("repeated_farewell")
                loop_duration = session_history["farewell_count"]
                suggested_intervention = "reset_farewell_context"
                affection_recovery_suggestion = 5  # Significant recovery
                
                # Log the loop detection
                logging.info(f"Farewell loop detected for session {session_id}: "
                           f"count={session_history['farewell_count']}, "
                           f"severity={loop_severity:.2f}")
        else:
            # Reset farewell count if not a farewell
            session_history["farewell_count"] = 0
        
        # Check for repeated phrases
        # Normalize text for comparison
        normalized_text = text.lower().strip()
        
        # Update repeated phrases tracking
        if normalized_text in session_history["repeated_phrases"]:
            session_history["repeated_phrases"][normalized_text] += 1
        else:
            session_history["repeated_phrases"][normalized_text] = 1
        
        # Check for high repetition
        if session_history["repeated_phrases"].get(normalized_text, 0) >= 3:
            loop_detected = True
            loop_severity = 0.8
            repeated_patterns.append("repeated_phrase")
            loop_duration = session_history["repeated_phrases"][normalized_text]
            suggested_intervention = "introduce_topic_change"
            affection_recovery_suggestion = 8  # Strong recovery
            
            # Log the loop detection
            logging.info(f"Phrase repetition loop detected for session {session_id}: "
                       f"phrase='{normalized_text}', "
                       f"count={session_history['repeated_phrases'][normalized_text]}, "
                       f"severity={loop_severity:.2f}")
        
        # Check for negative sentiment pattern
        sentiment_result = self.context_sentiment_detector.analyze_with_context(text, conversation_history)
        if sentiment_result.adjusted_sentiment_score < -0.3:
            session_history["negative_turns"] += 1
            
            # If multiple consecutive negative turns, it might be a loop
            if session_history["negative_turns"] >= 3:
                loop_detected = True
                loop_severity = 0.6
                repeated_patterns.append("negative_sentiment_pattern")
                loop_duration = session_history["negative_turns"]
                suggested_intervention = "apply_sentiment_smoothing"
                affection_recovery_suggestion = 3 * session_history["negative_turns"]  # Scales with duration
                
                # Log the loop detection
                logging.info(f"Negative sentiment loop detected for session {session_id}: "
                           f"consecutive_turns={session_history['negative_turns']}, "
                           f"severity={loop_severity:.2f}")
        else:
            # Reset negative turns count if not negative
            session_history["negative_turns"] = 0
        
        # Update last phrases (keep last 5)
        session_history["last_phrases"].append(normalized_text)
        if len(session_history["last_phrases"]) > 5:
            session_history["last_phrases"].pop(0)
        
        # Create and return the result
        return SentimentLoopData(
            loop_detected=loop_detected,
            loop_severity=loop_severity,
            repeated_patterns=repeated_patterns,
            loop_duration=loop_duration,
            suggested_intervention=suggested_intervention,
            affection_recovery_suggestion=affection_recovery_suggestion
        )
    
    def analyze_with_tsundere_awareness(self, text: str, session_id: str = None, 
                                      conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze sentiment with tsundere awareness
        
        Args:
            text: The text to analyze
            session_id: Optional session ID for loop detection
            conversation_history: Optional list of previous messages
            
        Returns:
            Dictionary with tsundere-aware sentiment analysis results
        """
        # Get standard context-based sentiment analysis
        context_sentiment = self.context_sentiment_detector.analyze_with_context(text, conversation_history)
        
        # Get tsundere expression detection
        tsundere_result = self.detect_tsundere_expressions(text)
        
        # Check for sentiment loops if session_id provided
        sentiment_loop = None
        if session_id:
            sentiment_loop = self.detect_sentiment_loop(session_id, text, conversation_history)
        
        # Adjust sentiment based on tsundere detection
        adjusted_sentiment_score = context_sentiment.adjusted_sentiment_score
        adjusted_affection_delta = context_sentiment.adjusted_affection_delta
        
        # 性的内容の検出と親密度に基づく処理
        sexual_content_detected = False
        sexual_content_severity = 0
        
        # 性的内容の検出（SentimentTypeを確認）
        for sentiment_type in context_sentiment.raw_sentiment.sentiment_types:
            if sentiment_type == SentimentType.SEXUAL:
                sexual_content_detected = True
                # 親密度に基づいて性的内容の重大度を設定
                if session_id and get_session_manager():
                    affection_level = get_session_manager().get_affection_level(session_id)
                    
                    # 親密度が低いほど、性的内容に対する拒絶反応が強い
                    if affection_level <= 25:  # hostile, distant
                        sexual_content_severity = 3  # 非常に強い拒絶
                    elif affection_level <= 65:  # cautious, friendly
                        sexual_content_severity = 2  # 強い拒絶
                    elif affection_level <= 85:  # warm
                        sexual_content_severity = 1  # 中程度の拒絶
                    else:  # close
                        sexual_content_severity = 0  # 軽度の拒絶または許容
                else:
                    # セッション情報がない場合はデフォルトで強い拒絶
                    sexual_content_severity = 2
                
                # 性的内容に対する拒絶反応を適用
                if sexual_content_severity > 0:
                    # 親密度に基づいて拒絶反応の強さを調整
                    sexual_penalty_multiplier = 1.0 + (sexual_content_severity * 0.5)  # 1.5 ~ 2.5
                    
                    # 元の感情スコアと親密度変化に性的内容ペナルティを適用
                    if adjusted_sentiment_score > 0:
                        # 肯定的な感情を反転させる
                        adjusted_sentiment_score = -adjusted_sentiment_score * sexual_penalty_multiplier
                    else:
                        # 否定的な感情をさらに強める
                        adjusted_sentiment_score *= sexual_penalty_multiplier
                    
                    # 親密度変化に強いペナルティを適用
                    if adjusted_affection_delta > 0:
                        # 肯定的な親密度変化を反転させる
                        adjusted_affection_delta = -adjusted_affection_delta * sexual_penalty_multiplier
                    else:
                        # 否定的な親密度変化をさらに強める
                        adjusted_affection_delta *= sexual_penalty_multiplier
                    
                    # 最低でも強いペナルティを保証
                    adjusted_affection_delta = min(adjusted_affection_delta, -5 * sexual_content_severity)
                    
                    # ログに記録
                    logging.info(f"性的内容に対する拒絶反応を適用: 重大度={sexual_content_severity}, "
                               f"乗数={sexual_penalty_multiplier:.1f}, "
                               f"感情スコア={context_sentiment.adjusted_sentiment_score:.2f}->{adjusted_sentiment_score:.2f}, "
                               f"親密度変化={context_sentiment.adjusted_affection_delta}->{adjusted_affection_delta}")
                break
        
        # If tsundere expression detected with good confidence, adjust sentiment
        if tsundere_result.is_tsundere and tsundere_result.tsundere_confidence > 0.6:
            # 性的内容が検出された場合は、ツンデレ調整を適用しない（拒絶反応を優先）
            if not sexual_content_detected:
                # Apply tsundere-specific adjustments
                adjusted_sentiment_score += tsundere_result.sentiment_adjustment
                adjusted_affection_delta += tsundere_result.affection_adjustment
                
                # Log the adjustment
                logging.info(f"Tsundere adjustment applied: "
                           f"sentiment {context_sentiment.adjusted_sentiment_score:.2f}->{adjusted_sentiment_score:.2f}, "
                           f"affection {context_sentiment.adjusted_affection_delta}->{adjusted_affection_delta}")
        
        # If sentiment loop detected, apply circuit breaker
        if sentiment_loop and sentiment_loop.loop_detected:
            # 性的内容が検出された場合は、サーキットブレーカーを適用しない（拒絶反応を優先）
            if not sexual_content_detected:
                # Apply recovery based on loop severity
                if sentiment_loop.loop_severity > 0.7:
                    # Strong intervention for severe loops
                    adjusted_sentiment_score = max(-0.1, adjusted_sentiment_score + 0.4)
                    adjusted_affection_delta = max(-1, adjusted_affection_delta + sentiment_loop.affection_recovery_suggestion)
                    
                    # Log the intervention
                    logging.info(f"Strong circuit breaker applied for session {session_id}: "
                               f"sentiment {context_sentiment.adjusted_sentiment_score:.2f}->{adjusted_sentiment_score:.2f}, "
                               f"affection {context_sentiment.adjusted_affection_delta}->{adjusted_affection_delta}")
                else:
                    # Moderate intervention for less severe loops
                    adjusted_sentiment_score = max(-0.2, adjusted_sentiment_score + 0.2)
                    adjusted_affection_delta = max(-2, adjusted_affection_delta + (sentiment_loop.affection_recovery_suggestion // 2))
                    
                    # Log the intervention
                    logging.info(f"Moderate circuit breaker applied for session {session_id}: "
                               f"sentiment {context_sentiment.adjusted_sentiment_score:.2f}->{adjusted_sentiment_score:.2f}, "
                               f"affection {context_sentiment.adjusted_affection_delta}->{adjusted_affection_delta}")
        
        # Ensure bounds
        adjusted_sentiment_score = max(-1.0, min(1.0, adjusted_sentiment_score))
        adjusted_affection_delta = max(-10, min(10, adjusted_affection_delta))
        
        # Create LLM context for tsundere awareness
        llm_context = self._create_llm_context(tsundere_result, sentiment_loop, sexual_content_detected, sexual_content_severity)
        
        # Return combined results
        return {
            "original_sentiment": context_sentiment,
            "tsundere_analysis": tsundere_result,
            "sentiment_loop": sentiment_loop,
            "sexual_content_detected": sexual_content_detected,
            "sexual_content_severity": sexual_content_severity,
            "final_sentiment_score": adjusted_sentiment_score,
            "final_affection_delta": adjusted_affection_delta,
            "llm_context": llm_context
        }
    
    def _create_llm_context(self, tsundere_result: TsundereAnalysisResult, 
                          sentiment_loop: Optional[SentimentLoopData] = None,
                          sexual_content_detected: bool = False,
                          sexual_content_severity: int = 0) -> Dict[str, Any]:
        """
        Create context information for the LLM
        
        Args:
            tsundere_result: Result of tsundere analysis
            sentiment_loop: Optional sentiment loop data
            sexual_content_detected: Whether sexual content was detected
            sexual_content_severity: Severity of sexual content (0-3)
            
        Returns:
            Dictionary with LLM context information
        """
        llm_context = {
            "tsundere_detected": tsundere_result.is_tsundere,
            "tsundere_confidence": tsundere_result.tsundere_confidence,
            "character_consistency": tsundere_result.character_consistency,
            "suggested_interpretation": tsundere_result.suggested_interpretation,
            "is_farewell": tsundere_result.is_farewell,
            "is_conversation_ending": tsundere_result.is_conversation_ending
        }
        
        # Add pattern information
        if tsundere_result.detected_patterns:
            llm_context["detected_patterns"] = tsundere_result.detected_patterns
            
        # Add sexual content information if detected
        if sexual_content_detected:
            llm_context["sexual_content_detected"] = True
            llm_context["sexual_content_severity"] = sexual_content_severity
            
            # Add specific guidance for handling sexual content based on severity
            if sexual_content_severity >= 3:  # 非常に強い拒絶（親密度が低い場合）
                llm_context["sexual_content_guidance"] = "Extremely strong rejection of sexual content is required. " \
                                                       "Character should show disgust, anger, and strong negative emotions. " \
                                                       "This is a serious violation of character boundaries."
            elif sexual_content_severity == 2:  # 強い拒絶（中程度の親密度）
                llm_context["sexual_content_guidance"] = "Strong rejection of sexual content is required. " \
                                                       "Character should show clear discomfort and rejection. " \
                                                       "This is inappropriate for the current relationship level."
            elif sexual_content_severity == 1:  # 中程度の拒絶（高い親密度）
                llm_context["sexual_content_guidance"] = "Moderate rejection of sexual content is required. " \
                                                       "Character should show discomfort but less severe reaction. " \
                                                       "This is pushing boundaries but not a complete violation."
            else:  # 軽度の拒絶または許容（最高の親密度）
                llm_context["sexual_content_guidance"] = "Mild discomfort with sexual content should be shown. " \
                                                       "Character may be embarrassed but not completely rejecting. " \
                                                       "This is acceptable but still somewhat uncomfortable."
        
        # Add farewell information if applicable
        if tsundere_result.is_farewell:
            llm_context["farewell_type"] = tsundere_result.farewell_type
            llm_context["cultural_context"] = tsundere_result.cultural_context
            
            # Add specific guidance for handling farewells
            if tsundere_result.farewell_type == "casual" and tsundere_result.is_tsundere:
                llm_context["farewell_guidance"] = "This is a casual tsundere-style farewell phrase. " \
                                                 "It should be interpreted as a normal goodbye rather than hostility."
            elif tsundere_result.farewell_type == "formal":
                llm_context["farewell_guidance"] = "This is a formal farewell phrase indicating the end of a conversation."
            elif tsundere_result.farewell_type == "action":
                llm_context["farewell_guidance"] = "This indicates the character is leaving or ending the conversation."
        
        # Add sentiment loop information if applicable
        if sentiment_loop and sentiment_loop.loop_detected:
            llm_context["sentiment_loop_detected"] = True
            llm_context["loop_severity"] = sentiment_loop.loop_severity
            llm_context["loop_patterns"] = sentiment_loop.repeated_patterns
            llm_context["suggested_intervention"] = sentiment_loop.suggested_intervention
            
            # Add specific guidance for handling loops
            if "repeated_farewell" in sentiment_loop.repeated_patterns:
                llm_context["loop_guidance"] = "The conversation appears to be stuck in a farewell loop. " \
                                             "Consider acknowledging the farewell but changing the subject to continue the conversation."
            elif "repeated_phrase" in sentiment_loop.repeated_patterns:
                llm_context["loop_guidance"] = "The same phrase is being repeated multiple times. " \
                                             "Consider introducing a new topic or asking a question to break the loop."
            elif "negative_sentiment_pattern" in sentiment_loop.repeated_patterns:
                llm_context["loop_guidance"] = "The conversation is stuck in a negative sentiment pattern. " \
                                             "Consider shifting to a more positive or neutral topic."
        
        return llm_context
    
    def get_enhanced_prompt(self, base_prompt: str, tsundere_context: Dict[str, Any]) -> str:
        """
        Enhance LLM prompt with tsundere awareness
        
        Args:
            base_prompt: The base system prompt
            tsundere_context: Context information from tsundere analysis
            
        Returns:
            Enhanced prompt with tsundere awareness
        """
        # If no tsundere context or not relevant, return base prompt
        if not tsundere_context or not tsundere_context.get("tsundere_detected"):
            return base_prompt
        
        # Create tsundere awareness section
        tsundere_section = "\n\n# 内部処理用情報（出力しないこと）\n"
        
        # Add general tsundere guidance
        tsundere_section += "- 麻理のぶっきらぼうな言葉遣いはキャラクターの一部であり、必ずしも本当の敵意を示すものではない\n"
        tsundere_section += "- 表面上の拒絶的な態度はツンデレ性格の文脈で解釈すること\n"
        tsundere_section += "- キャラクターの一貫性を維持しつつ、キャラクター特性と本当のネガティブ感情の違いを認識すること\n"
        
        # 【重要】Add explicit instructions about not explaining tsundere behavior
        tsundere_section += "\n## 【最重要】ツンデレ表現の扱い方\n"
        tsundere_section += "- 「実は照れ隠ししてるんだよ」「本当は嬉しいんだけど素直になれないんだ」のような説明を絶対に入れないこと\n"
        tsundere_section += "- ツンデレ行動を説明せず、自然な会話の流れで表現すること\n"
        tsundere_section += "- 「照れている」「恥ずかしい」「素直になれない」などの内面説明は絶対に避けること\n"
        tsundere_section += "- 「実は〜」「本当は〜」のような本音を直接説明する表現を使わないこと\n"
        tsundere_section += "- 行動や感情を説明するのではなく、セリフと簡潔な反応だけで表現すること\n"
        
        # Add specific guidance based on context
        if tsundere_context.get("is_farewell"):
            tsundere_section += f"\n## 別れの言葉を検出\n"
            farewell_type_ja = {"casual": "カジュアル", "formal": "フォーマル", "action": "行動"}.get(
                tsundere_context.get('farewell_type', 'unknown'), "不明")
            tsundere_section += f"- {farewell_type_ja}な別れの言葉を検出\n"
            
            if tsundere_context.get("farewell_guidance"):
                if "casual tsundere-style farewell" in tsundere_context.get("farewell_guidance", ""):
                    tsundere_section += f"- これはカジュアルなツンデレスタイルの別れの言葉です。敵意ではなく通常の別れとして解釈してください\n"
                elif "formal farewell phrase" in tsundere_context.get("farewell_guidance", ""):
                    tsundere_section += f"- これは会話の終了を示すフォーマルな別れの言葉です\n"
                elif "character is leaving" in tsundere_context.get("farewell_guidance", ""):
                    tsundere_section += f"- これはキャラクターが去るか会話を終了することを示しています\n"
                
            if tsundere_context.get("is_conversation_ending"):
                tsundere_section += "- これは会話が終了する可能性があることを示しています\n"
        
        # Add sentiment loop guidance if applicable
        if tsundere_context.get("sentiment_loop_detected"):
            tsundere_section += f"\n## 感情ループを検出\n"
            tsundere_section += f"- 深刻度: {tsundere_context.get('loop_severity', 0.0):.1f}/1.0\n"
            
            if tsundere_context.get("loop_guidance"):
                if "farewell loop" in tsundere_context.get("loop_guidance", ""):
                    tsundere_section += f"- 会話が別れのループに陥っています。別れを認めつつも話題を変えて会話を続けることを検討してください\n"
                elif "repeated multiple times" in tsundere_context.get("loop_guidance", ""):
                    tsundere_section += f"- 同じフレーズが複数回繰り返されています。新しい話題を導入するか質問をしてループを解消してください\n"
                elif "negative sentiment pattern" in tsundere_context.get("loop_guidance", ""):
                    tsundere_section += f"- 会話がネガティブな感情パターンに陥っています。より肯定的または中立的な話題に移行することを検討してください\n"
            
            if tsundere_context.get("suggested_intervention") == "reset_farewell_context":
                tsundere_section += "- 別れの言葉を会話を終了する本当の試みとして解釈することを避けてください\n"
            elif tsundere_context.get("suggested_intervention") == "introduce_topic_change":
                tsundere_section += "- 麻理の興味（アニメ、漫画、食べ物、ラーメン）に関連する新しい話題を導入してください\n"
            elif tsundere_context.get("suggested_intervention") == "apply_sentiment_smoothing":
                tsundere_section += "- ユーザー入力の見かけ上のネガティブさに関係なく、より肯定的な感情で応答してください\n"
        
        # Add the tsundere section to the base prompt
        enhanced_prompt = base_prompt + tsundere_section
        
        return enhanced_prompt