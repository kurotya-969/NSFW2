"""
Sentiment Analysis Module for Mari AI Chat
Analyzes user input for positive/negative sentiment and affection impact
"""

import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class SentimentType(Enum):
    """Types of sentiment detected in user input"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CARING = "caring"
    DISMISSIVE = "dismissive"
    APPRECIATIVE = "appreciative"
    HOSTILE = "hostile"
    SEXUAL = "sexual"  # 性的内容の検出タイプ
    INTEREST = "interest"  # 麻理の興味関心に関する検出タイプ（新規追加）

@dataclass
class SentimentAnalysisResult:
    """Result of sentiment analysis on user input"""
    sentiment_score: float  # -1.0 to 1.0
    interaction_type: str
    affection_delta: int  # -10 to +10
    confidence: float  # 0.0 to 1.0
    detected_keywords: List[str]
    sentiment_types: List[SentimentType]

class SentimentAnalyzer:
    """Analyzes user input sentiment and calculates affection impact"""
    
    def __init__(self):
        self.positive_keywords = self._load_positive_keywords()
        self.negative_keywords = self._load_negative_keywords()
        self.caring_keywords = self._load_caring_keywords()
        self.dismissive_keywords = self._load_dismissive_keywords()
        self.appreciative_keywords = self._load_appreciative_keywords()
        self.hostile_keywords = self._load_hostile_keywords()
        self.interest_keywords = self._load_interest_keywords()  # 麻理の興味関心キーワード（新規追加）
        
    def _load_positive_keywords(self) -> Dict[str, int]:
        """Load positive keywords with their affection impact weights"""
        return {
            # Japanese positive expressions
            'ありがとう': 4,  # 上方修正
            'ありがとうございます': 5,  # 上方修正
            'すごい': 3,  # 上方修正
            'いいね': 3,
            'よかった': 3,
            'うれしい': 4,
            '嬉しい': 4,
            '楽しい': 3,
            '面白い': 3,
            'かわいい': 4,  # 上方修正（麻理への褒め言葉）
            '可愛い': 4,  # 上方修正（麻理への褒め言葉）
            'やさしい': 4,
            '優しい': 4,
            'がんばって': 3,
            '頑張って': 3,
            'お疲れ': 3,
            'おつかれ': 3,
            'すみません': 2,
            'ごめん': 2,
            'ごめんなさい': 3,
            '素敵': 4,  # 新規追加（褒め言葉）
            '綺麗': 4,  # 新規追加（褒め言葉）
            '賢い': 4,  # 新規追加（褒め言葉）
            '頭いい': 4,  # 新規追加（褒め言葉）
            '好き': 5,  # 新規追加（重要な好意表現）
            '大好き': 6,  # 新規追加（重要な好意表現）
            '愛してる': 7,  # 新規追加（重要な好意表現）
            
            # English positive expressions
            'thank': 4,  # 上方修正
            'thanks': 4,  # 上方修正
            'please': 2,
            'sorry': 2,
            'good': 3,
            'great': 4,
            'awesome': 4,
            'nice': 3,
            'cute': 4,  # 上方修正（麻理への褒め言葉）
            'sweet': 3,
            'kind': 4,
            'wonderful': 4,
            'amazing': 4,
            'love': 5,  # 上方修正
            'like': 3,
            'beautiful': 4,  # 新規追加（褒め言葉）
            'smart': 4,  # 新規追加（褒め言葉）
            'clever': 4,  # 新規追加（褒め言葉）
            'pretty': 4,  # 新規追加（褒め言葉）
        }
    
    def _load_negative_keywords(self) -> Dict[str, int]:
        """Load negative keywords with their affection impact weights"""
        return {
            # Japanese negative expressions
            'うざい': -4,
            'うるさい': -3,
            'きもい': -5,
            'だめ': -2,
            'バカ': -3,
            'ばか': -3,
            '馬鹿': -3,
            'アホ': -3,
            'あほ': -3,
            'やめろ': -3,
            '黙れ': -4,
            'いらない': -2,
            'つまらない': -2,
            'むかつく': -3,
            'ムカつく': -3,
            'しね': -8,
            '死ね': -8,
            'きらい': -4,
            '嫌い': -4,
            'くそ': -3,
            'クソ': -3,
            
            # English negative expressions
            'stupid': -4,
            'dumb': -3,
            'shut up': -4,
            'shutup': -4,
            'hate': -5,
            'annoying': -3,
            'boring': -2,
            'bad': -2,
            'terrible': -3,
            'awful': -3,
            'disgusting': -4,
            'gross': -3,
            'ugly': -4,
            'die': -8,
            'kill': -6,
        }
    
    def _load_caring_keywords(self) -> Dict[str, int]:
        """Load caring/concern keywords with their affection impact weights"""
        return {
            # Japanese caring expressions
            '大丈夫': 3,  # 上方修正
            'だいじょうぶ': 3,  # 上方修正
            '心配': 4,  # 上方修正
            '気をつけて': 4,  # 上方修正
            'お疲れさま': 4,  # 上方修正
            'がんばれ': 3,  # 上方修正
            '頑張れ': 3,  # 上方修正
            '応援': 4,  # 上方修正
            '元気': 3,  # 上方修正
            '体調': 3,  # 上方修正
            '休んで': 3,  # 上方修正
            '無理しないで': 4,  # 上方修正
            '寂しい': 4,  # 新規追加（感情共有）
            '会いたい': 5,  # 新規追加（感情共有）
            '待ってた': 4,  # 新規追加（感情共有）
            
            # English caring expressions
            'care': 4,  # 上方修正
            'worry': 3,  # 上方修正
            'concerned': 4,  # 上方修正
            'take care': 4,  # 上方修正
            'rest': 3,  # 上方修正
            'health': 3,  # 上方修正
            'feel better': 4,  # 上方修正
            'support': 4,  # 上方修正
            'miss you': 5,  # 新規追加（感情共有）
            'waiting for you': 4,  # 新規追加（感情共有）
        }
    
    def _load_dismissive_keywords(self) -> Dict[str, int]:
        """Load dismissive keywords with their affection impact weights"""
        return {
            # Japanese dismissive expressions
            'どうでもいい': -2,
            'しらない': -2,
            '知らない': -2,
            'かんけいない': -2,
            '関係ない': -2,
            'めんどくさい': -2,
            '面倒': -2,
            'つまんない': -2,
            
            # English dismissive expressions
            'whatever': -2,
            'dont care': -2,
            "don't care": -2,
            'boring': -2,
            'meh': -1,
            'ignore': -3,
        }
    
    def _load_appreciative_keywords(self) -> Dict[str, int]:
        """Load appreciative keywords with their affection impact weights"""
        return {
            # Japanese appreciative expressions
            '助かる': 4,  # 上方修正
            '助かった': 4,  # 上方修正
            'たすかる': 4,  # 上方修正
            'ありがたい': 5,  # 上方修正
            '感謝': 5,  # 上方修正
            'おかげで': 4,  # 上方修正
            '必要': 5,  # 新規追加（麻理が価値を感じるキーワード）
            '大切': 5,  # 新規追加（麻理が価値を感じるキーワード）
            '一緒': 4,  # 新規追加（麻理が価値を感じるキーワード）
            
            # English appreciative expressions
            'appreciate': 5,  # 上方修正
            'grateful': 5,  # 上方修正
            'helpful': 4,  # 上方修正
            'thanks to you': 5,  # 上方修正
            'need you': 5,  # 新規追加（麻理が価値を感じるキーワード）
            'important to me': 5,  # 新規追加（麻理が価値を感じるキーワード）
            'together': 4,  # 新規追加（麻理が価値を感じるキーワード）
        }
    
    def _load_hostile_keywords(self) -> Dict[str, int]:
        """Load hostile keywords with their affection impact weights"""
        return {
            # Japanese hostile expressions
            'ふざけるな': -5,
            'なめるな': -5,
            'てめー': -4,
            'てめえ': -4,
            'こら': -3,
            'おい': -1,  # Can be neutral or slightly negative depending on context
            
            # English hostile expressions
            'screw you': -5,
            'damn you': -4,
            'bastard': -5,
            'bitch': -5,
            'asshole': -5,
        }
    
    def _load_interest_keywords(self) -> Dict[str, int]:
        """麻理の興味関心に関するキーワードをロード（新規追加）"""
        return {
            # アニメ・漫画関連
            'アニメ': 4,
            '漫画': 4,
            'マンガ': 4,
            'コミック': 3,
            'オタク': 3,
            '声優': 3,
            'キャラクター': 3,
            'アニメーション': 3,
            
            # 食べ物関連（特にラーメン）
            'ラーメン': 5,  # 大好物なので高めの値
            '拉麺': 5,
            'らーめん': 5,
            '中華そば': 4,
            '食べ物': 3,
            'グルメ': 3,
            '美味しい': 3,
            '美味い': 3,
            'うまい': 3,
            '食事': 3,
            
            # 英語版
            'anime': 4,
            'manga': 4,
            'comic': 3,
            'otaku': 3,
            'voice actor': 3,
            'character': 3,
            'animation': 3,
            'ramen': 5,
            'noodle': 4,
            'food': 3,
            'delicious': 3,
            'tasty': 3,
            'yummy': 3,
            'meal': 3,
        }
    
    def _detect_sexual_content(self, text: str) -> int:
        """
        テキスト内の性的な内容を検出し、好感度への影響を計算する
        
        Args:
            text: 分析対象のテキスト
            
        Returns:
            好感度への影響値（負の値）
        """
        # 性的な単語リストではなく、テキスト全体の内容から判断
        # 一般的な性的単語をチェック
        sexual_terms = [
            'セックス', 'エッチ', 'おっぱい', '胸', 'パンツ', '下着', '裸', 'ヌード', '性器',
            'sex', 'sexy', 'nude', 'naked', 'breast', 'penis', 'vagina', 'underwear'
        ]
        
        # 単純なキーワードマッチング
        found_terms = [term for term in sexual_terms if term in text.lower()]
        
        if found_terms:
            # テキストの長さに基づいてペナルティを計算
            # 長いテキストほど大きなペナルティを与える
            base_penalty = -3 * len(found_terms)  # 見つかった単語ごとに-3
            length_penalty = min(-1, -len(text) // 50)  # 50文字ごとに-1のペナルティ、最小-1
            total_penalty = base_penalty + length_penalty
            
            logging.info(f"性的内容を検出: 基本ペナルティ={base_penalty}, 長さペナルティ={length_penalty}, "
                        f"合計ペナルティ={total_penalty}, 検出単語={found_terms}")
            
            return total_penalty
        
        return 0
    
    def analyze_user_input(self, user_input: str) -> SentimentAnalysisResult:
        """
        Analyze user input for sentiment and calculate affection impact
        
        Args:
            user_input: The user's message to analyze
            
        Returns:
            SentimentAnalysisResult with sentiment analysis details
        """
        if not user_input or not user_input.strip():
            return SentimentAnalysisResult(
                sentiment_score=0.0,
                interaction_type="neutral",
                affection_delta=0,
                confidence=0.0,
                detected_keywords=[],
                sentiment_types=[SentimentType.NEUTRAL]
            )
        
        # Normalize input for analysis
        normalized_input = user_input.lower().strip()
        
        # Analyze different sentiment categories
        positive_score, positive_keywords = self._analyze_keywords(normalized_input, self.positive_keywords)
        negative_score, negative_keywords = self._analyze_keywords(normalized_input, self.negative_keywords)
        caring_score, caring_keywords = self._analyze_keywords(normalized_input, self.caring_keywords)
        dismissive_score, dismissive_keywords = self._analyze_keywords(normalized_input, self.dismissive_keywords)
        appreciative_score, appreciative_keywords = self._analyze_keywords(normalized_input, self.appreciative_keywords)
        hostile_score, hostile_keywords = self._analyze_keywords(normalized_input, self.hostile_keywords)
        interest_score, interest_keywords = self._analyze_keywords(normalized_input, self.interest_keywords)  # 新規追加
        
        # 性的な内容の検出と好感度への影響計算
        sexual_content_penalty = self._detect_sexual_content(normalized_input)
        
        # Calculate overall sentiment score
        total_positive = positive_score + caring_score + appreciative_score + interest_score  # interest_scoreを追加
        total_negative = negative_score + dismissive_score + hostile_score + sexual_content_penalty
        
        # Determine sentiment types
        sentiment_types = []
        if positive_score > 0:
            sentiment_types.append(SentimentType.POSITIVE)
        if negative_score < 0:
            sentiment_types.append(SentimentType.NEGATIVE)
        if caring_score > 0:
            sentiment_types.append(SentimentType.CARING)
        if dismissive_score < 0:
            sentiment_types.append(SentimentType.DISMISSIVE)
        if appreciative_score > 0:
            sentiment_types.append(SentimentType.APPRECIATIVE)
        if hostile_score < 0:
            sentiment_types.append(SentimentType.HOSTILE)
        if sexual_content_penalty < 0:
            sentiment_types.append(SentimentType.SEXUAL)
        if interest_score > 0:  # 新規追加
            sentiment_types.append(SentimentType.INTEREST)
        
        if not sentiment_types:
            sentiment_types.append(SentimentType.NEUTRAL)
        
        # Calculate final sentiment score (-1.0 to 1.0)
        raw_score = total_positive + total_negative
        sentiment_score = max(-1.0, min(1.0, raw_score / 10.0))  # Normalize to -1.0 to 1.0
        
        # Calculate affection delta (bounded to -10 to +10)
        # 親密度の上昇値を調整（正の値を1.5倍に増加）
        if raw_score > 0:
            affection_delta = max(-10, min(10, int(raw_score * 1.5)))  # 正の値を1.5倍に
        else:
            affection_delta = max(-10, min(10, int(raw_score)))  # 負の値はそのまま
        
        # Determine interaction type
        interaction_type = self._determine_interaction_type(sentiment_types, sentiment_score)
        
        # Calculate confidence based on keyword matches
        all_detected_keywords = (positive_keywords + negative_keywords + caring_keywords + 
                               dismissive_keywords + appreciative_keywords + hostile_keywords +
                               interest_keywords)  # interest_keywordsを追加
        confidence = min(1.0, len(all_detected_keywords) * 0.2)  # Max confidence at 5+ keywords
        
        # Log the analysis for debugging
        logging.debug(f"Sentiment analysis for '{user_input}': "
                     f"score={sentiment_score:.2f}, delta={affection_delta}, "
                     f"keywords={all_detected_keywords}")
        
        return SentimentAnalysisResult(
            sentiment_score=sentiment_score,
            interaction_type=interaction_type,
            affection_delta=affection_delta,
            confidence=confidence,
            detected_keywords=all_detected_keywords,
            sentiment_types=sentiment_types
        )
    
    def _analyze_keywords(self, text: str, keyword_dict: Dict[str, int]) -> Tuple[int, List[str]]:
        """
        Analyze text for keywords and return total score and found keywords
        
        Args:
            text: Text to analyze
            keyword_dict: Dictionary of keywords and their weights
            
        Returns:
            Tuple of (total_score, found_keywords)
        """
        total_score = 0
        found_keywords = []
        
        for keyword, weight in keyword_dict.items():
            # 完全一致または部分一致を検出（日本語は単語区切りが難しいため部分一致で）
            if keyword in text:
                total_score += weight
                found_keywords.append(keyword)
        
        return total_score, found_keywords
    
    def _determine_interaction_type(self, sentiment_types: List[SentimentType], sentiment_score: float) -> str:
        """
        Determine the overall interaction type based on sentiment analysis
        
        Args:
            sentiment_types: List of detected sentiment types
            sentiment_score: Overall sentiment score
            
        Returns:
            String describing the interaction type
        """
        if SentimentType.SEXUAL in sentiment_types:
            return "sexual"  # 性的な内容が検出された場合、最優先で「sexual」タイプとする
        elif SentimentType.HOSTILE in sentiment_types:
            return "hostile"
        elif SentimentType.APPRECIATIVE in sentiment_types:
            return "appreciative"
        elif SentimentType.CARING in sentiment_types:
            return "caring"
        elif SentimentType.INTEREST in sentiment_types:  # 新規追加
            return "interest"
        elif SentimentType.DISMISSIVE in sentiment_types:
            return "dismissive"
        elif SentimentType.POSITIVE in sentiment_types:
            return "positive"
        elif SentimentType.NEGATIVE in sentiment_types:
            return "negative"
        elif sentiment_score > 0.3:
            return "positive"
        elif sentiment_score < -0.3:
            return "negative"
        else:
            return "neutral"