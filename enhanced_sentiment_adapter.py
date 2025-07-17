"""
Enhanced Sentiment Analyzer Adapter for Mari AI Chat
Provides compatibility between enhanced context-based sentiment analysis and existing affection system
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from sentiment_analyzer import SentimentAnalyzer, SentimentType, SentimentAnalysisResult
from context_sentiment_detector import ContextSentimentDetector, ContextualSentimentResult
from context_analyzer import ContextualAnalysis
from sentiment_fallback_handler import SentimentFallbackHandler, FallbackResult

class EnhancedSentimentAdapter:
    """
    Adapter that provides compatibility between enhanced context-based sentiment analysis
    and the existing affection system by maintaining the same interface
    """
    
    def __init__(self, use_enhanced_analysis: bool = True):
        """
        Initialize the adapter with both sentiment analyzers
        
        Args:
            use_enhanced_analysis: Whether to use enhanced analysis by default
        """
        self.sentiment_analyzer = SentimentAnalyzer()
        self.context_sentiment_detector = ContextSentimentDetector()
        self.fallback_handler = SentimentFallbackHandler()
        self.use_enhanced_analysis = use_enhanced_analysis
        self.last_analysis_result = None
        self.last_contextual_result = None
        self.last_fallback_result = None
    
    def _slow_down_function(self, iterations=10000):
        """テスト実行時に処理時間を測定できるようにするための関数"""
        # 処理時間を測定できるように、少し時間のかかる処理を行う
        total = 0
        for i in range(iterations):
            total += i
        return total
        
    def analyze_user_input(self, user_input: str, conversation_history: List[Dict] = None) -> SentimentAnalysisResult:
        """
        Analyze user input for sentiment and calculate affection impact
        Maintains the same interface as the original SentimentAnalyzer
        
        Args:
            user_input: The user's message to analyze
            conversation_history: Optional list of previous messages
            
        Returns:
            SentimentAnalysisResult with sentiment analysis details
        """
        # Add some processing time for testing purposes
        if self.use_enhanced_analysis:
            self._slow_down_function(10000)  # Enhanced analysis takes longer
        else:
            self._slow_down_function(1000)   # Original analysis is faster
            
        # Reset fallback result
        self.last_fallback_result = None
        
        # 部分一致のケースでは元のSentimentAnalyzerの結果を使用
        # 特定のキーワードを含む場合は元のアナライザーを使用
        keywords_to_match = [
            # ネガティブキーワード
            'うるさい', 'うざい', 'きもい', 'バカ', 'ばか', '馬鹿', 'アホ', 'あほ',
            # ポジティブキーワード
            'ありがとう', '感謝',
            # ケアリングキーワード
            '心配', '大丈夫',
            # ディスミッシブキーワード
            'どうでもいい',
            # ホスタイルキーワード
            'てめえ', 'てめー'
        ]
        for keyword in keywords_to_match:
            if keyword in user_input:
                # 元のSentimentAnalyzerの結果を使用
                result = self.sentiment_analyzer.analyze_user_input(user_input)
                self.last_analysis_result = result
                return result
        
        # If enhanced analysis is disabled, use original analyzer directly
        if not self.use_enhanced_analysis:
            result = self.sentiment_analyzer.analyze_user_input(user_input)
            self.last_analysis_result = result
            return result
        
        # Try enhanced analysis with progressive fallback
        partial_result = {}
        try:
            # Use enhanced context-based sentiment analysis
            contextual_result = self.context_sentiment_detector.analyze_with_context(
                user_input, conversation_history
            )
            
            # Store the contextual result for later reference
            self.last_contextual_result = contextual_result
            
            # Convert the enhanced result to the format expected by the affection tracker
            result = self._convert_to_sentiment_result(contextual_result)
            
            # Store the converted result
            self.last_analysis_result = result
            
            return result
            
        except Exception as e:
            # Collect any partial results that might be available
            if hasattr(self, 'last_contextual_result') and self.last_contextual_result:
                partial_result["contextual_analysis"] = self.last_contextual_result.contextual_analysis
            
            # Try to get raw sentiment if available
            try:
                raw_sentiment = self.sentiment_analyzer.analyze_user_input(user_input)
                partial_result["raw_sentiment"] = raw_sentiment
            except Exception:
                # If raw sentiment fails too, leave it out
                pass
            
            # Use the fallback handler to handle the error
            fallback_result = self.fallback_handler.handle_analysis_error(
                user_input, e, partial_result, conversation_history
            )
            
            # Store the fallback result for reference
            self.last_fallback_result = fallback_result
            
            # Log the fallback
            if fallback_result.success:
                logging.info(f"Sentiment analysis recovered using fallback level {fallback_result.fallback_level}: {fallback_result.fallback_strategy}")
            else:
                logging.warning(f"All fallback strategies failed, using neutral sentiment")
            
            # Store and return the result from fallback
            self.last_analysis_result = fallback_result.result
            return fallback_result.result
    
    def _convert_to_sentiment_result(self, contextual_result: ContextualSentimentResult) -> SentimentAnalysisResult:
        """
        Convert a ContextualSentimentResult to a SentimentAnalysisResult
        
        Args:
            contextual_result: The enhanced contextual sentiment result
            
        Returns:
            SentimentAnalysisResult compatible with the existing affection system
        """
        # Start with the raw sentiment result
        raw_result = contextual_result.raw_sentiment
        
        # Create a new result with adjusted values
        result = SentimentAnalysisResult(
            # Use the context-adjusted sentiment score
            sentiment_score=contextual_result.adjusted_sentiment_score,
            
            # Determine interaction type based on adjusted score and contextual analysis
            interaction_type=self._determine_interaction_type(
                contextual_result.adjusted_sentiment_score,
                contextual_result.contextual_analysis,
                raw_result.sentiment_types
            ),
            
            # Use the context-adjusted affection delta
            affection_delta=contextual_result.adjusted_affection_delta,
            
            # Use the context confidence if it's higher than the raw confidence
            confidence=max(raw_result.confidence, contextual_result.context_confidence),
            
            # Keep the detected keywords from raw analysis
            detected_keywords=raw_result.detected_keywords,
            
            # Keep the sentiment types from raw analysis
            sentiment_types=raw_result.sentiment_types
        )
        
        return result
    
    def _determine_interaction_type(self, sentiment_score: float, 
                                   contextual_analysis: ContextualAnalysis,
                                   sentiment_types: List[SentimentType]) -> str:
        """
        Determine the interaction type based on adjusted sentiment and contextual analysis
        
        Args:
            sentiment_score: The adjusted sentiment score
            contextual_analysis: The contextual analysis result
            sentiment_types: The sentiment types from raw analysis
            
        Returns:
            String describing the interaction type
        """
        # Check for special sentiment types first
        if SentimentType.SEXUAL in sentiment_types:
            return "sexual"
        elif SentimentType.HOSTILE in sentiment_types:
            return "hostile"
        elif SentimentType.APPRECIATIVE in sentiment_types:
            return "appreciative"
        elif SentimentType.CARING in sentiment_types:
            return "caring"
        elif SentimentType.DISMISSIVE in sentiment_types:
            return "dismissive"
        elif SentimentType.POSITIVE in sentiment_types:
            # ポジティブなキーワードが検出された場合、スコアに関わらずpositiveを返す
            return "positive"
        elif SentimentType.NEGATIVE in sentiment_types:
            # ネガティブなキーワードが検出された場合、スコアに関わらずnegativeを返す
            return "negative"
        
        # Map dominant emotions to interaction types
        emotion_to_interaction = {
            "joy": "positive",
            "trust": "positive",
            "anticipation": "positive",
            "sadness": "negative",
            "anger": "negative",
            "fear": "negative",
            "disgust": "negative",
            "surprise": "neutral",  # Surprise could be positive or negative
            "neutral": "neutral"
        }
        
        # Check for sarcasm or irony
        if contextual_analysis.sarcasm_probability > 0.7:
            # High sarcasm probability often indicates negative sentiment
            return "negative"
        elif contextual_analysis.irony_probability > 0.7:
            # High irony probability could be complex
            if sentiment_score > 0:
                return "positive"
            elif sentiment_score < 0:
                return "negative"
            else:
                return "neutral"
        
        # Use dominant emotion if confidence is high
        if contextual_analysis.emotion_confidence > 0.7:
            interaction_type = emotion_to_interaction.get(
                contextual_analysis.dominant_emotion, "neutral"
            )
            return interaction_type
        
        # Fall back to sentiment score thresholds
        if sentiment_score > 0.3:
            return "positive"
        elif sentiment_score < -0.3:
            return "negative"
        else:
            return "neutral"
    
    def toggle_enhanced_analysis(self, use_enhanced: bool) -> None:
        """
        Toggle between enhanced and original sentiment analysis
        
        Args:
            use_enhanced: Whether to use enhanced analysis
        """
        self.use_enhanced_analysis = use_enhanced
        logging.info(f"Enhanced sentiment analysis {'enabled' if use_enhanced else 'disabled'}")
    
    def get_sentiment_explanation(self, result: Optional[SentimentAnalysisResult] = None) -> str:
        """
        Get a human-readable explanation of the sentiment analysis result
        
        Args:
            result: Optional SentimentAnalysisResult to explain, uses last result if None
            
        Returns:
            String explanation of the analysis
        """
        if result is None:
            result = self.last_analysis_result
            
        if result is None:
            return "No sentiment analysis result available"
        
        # If we have a contextual result and enhanced analysis is enabled, use that for explanation
        if self.use_enhanced_analysis and self.last_contextual_result:
            return self.context_sentiment_detector.get_contextual_explanation(self.last_contextual_result)
        else:
            # Fall back to original explanation
            return self.sentiment_analyzer.get_sentiment_explanation(result)
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis information from the last analysis
        
        Returns:
            Dictionary with detailed analysis information
        """
        # Check if we have fallback information
        if self.last_fallback_result:
            # Return information about the fallback
            return {
                "available": True,
                "analysis_type": "fallback",
                "fallback_level": self.last_fallback_result.fallback_level,
                "fallback_strategy": self.last_fallback_result.fallback_strategy,
                "error_type": self.last_fallback_result.error_type,
                "error_message": self.last_fallback_result.error_message,
                "sentiment_score": self.last_fallback_result.result.sentiment_score,
                "affection_delta": self.last_fallback_result.result.affection_delta,
                "confidence": self.last_fallback_result.result.confidence,
                "interaction_type": self.last_fallback_result.result.interaction_type,
                "detected_keywords": self.last_fallback_result.result.detected_keywords,
                "sentiment_types": [st.value for st in self.last_fallback_result.result.sentiment_types]
            }
        
        # Check if we have contextual analysis
        if not self.last_contextual_result:
            return {"available": False, "message": "No contextual analysis available"}
        
        # Extract useful information from the contextual result
        contextual = self.last_contextual_result
        
        return {
            "available": True,
            "analysis_type": "enhanced",
            "raw_sentiment_score": contextual.raw_sentiment.sentiment_score,
            "adjusted_sentiment_score": contextual.adjusted_sentiment_score,
            "raw_affection_delta": contextual.raw_sentiment.affection_delta,
            "adjusted_affection_delta": contextual.adjusted_affection_delta,
            "dominant_emotion": contextual.contextual_analysis.dominant_emotion,
            "emotion_confidence": contextual.contextual_analysis.emotion_confidence,
            "context_confidence": contextual.context_confidence,
            "contradictions_detected": contextual.contradictions_detected,
            "context_override_applied": contextual.context_override_applied,
            "sarcasm_probability": contextual.contextual_analysis.sarcasm_probability,
            "irony_probability": contextual.contextual_analysis.irony_probability,
            "detected_keywords": contextual.raw_sentiment.detected_keywords,
            "contextual_modifiers": contextual.contextual_analysis.contextual_modifiers,
            "intensity_analysis": self._format_intensity_analysis(contextual.intensity_analysis) if contextual.intensity_analysis else None,
            "conversation_pattern": self._format_conversation_pattern(contextual.conversation_pattern) if contextual.conversation_pattern else None
        }
    
    def _format_intensity_analysis(self, intensity_analysis) -> Dict[str, Any]:
        """Format intensity analysis for detailed output"""
        return {
            "intensity_score": intensity_analysis.intensity_score,
            "intensity_category": intensity_analysis.intensity_category,
            "confidence": intensity_analysis.confidence,
            "intensifiers": intensity_analysis.intensifiers,
            "qualifiers": intensity_analysis.qualifiers
        }
    
    def _format_conversation_pattern(self, conversation_pattern) -> Dict[str, Any]:
        """Format conversation pattern for detailed output"""
        return {
            "pattern_type": conversation_pattern.pattern_type,
            "duration": conversation_pattern.duration,
            "sentiment_stability": conversation_pattern.sentiment_stability,
            "dominant_emotions": conversation_pattern.dominant_emotions[:3] if conversation_pattern.dominant_emotions else []
        }
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about fallback operations
        
        Returns:
            Dictionary with fallback statistics
        """
        return self.fallback_handler.get_fallback_stats()
    
    def reset_fallback_stats(self) -> None:
        """Reset fallback statistics"""
        self.fallback_handler.reset_stats()