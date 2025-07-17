"""
Confidence-Based Impact Adjustment Module for Context-Based Sentiment Analysis
Adjusts affection impact based on confidence in sentiment analysis
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from sentiment_analyzer import SentimentAnalysisResult
from context_analyzer import ContextualAnalysis
from confidence_calculator import ConfidenceCalculator, ConfidenceAssessmentResult

@dataclass
class ImpactAdjustmentResult:
    """Result of confidence-based impact adjustment"""
    original_affection_delta: int
    adjusted_affection_delta: int
    confidence_score: float
    adjustment_factor: float
    fallback_applied: bool
    adjustment_reason: str
    confidence_assessment: ConfidenceAssessmentResult

class ConfidenceBasedImpactAdjuster:
    """Adjusts affection impact based on confidence in sentiment analysis"""
    
    def __init__(self):
        """Initialize the impact adjuster"""
        self.confidence_calculator = ConfidenceCalculator()
        self.fallback_thresholds = self._load_fallback_thresholds()
    
    def _load_fallback_thresholds(self) -> Dict[str, float]:
        """
        Load thresholds for fallback mechanisms
        
        Returns:
            Dictionary mapping fallback types to their confidence thresholds
        """
        return {
            "severe_reduction": 0.3,  # Below this, severely reduce impact
            "moderate_reduction": 0.5,  # Below this, moderately reduce impact
            "slight_reduction": 0.7,  # Below this, slightly reduce impact
            "no_reduction": 0.9  # Above this, no reduction
        }
    
    def adjust_impact(self, 
                    sentiment_result: SentimentAnalysisResult,
                    contextual_analysis: ContextualAnalysis,
                    text: str,
                    contradictions: List[str] = None,
                    conversation_pattern = None) -> ImpactAdjustmentResult:
        """
        Adjust affection impact based on confidence assessment
        
        Args:
            sentiment_result: Result from sentiment analysis
            contextual_analysis: Result from contextual analysis
            text: Original text for analysis
            contradictions: List of detected contradictions
            conversation_pattern: Optional conversation pattern analysis
            
        Returns:
            ImpactAdjustmentResult with adjusted impact details
        """
        # Calculate confidence
        confidence_assessment = self.confidence_calculator.calculate_confidence(
            sentiment_result,
            contextual_analysis,
            conversation_pattern,
            text,
            contradictions
        )
        
        # Get original affection delta
        original_delta = sentiment_result.affection_delta
        
        # Determine adjustment factor based on confidence
        adjustment_factor, reason = self._determine_adjustment_factor(confidence_assessment)
        
        # Apply adjustment
        adjusted_delta = int(original_delta * adjustment_factor)
        
        # Check if fallback is needed
        fallback_applied = False
        if confidence_assessment.overall_confidence < self.fallback_thresholds["severe_reduction"]:
            # For very low confidence, apply more conservative fallback
            adjusted_delta, fallback_applied = self._apply_fallback_mechanism(
                adjusted_delta, 
                confidence_assessment.overall_confidence
            )
        
        # Ensure bounds
        adjusted_delta = max(-10, min(10, adjusted_delta))
        
        # Create and return the result
        return ImpactAdjustmentResult(
            original_affection_delta=original_delta,
            adjusted_affection_delta=adjusted_delta,
            confidence_score=confidence_assessment.overall_confidence,
            adjustment_factor=adjustment_factor,
            fallback_applied=fallback_applied,
            adjustment_reason=reason,
            confidence_assessment=confidence_assessment
        )
    
    def _determine_adjustment_factor(self, confidence_assessment: ConfidenceAssessmentResult) -> Tuple[float, str]:
        """
        Determine the adjustment factor based on confidence assessment
        
        Args:
            confidence_assessment: Result from confidence calculation
            
        Returns:
            Tuple of (adjustment_factor, reason)
        """
        confidence = confidence_assessment.overall_confidence
        
        # Start with the recommended weight from confidence assessment
        adjustment_factor = confidence_assessment.recommended_weight
        
        # Apply additional adjustments based on specific factors
        reason = "Confidence-based adjustment"
        
        # Check for contradictions first (highest priority)
        if "contradiction_penalty" in confidence_assessment.confidence_breakdown and \
           confidence_assessment.confidence_breakdown["contradiction_penalty"] < -0.1:
            adjustment_factor *= 0.8
            reason = "Contradictions detected"
        
        # Check for sarcasm/irony (second priority)
        elif "sarcasm_irony_penalty" in confidence_assessment.confidence_breakdown and \
             confidence_assessment.confidence_breakdown["sarcasm_irony_penalty"] < -0.1:
            adjustment_factor *= 0.7
            reason = "Sarcasm or irony detected"
        
        # Check for high ambiguity (third priority)
        elif confidence_assessment.ambiguity_score > 0.5:
            adjustment_factor *= 0.7
            reason = "High ambiguity detected"
        
        # Check confidence levels (lowest priority)
        elif confidence < 0.3:
            # Limit maximum impact to +/-3 for very low confidence
            adjustment_factor = min(adjustment_factor, 0.3)
            reason = "Very low confidence"
        
        elif confidence < 0.6:
            # Limit maximum impact to +/-6 for moderate confidence
            adjustment_factor = min(adjustment_factor, 0.6)
            reason = "Moderate confidence"
        
        # Ensure bounds
        adjustment_factor = max(0.1, min(1.0, adjustment_factor))
        
        return adjustment_factor, reason
    
    def _apply_fallback_mechanism(self, adjusted_delta: int, confidence: float) -> Tuple[int, bool]:
        """
        Apply fallback mechanism for low confidence scenarios
        
        Args:
            adjusted_delta: The already adjusted affection delta
            confidence: The overall confidence score
            
        Returns:
            Tuple of (fallback_delta, fallback_applied)
        """
        # For very low confidence, limit impact to +/-1
        if confidence < 0.2:
            fallback_delta = max(-1, min(1, adjusted_delta))
            return fallback_delta, True
        
        # For low confidence, limit impact to +/-2
        elif confidence < 0.3:
            fallback_delta = max(-2, min(2, adjusted_delta))
            return fallback_delta, True
        
        # No fallback needed
        return adjusted_delta, False
    
    def get_adjustment_explanation(self, result: ImpactAdjustmentResult) -> str:
        """
        Get a human-readable explanation of the impact adjustment
        
        Args:
            result: ImpactAdjustmentResult to explain
            
        Returns:
            String explanation of the impact adjustment
        """
        explanation_parts = []
        
        # Basic adjustment information
        explanation_parts.append(f"Original affection impact: {result.original_affection_delta:+d}")
        explanation_parts.append(f"Adjusted affection impact: {result.adjusted_affection_delta:+d}")
        explanation_parts.append(f"Confidence score: {result.confidence_score:.2f}")
        explanation_parts.append(f"Adjustment factor: {result.adjustment_factor:.2f}")
        
        # Adjustment reason
        explanation_parts.append(f"Adjustment reason: {result.adjustment_reason}")
        
        # Fallback information
        if result.fallback_applied:
            explanation_parts.append("Fallback mechanism applied due to low confidence")
        
        # Add confidence explanation
        confidence_explanation = self.confidence_calculator.get_confidence_explanation(result.confidence_assessment)
        explanation_parts.append("\nConfidence assessment details:")
        explanation_parts.append(str(confidence_explanation))
        
        return "\n".join(explanation_parts)
    
    def apply_to_sentiment_result(self, 
                                sentiment_result: SentimentAnalysisResult,
                                contextual_analysis: ContextualAnalysis,
                                text: str,
                                contradictions: List[str] = None,
                                conversation_pattern = None) -> SentimentAnalysisResult:
        """
        Apply confidence-based adjustment to a sentiment result and return an updated result
        
        Args:
            sentiment_result: Original sentiment analysis result
            contextual_analysis: Result from contextual analysis
            text: Original text for analysis
            contradictions: List of detected contradictions
            conversation_pattern: Optional conversation pattern analysis
            
        Returns:
            Updated SentimentAnalysisResult with adjusted affection delta
        """
        # Get adjustment
        adjustment_result = self.adjust_impact(
            sentiment_result,
            contextual_analysis,
            text,
            contradictions,
            conversation_pattern
        )
        
        # Create a new result with adjusted affection delta
        adjusted_result = SentimentAnalysisResult(
            sentiment_score=sentiment_result.sentiment_score,
            interaction_type=sentiment_result.interaction_type,
            affection_delta=adjustment_result.adjusted_affection_delta,
            confidence=adjustment_result.confidence_score,  # Update confidence with our assessment
            detected_keywords=sentiment_result.detected_keywords,
            sentiment_types=sentiment_result.sentiment_types
        )
        
        # Log the adjustment
        logging.info(f"Adjusted affection impact from {sentiment_result.affection_delta} to "
                    f"{adjusted_result.affection_delta} (confidence: {adjustment_result.confidence_score:.2f}, "
                    f"reason: {adjustment_result.adjustment_reason})")
        
        return adjusted_result
    
    def get_fallback_recommendation(self, confidence_score: float) -> str:
        """
        Get a recommendation for handling low-confidence scenarios
        
        Args:
            confidence_score: The confidence score from analysis
            
        Returns:
            String with recommendation for handling the scenario
        """
        if confidence_score < 0.2:
            return ("Very low confidence detected. Consider asking for clarification "
                   "or responding neutrally to avoid misinterpretation.")
        elif confidence_score < 0.4:
            return ("Low confidence detected. Consider acknowledging the ambiguity "
                   "in your response or seeking additional context.")
        elif confidence_score < 0.6:
            return ("Moderate confidence detected. The sentiment analysis may not fully "
                   "capture the nuance of the message.")
        else:
            return "Confidence is sufficient for normal response handling."