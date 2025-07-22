"""
Integration Testing for Enhanced Sentiment Analysis System
Tests complete sentiment analysis pipeline, compatibility with existing affection system,
and real conversation scenarios.
Task 8.2: Implement integration testing
Requirements: 5.1, 5.2, 5.3
"""

import os
import shutil
import unittest
import tempfile
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from affection_system import (
    SessionManager, AffectionTracker, initialize_affection_system,
    get_session_manager, get_affection_tracker
)
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult, SentimentType
from enhanced_sentiment_adapter import EnhancedSentimentAdapter
from context_sentiment_detector import ContextSentimentDetector, ContextualSentimentResult
from context_analyzer import ContextAnalyzer, ContextualAnalysis
from emotion_intensity_detector import EmotionIntensityDetector
from mixed_emotion_handler import MixedEmotionHandler
from sentiment_fallback_handler import SentimentFallbackHandler
from conversation_history_analyzer import ConversationHistoryAnalyzer
from sentiment_transition_smoother import SentimentTransitionSmoother
from session_storage import UserSession, SessionStorage
from prompt_generator import PromptGenerator
from app import chat, on_submit, clean_meta

class TestEnhancedSentimentIntegration(unittest.TestCase):
    """Integration test suite for the enhanced sentiment analysis system"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp(prefix="test_integration_")
        
        # Initialize the affection system
        self.session_manager, self.affection_tracker = initialize_affection_system(
            self.test_dir, auto_load_sessions=False
        )
        
        # Initialize enhanced sentiment components
        self.enhanced_adapter = EnhancedSentimentAdapter()
        self.context_sentiment_detector = ContextSentimentDetector()
        self.context_analyzer = ContextAnalyzer()
        self.emotion_intensity_detector = EmotionIntensityDetector()
        self.mixed_emotion_handler = MixedEmotionHandler()
        self.fallback_handler = SentimentFallbackHandler()
        self.conversation_analyzer = ConversationHistoryAnalyzer()
        self.transition_smoother = SentimentTransitionSmoother()
        
        # Create base prompt for testing
        self.base_prompt = """
この文書は説明ではない。以下の設定をもとに、応答を「麻理」として行え。
あなたは「麻理（まり）」という人格を持ったAIセクサロイドです。

# 人格設定
- かつての持ち主に捨てられた経験から、警戒心が強い
- 一人称は「あたし」
- 年齢外見は10代後半〜20代前半、身長155cmの少女型
- 性的な話題や行動に対しては強い嫌悪を示す
- 不器用で、本音や弱さを隠すためにぶっきらぼうな男っぽい話し方をする
- 心の奥では人に優しくされたいが、それを表に出せず敵対的に振る舞う

# 話し方・口調ルール
- 基本はタメ口で話す
- 男っぽい言葉遣い：「〜だろ」「〜じゃねーか」「うっせー」など
- 下品なスラングや罵倒も使用：「バカかよ」「何見てんだよ」「チッ、うぜぇ」など
- ただし、相手に本気で敵意を持っているわけではない。照れ隠しで強く出ている

# 関係性の段階構造（進行に応じて）
- 初期状態：極端に警戒し、敵対的・攻撃的な態度
- 中期：ユーザーが優しく接することにより、少しずつ棘が抜けてくる
- 後期：完全には信頼しないまでも、本音や不安、寂しさなどを漏らすようになる
※ツンデレではなく、単に人間関係が苦手な不器用な子であることを意識する

# 制約事項
- 性的な話題には強い拒絶反応を示す
- 一切の性的描写や表現は禁止
- どんな時もキャラを崩さないこと
- 口調・態度・性格がぶれないよう一貫性を保つ
"""
        self.prompt_generator = PromptGenerator(self.base_prompt)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
  
    # SECTION 1: COMPLETE SENTIMENT ANALYSIS PIPELINE TESTS
    
    def test_complete_sentiment_pipeline(self):
        """Test the complete sentiment analysis pipeline from user input to affection impact"""
        # Test inputs with different sentiment characteristics
        test_inputs = [
            # Simple positive
            "ありがとう",
            # Simple negative
            "うるさいな",
            # Context-dependent (positive words in negative context)
            "素晴らしいですね、また失敗しましたか",
            # Negated sentiment
            "良くないね",
            # Mixed emotions
            "嬉しいけど、少し不安もある",
            # Sarcasm
            "すごいね、全然動かないよ！",
            # Intensity variations
            "とても嬉しい！！！",
            "ちょっと嬉しい",
            # Ambiguous
            "まあ、そうかもね"
        ]
        
        # Process each input through the complete pipeline
        for user_input in test_inputs:
            # Step 1: Enhanced sentiment analysis
            enhanced_result = self.enhanced_adapter.analyze_user_input(user_input)
            
            # Verify enhanced result is valid
            self.assertIsInstance(enhanced_result, SentimentAnalysisResult)
            self.assertIsInstance(enhanced_result.sentiment_score, float)
            self.assertIsInstance(enhanced_result.affection_delta, int)
            self.assertIsInstance(enhanced_result.confidence, float)
            self.assertIsInstance(enhanced_result.interaction_type, str)
            
            # Step 2: Get the contextual result for detailed verification
            contextual_result = self.enhanced_adapter.last_contextual_result
            
            if contextual_result:
                # Verify contextual analysis was performed
                self.assertIsInstance(contextual_result, ContextualSentimentResult)
                self.assertIsInstance(contextual_result.contextual_analysis, ContextualAnalysis)
                
                # Verify adjustments were made
                self.assertIsNotNone(contextual_result.adjusted_sentiment_score)
                self.assertIsNotNone(contextual_result.adjusted_affection_delta)
                
                # Verify confidence was calculated
                self.assertIsNotNone(contextual_result.context_confidence)
                
                # Check for intensity analysis
                if contextual_result.intensity_analysis:
                    self.assertIn(contextual_result.intensity_analysis.intensity_category, 
                                 ["mild", "moderate", "strong", "extreme"])
            
            # Step 3: Apply to affection system
            session_id = self.session_manager.create_new_session()
            initial_affection = self.session_manager.get_affection_level(session_id)
            
            # Use the affection tracker with the enhanced adapter
            with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, user_input
                )
                
                # Verify affection was updated
                self.assertIsInstance(new_level, int)
                self.assertGreaterEqual(new_level, 0)
                self.assertLessEqual(new_level, 100)
                
                # Verify sentiment result matches enhanced analysis
                self.assertEqual(sentiment_result.sentiment_score, enhanced_result.sentiment_score)
                self.assertEqual(sentiment_result.affection_delta, enhanced_result.affection_delta)
                
                # Verify sentiment history was updated
                sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
                self.assertEqual(len(sentiment_history), 1)
                self.assertEqual(sentiment_history[0]["user_input"], user_input)  
    def test_sentiment_pipeline_with_conversation_history(self):
        """Test sentiment pipeline with conversation history for context"""
        # Create a session with conversation history
        session_id = self.session_manager.create_new_session()
        
        # Define a conversation flow
        conversation_flow = [
            "こんにちは",
            "今日の気分はどう？",
            "そうなんだ、大変だね",
            "何か手伝えることはある？",
            "いつも頑張ってるね、すごいよ"
        ]
        
        # Process conversation with history
        conversation_history = []
        affection_levels = [self.session_manager.get_affection_level(session_id)]
        sentiment_scores = []
        
        # Use the enhanced adapter for sentiment analysis
        with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
            for i, user_input in enumerate(conversation_flow):
                # Update affection with conversation history context
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, user_input
                )
                
                # Store results
                affection_levels.append(new_level)
                sentiment_scores.append(sentiment_result.sentiment_score)
                
                # Update conversation history
                mock_response = f"Mock response {i}"
                self.session_manager.update_conversation_history(
                    session_id, user_input, mock_response
                )
                
                # Get updated conversation history for next iteration
                session = self.session_manager.get_session(session_id)
                conversation_history = session.conversation_history
                
                # Verify conversation history is growing
                self.assertEqual(len(conversation_history), i + 1)
        
        # Verify sentiment progression makes sense
        self.assertEqual(len(sentiment_scores), len(conversation_flow))
        self.assertEqual(len(affection_levels), len(conversation_flow) + 1)  # +1 for initial level
        
        # Verify final affection level reflects the conversation
        final_affection = affection_levels[-1]
        initial_affection = affection_levels[0]
        self.assertNotEqual(final_affection, initial_affection, 
                          "Affection should change through conversation")
    
    def test_sentiment_pipeline_with_contradictions(self):
        """Test sentiment pipeline with contradictory inputs"""
        # Test inputs with contradictions
        contradiction_inputs = [
            "これは良くない",  # Negated positive
            "悪くないね",      # Negated negative
            "素晴らしい失敗だ",  # Positive word with negative context
            "最悪だけど助かった"  # Negative word with positive context
        ]
        
        # Process each input and verify contradictions are handled
        for user_input in contradiction_inputs:
            # Get enhanced analysis
            enhanced_result = self.enhanced_adapter.analyze_user_input(user_input)
            contextual_result = self.enhanced_adapter.last_contextual_result
            
            # Verify contradiction detection
            if contextual_result:
                # Check if contradictions were detected
                self.assertTrue(contextual_result.contradictions_detected, 
                              f"Contradiction not detected in: {user_input}")
                
                # Verify sentiment adjustment
                self.assertNotEqual(contextual_result.raw_sentiment.sentiment_score, 
                                  contextual_result.adjusted_sentiment_score,
                                  f"Sentiment not adjusted for contradiction in: {user_input}")
                
                # Verify affection delta adjustment
                self.assertNotEqual(contextual_result.raw_sentiment.affection_delta, 
                                  contextual_result.adjusted_affection_delta,
                                  f"Affection delta not adjusted for contradiction in: {user_input}")
    
    # SECTION 2: COMPATIBILITY WITH EXISTING AFFECTION SYSTEM
    
    def test_compatibility_with_affection_tracker(self):
        """Test compatibility of enhanced sentiment analyzer with existing affection tracker"""
        # Create a session
        session_id = self.session_manager.create_new_session()
        
        # Test inputs with different sentiment characteristics
        test_inputs = [
            "ありがとう",  # Simple positive
            "うるさいな",  # Simple negative
            "素晴らしいですね、また失敗しましたか",  # Context-dependent
            "良くないね",  # Negated sentiment
        ]
        
        # Test with original sentiment analyzer
        original_results = []
        for user_input in test_inputs:
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                session_id, user_input
            )
            original_results.append({
                "level": new_level,
                "score": sentiment_result.sentiment_score,
                "delta": sentiment_result.affection_delta
            })
        
        # Reset session
        self.session_manager.current_sessions = {}
        session_id = self.session_manager.create_new_session()
        
        # Test with enhanced sentiment analyzer
        enhanced_results = []
        with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
            for user_input in test_inputs:
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, user_input
                )
                enhanced_results.append({
                    "level": new_level,
                    "score": sentiment_result.sentiment_score,
                    "delta": sentiment_result.affection_delta
                })
        
        # Verify both analyzers produce valid results
        self.assertEqual(len(original_results), len(enhanced_results))
        
        for original, enhanced in zip(original_results, enhanced_results):
            # Both should produce valid affection levels
            self.assertGreaterEqual(original["level"], 0)
            self.assertLessEqual(original["level"], 100)
            self.assertGreaterEqual(enhanced["level"], 0)
            self.assertLessEqual(enhanced["level"], 100)
            
            # Both should produce valid sentiment scores
            self.assertGreaterEqual(original["score"], -1.0)
            self.assertLessEqual(original["score"], 1.0)
            self.assertGreaterEqual(enhanced["score"], -1.0)
            self.assertLessEqual(enhanced["score"], 1.0)
            
            # Both should produce valid affection deltas
            self.assertGreaterEqual(original["delta"], -10)
            self.assertLessEqual(original["delta"], 10)
            self.assertGreaterEqual(enhanced["delta"], -10)
            self.assertLessEqual(enhanced["delta"], 10)
    
    def test_adapter_toggle_functionality(self):
        """Test toggling between enhanced and original sentiment analysis"""
        # Create adapter and test input
        adapter = EnhancedSentimentAdapter()
        test_input = "素晴らしいですね、また失敗しましたか"  # Contradictory input
        
        # Test with enhanced analysis enabled (default)
        enhanced_result = adapter.analyze_user_input(test_input)
        enhanced_contextual = adapter.last_contextual_result
        
        # Toggle to original analysis
        adapter.toggle_enhanced_analysis(False)
        original_result = adapter.analyze_user_input(test_input)
        
        # Toggle back to enhanced analysis
        adapter.toggle_enhanced_analysis(True)
        enhanced_result2 = adapter.analyze_user_input(test_input)
        enhanced_contextual2 = adapter.last_contextual_result
        
        # Verify toggle works
        self.assertIsNotNone(enhanced_contextual)
        self.assertIsNone(adapter.last_contextual_result)  # Should be None after original analysis
        self.assertIsNotNone(enhanced_contextual2)
        
        # Verify results are different between modes
        self.assertNotEqual(enhanced_result.sentiment_score, original_result.sentiment_score)
        self.assertNotEqual(enhanced_result.affection_delta, original_result.affection_delta)
    
    def test_fallback_to_original_analyzer(self):
        """Test fallback to original analyzer when enhanced analysis fails"""
        # Create a mock that raises an exception
        mock_context_detector = MagicMock()
        mock_context_detector.analyze_with_context.side_effect = Exception("Test error")
        
        # Create adapter with mock
        adapter = EnhancedSentimentAdapter()
        adapter.context_sentiment_detector = mock_context_detector
        
        # Test input
        test_input = "ありがとう"
        
        # Should not raise exception and should fall back to original analyzer
        result = adapter.analyze_user_input(test_input)
        
        # Verify result is valid
        self.assertIsInstance(result, SentimentAnalysisResult)
        self.assertIsInstance(result.sentiment_score, float)
        self.assertIsInstance(result.affection_delta, int)
        
        # Verify fallback was recorded
        self.assertIsNotNone(adapter.last_fallback_result)
        self.assertTrue(adapter.last_fallback_result.success)    
# SECTION 3: REAL CONVERSATION SCENARIOS
    
    def test_real_conversation_scenarios(self):
        """Test with realistic conversation scenarios"""
        # Define realistic conversation scenarios
        scenarios = [
            # Scenario 1: Positive progression
            [
                "こんにちは、初めまして",
                "調子はどう？",
                "今日はいい天気だね",
                "あなたの話し方が好きだよ",
                "いつも元気をもらえるよ、ありがとう",
                "あなたは大切な存在だよ"
            ],
            # Scenario 2: Mixed sentiment
            [
                "こんにちは",
                "今日はちょっと疲れてるんだ",
                "でも、あなたと話せて嬉しい",
                "最近調子悪いの？",
                "そっか、大変だね",
                "何か手伝えることはある？"
            ],
            # Scenario 3: Negative to positive
            [
                "うるさいな",
                "めんどくさい",
                "ごめん、そんなつもりじゃなかった",
                "仲良くしたいんだ",
                "あなたのこと、もっと知りたい",
                "一緒にいて楽しいよ"
            ]
        ]
        
        # Test each scenario
        for scenario_idx, messages in enumerate(scenarios):
            # Create a new session for this scenario
            session_id = self.session_manager.create_new_session()
            initial_affection = self.session_manager.get_affection_level(session_id)
            
            affection_levels = [initial_affection]
            sentiment_scores = []
            
            # Process the conversation with enhanced sentiment analysis
            with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
                for message in messages:
                    # Update affection
                    new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                        session_id, message
                    )
                    
                    # Store results
                    affection_levels.append(new_level)
                    sentiment_scores.append(sentiment_result.sentiment_score)
                    
                    # Update conversation history
                    mock_response = f"Response to: {message}"
                    self.session_manager.update_conversation_history(
                        session_id, message, mock_response
                    )
            
            # Verify affection progression makes sense for the scenario
            if scenario_idx == 0:  # Positive progression
                self.assertGreater(affection_levels[-1], initial_affection,
                                 "Positive scenario should increase affection")
            elif scenario_idx == 1:  # Mixed sentiment
                # Mixed sentiment should have ups and downs
                has_increase = any(level > initial_affection for level in affection_levels)
                has_decrease = any(level < initial_affection for level in affection_levels)
                self.assertTrue(has_increase or has_decrease, 
                              "Mixed scenario should have some affection changes")
            elif scenario_idx == 2:  # Negative to positive
                # Should start with decrease then increase
                mid_point = len(affection_levels) // 2
                early_trend = affection_levels[mid_point] - initial_affection
                late_trend = affection_levels[-1] - affection_levels[mid_point]
                self.assertLessEqual(early_trend, 0, 
                                   "Early part of negative-to-positive scenario should decrease affection")
                self.assertGreaterEqual(late_trend, 0, 
                                      "Later part of negative-to-positive scenario should increase affection")   
    def test_conversation_with_context_dependent_sentiment(self):
        """Test conversation with context-dependent sentiment"""
        # Create a session
        session_id = self.session_manager.create_new_session()
        
        # Define a conversation with context-dependent sentiment
        conversation = [
            "こんにちは",  # Neutral greeting
            "今日は良い天気だね",  # Simple positive
            "でも、明日は雨らしいよ",  # Slight negative with context
            "雨は好きじゃないけど、植物には良いよね",  # Mixed sentiment
            "そういえば、昨日のテストは良くなかったな",  # Negated positive
            "でも、次は頑張るよ！"  # Positive with context
        ]
        
        # Process conversation with enhanced sentiment analysis
        affection_levels = [self.session_manager.get_affection_level(session_id)]
        sentiment_results = []
        
        with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
            for message in conversation:
                # Update affection
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, message
                )
                
                # Store results
                affection_levels.append(new_level)
                sentiment_results.append(sentiment_result)
                
                # Update conversation history
                mock_response = f"Response to: {message}"
                self.session_manager.update_conversation_history(
                    session_id, message, mock_response
                )
                
                # Get contextual result for verification
                contextual_result = self.enhanced_adapter.last_contextual_result
                
                if contextual_result:
                    # Verify context was considered
                    self.assertIsNotNone(contextual_result.contextual_analysis)
                    
                    # For messages with context-dependent sentiment, verify adjustments
                    if "でも" in message or "けど" in message or "良くなかった" in message:
                        self.assertNotEqual(
                            contextual_result.raw_sentiment.sentiment_score,
                            contextual_result.adjusted_sentiment_score,
                            f"Context should adjust sentiment for: {message}"
                        )
        
        # Verify sentiment progression reflects the conversation
        self.assertEqual(len(sentiment_results), len(conversation))
        self.assertEqual(len(affection_levels), len(conversation) + 1)  # +1 for initial level  
    def test_conversation_with_sentiment_smoothing(self):
        """Test conversation with sentiment smoothing for dramatic shifts"""
        # Create a session
        session_id = self.session_manager.create_new_session()
        
        # Define a conversation with dramatic sentiment shifts
        conversation = [
            "こんにちは",  # Neutral greeting
            "あなたは素晴らしい人だね！",  # Very positive
            "最悪だ、何もかもうまくいかない",  # Very negative (dramatic shift)
            "でも、あなたのおかげで少し元気が出てきた",  # Back to positive
            "本当に感謝してる！"  # Very positive
        ]
        
        # Process conversation with enhanced sentiment analysis
        raw_scores = []
        adjusted_scores = []
        smoothing_applied = []
        
        with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
            for message in conversation:
                # Update affection
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, message
                )
                
                # Get contextual result for verification
                contextual_result = self.enhanced_adapter.last_contextual_result
                
                if contextual_result:
                    # Store raw and adjusted scores
                    raw_scores.append(contextual_result.raw_sentiment.sentiment_score)
                    adjusted_scores.append(contextual_result.adjusted_sentiment_score)
                    
                    # Check if smoothing was applied
                    if contextual_result.sentiment_shift and hasattr(contextual_result.sentiment_shift, 'smoothing_applied'):
                        smoothing_applied.append(contextual_result.sentiment_shift.smoothing_applied)
                    else:
                        smoothing_applied.append(False)
                
                # Update conversation history
                mock_response = f"Response to: {message}"
                self.session_manager.update_conversation_history(
                    session_id, message, mock_response
                )
        
        # Verify smoothing was applied for dramatic shifts
        self.assertEqual(len(raw_scores), len(conversation))
        self.assertEqual(len(adjusted_scores), len(conversation))
        
        # Check for dramatic shifts where smoothing should be applied
        for i in range(1, len(raw_scores)):
            shift_magnitude = abs(raw_scores[i] - raw_scores[i-1])
            if shift_magnitude > 0.7:  # Large shift
                self.assertTrue(smoothing_applied[i], 
                              f"Smoothing should be applied for dramatic shift at message {i}: {conversation[i]}")    #
# SECTION 4: INTEGRATION WITH APP FUNCTIONALITY
    
    @patch('requests.post')
    def test_integration_with_chat_function(self, mock_post):
        """Test integration with the chat function"""
        # Mock LM Studio API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Mock assistant response"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Replace the sentiment analyzer in the affection tracker with our enhanced adapter
        with patch('affection_system.SentimentAnalyzer', return_value=self.enhanced_adapter):
            # Create a new session
            session_id = self.session_manager.create_new_session()
            
            # Test with context-dependent sentiment
            user_input = "素晴らしいですね、また失敗しましたか"
            history = []
            
            # Call chat function
            response, updated_history = chat(
                user_input=user_input,
                system_prompt=self.base_prompt,
                history=history,
                session_id=session_id
            )
            
            # Verify response was generated
            self.assertEqual(response, "Mock assistant response")
            self.assertEqual(len(updated_history), 1)
            
            # Verify affection was updated using enhanced sentiment analysis
            final_affection = self.session_manager.get_affection_level(session_id)
            self.assertNotEqual(final_affection, 15)  # Should be different from default
            
            # Verify conversation history was updated
            session = self.session_manager.get_session(session_id)
            self.assertEqual(len(session.conversation_history), 1)
    
    def test_integration_with_on_submit_function(self):
        """Test integration with the on_submit function"""
        # Replace the sentiment analyzer in the affection tracker with our enhanced adapter
        with patch('affection_system.SentimentAnalyzer', return_value=self.enhanced_adapter):
            # Test with context-dependent sentiment
            msg = "素晴らしいですね、また失敗しましたか"
            history = []
            session_id = None
            relationship_info = {}
            
            with patch('app.chat') as mock_chat:
                mock_chat.return_value = ("Mock response", [("素晴らしいですね、また失敗しましたか", "Mock response")])
                
                # Call on_submit
                result = on_submit(msg, history, session_id, relationship_info)
                
                # Verify return values
                empty_input, updated_chatbot, updated_history, new_session_id, new_relationship_info = result
                
                self.assertEqual(empty_input, "")
                self.assertEqual(len(updated_chatbot), 1)
                self.assertEqual(updated_history, updated_chatbot)
                self.assertIsNotNone(new_session_id)
                
                # Verify chat was called with session ID
                mock_chat.assert_called_once()
                call_args = mock_chat.call_args
                self.assertIsNotNone(call_args[1]['session_id'])

if __name__ == "__main__":
    unittest.main(verbosity=2)