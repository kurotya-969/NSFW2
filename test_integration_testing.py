"""
Integration Testing for Mari AI Chat Affection System
Task 6.2: Implement integration testing
Tests complete conversation flows, session persistence, and system behavior
Requirements: 4.4, 5.3, 5.4
"""

import os
import shutil
import unittest
import tempfile
import time
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from affection_system import (
    SessionManager, AffectionTracker, initialize_affection_system,
    get_session_manager, get_affection_tracker
)
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult
from session_storage import UserSession, SessionStorage
from prompt_generator import PromptGenerator
from app import chat, on_submit, clear_history

class TestIntegrationTesting(unittest.TestCase):
    """Integration test suite for the affection system"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp(prefix="test_integration_")
        
        # Initialize the affection system
        self.session_manager, self.affection_tracker = initialize_affection_system(
            self.test_dir, auto_load_sessions=False
        )
        
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

    # SECTION 1: COMPLETE CONVERSATION FLOW TESTS
    
    def test_complete_conversation_flow_with_affection_changes(self):
        """Test complete conversation flows with affection changes"""
        # Create a new session
        session_id = self.session_manager.create_new_session()
        initial_affection = self.session_manager.get_affection_level(session_id)
        
        # Define a conversation flow that should increase affection
        conversation_flow = [
            ("こんにちは", "greeting"),
            ("今日の天気はどう？", "neutral_question"),
            ("ありがとう", "appreciation"),
            ("あなたの話し方が好きだよ", "positive_feedback"),
            ("大丈夫？心配してるよ", "caring"),
            ("いつも助けてくれてありがとう", "gratitude"),
            ("あなたは大切な存在だよ", "deep_appreciation")
        ]
        
        affection_levels = [initial_affection]
        relationship_stages = [self.affection_tracker.get_relationship_stage(initial_affection)]
        
        # Process each message in the conversation
        for user_input, interaction_type in conversation_flow:
            # Update affection based on user input
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                session_id, user_input
            )
            
            # Generate dynamic prompt based on current affection
            dynamic_prompt = self.prompt_generator.generate_dynamic_prompt(new_level)
            
            # Simulate assistant response (in real app, this would come from LM Studio)
            mock_response = f"Mock response at affection level {new_level}"
            
            # Update conversation history
            success = self.session_manager.update_conversation_history(
                session_id, user_input, mock_response
            )
            self.assertTrue(success)
            
            # Track progression
            affection_levels.append(new_level)
            relationship_stages.append(self.affection_tracker.get_relationship_stage(new_level))
        
        # Verify affection progression
        final_affection = affection_levels[-1]
        self.assertGreater(final_affection, initial_affection, 
                          "Affection should have increased through positive conversation")
        
        # Verify conversation history was properly stored
        session = self.session_manager.get_session(session_id)
        self.assertEqual(len(session.conversation_history), len(conversation_flow))
        
        # Verify sentiment history was tracked
        sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
        self.assertEqual(len(sentiment_history), len(conversation_flow))
        
        # Verify relationship stage progression
        self.assertGreaterEqual(
            self.affection_tracker.get_relationship_stage(final_affection),
            self.affection_tracker.get_relationship_stage(initial_affection)
        )
    
    def test_negative_conversation_flow(self):
        """Test conversation flow with negative interactions"""
        # Create a session with higher initial affection
        session_id = self.session_manager.create_new_session()
        session = self.session_manager.get_session(session_id)
        session.affection_level = 60  # Start at friendly level
        self.session_manager.save_session(session_id)
        
        initial_affection = self.session_manager.get_affection_level(session_id)
        
        # Define a negative conversation flow
        negative_flow = [
            ("うるさいな", "mild_negative"),
            ("めんどくさい", "dismissive"),
            ("知らないよ", "indifferent"),
            ("うざい", "negative"),
            ("バカじゃないの？", "insulting"),
            ("黙れ", "hostile")
        ]
        
        affection_levels = [initial_affection]
        
        # Process each negative message
        for user_input, interaction_type in negative_flow:
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                session_id, user_input
            )
            
            # Verify negative sentiment was detected
            self.assertLess(sentiment_result.affection_delta, 0,
                          f"Expected negative delta for '{user_input}'")
            
            # Update conversation history
            mock_response = f"Defensive response at affection level {new_level}"
            self.session_manager.update_conversation_history(
                session_id, user_input, mock_response
            )
            
            affection_levels.append(new_level)
        
        # Verify affection decreased
        final_affection = affection_levels[-1]
        self.assertLess(final_affection, initial_affection,
                       "Affection should have decreased through negative conversation")
        
        # Verify relationship stage regressed appropriately
        initial_stage = self.affection_tracker.get_relationship_stage(initial_affection)
        final_stage = self.affection_tracker.get_relationship_stage(final_affection)
        
        stage_order = ["hostile", "distant", "cautious", "friendly", "warm", "close"]
        initial_index = stage_order.index(initial_stage)
        final_index = stage_order.index(final_stage)
        
        self.assertLessEqual(final_index, initial_index,
                           f"Relationship stage should not have improved: {initial_stage} -> {final_stage}")
    
    def test_mixed_conversation_patterns(self):
        """Test conversation with mixed positive and negative interactions"""
        session_id = self.session_manager.create_new_session()
        session = self.session_manager.get_session(session_id)
        session.affection_level = 40  # Start at cautious level
        self.session_manager.save_session(session_id)
        
        # Define alternating positive and negative interactions
        mixed_flow = [
            ("ありがとう", "positive"),
            ("うるさいな", "negative"),
            ("あなたの話し方が好きだよ", "positive"),
            ("めんどくさい", "negative"),
            ("大丈夫？心配してるよ", "positive"),
            ("知らないよ", "negative"),
            ("いつも助けてくれてありがとう", "positive")
        ]
        
        affection_levels = []
        sentiment_deltas = []
        
        # Process mixed interactions
        for user_input, expected_sentiment in mixed_flow:
            initial_level = self.session_manager.get_affection_level(session_id)
            
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                session_id, user_input
            )
            
            # Verify sentiment matches expectation
            if expected_sentiment == "positive":
                self.assertGreaterEqual(sentiment_result.affection_delta, 0,
                                      f"Expected positive delta for '{user_input}'")
            else:
                self.assertLessEqual(sentiment_result.affection_delta, 0,
                                   f"Expected negative delta for '{user_input}'")
            
            affection_levels.append(new_level)
            sentiment_deltas.append(sentiment_result.affection_delta)
            
            # Update conversation history
            mock_response = f"Response reflecting {expected_sentiment} interaction"
            self.session_manager.update_conversation_history(
                session_id, user_input, mock_response
            )
        
        # Verify conversation history is complete
        session = self.session_manager.get_session(session_id)
        self.assertEqual(len(session.conversation_history), len(mixed_flow))
        
        # Verify sentiment history tracks all interactions
        sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
        self.assertEqual(len(sentiment_history), len(mixed_flow))
    
    # SECTION 2: SESSION PERSISTENCE TESTS
    
    def test_session_persistence_across_restarts(self):
        """Test session persistence works across app restarts"""
        # Create a session with some conversation history
        session_id = self.session_manager.create_new_session()
        
        # Add some interactions
        interactions = [
            ("こんにちは", "Hello response"),
            ("ありがとう", "Thanks response"),
            ("あなたの話し方が好きだよ", "Appreciation response")
        ]
        
        for user_input, assistant_response in interactions:
            # Update affection
            new_level, _ = self.affection_tracker.update_affection_for_interaction(
                session_id, user_input
            )
            
            # Update conversation history
            self.session_manager.update_conversation_history(
                session_id, user_input, assistant_response
            )
        
        # Get current state
        original_session = self.session_manager.get_session(session_id)
        original_affection = original_session.affection_level
        original_history_length = len(original_session.conversation_history)
        original_sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
        
        # Simulate app restart by creating new instances
        new_session_manager = SessionManager(self.test_dir)
        new_affection_tracker = AffectionTracker(new_session_manager)
        
        # Load the session
        restored_session = new_session_manager.get_session(session_id)
        
        # Verify session was restored correctly
        self.assertIsNotNone(restored_session)
        self.assertEqual(restored_session.affection_level, original_affection)
        self.assertEqual(len(restored_session.conversation_history), original_history_length)
        
        # Verify conversation history content
        for i, (user_input, assistant_response) in enumerate(interactions):
            history_entry = restored_session.conversation_history[i]
            self.assertEqual(history_entry["user"], user_input)
            self.assertEqual(history_entry["assistant"], assistant_response)
            self.assertIn("timestamp", history_entry)
        
        # Continue conversation with restored session
        new_level, _ = new_affection_tracker.update_affection_for_interaction(
            session_id, "継続してありがとう"
        )
        
        # Verify affection continues to work correctly
        self.assertGreater(new_level, original_affection)
        
        # Update conversation history
        success = new_session_manager.update_conversation_history(
            session_id, "継続してありがとう", "Continued response"
        )
        self.assertTrue(success)
        
        # Verify updated session can be saved and loaded again
        self.assertTrue(new_session_manager.save_session(session_id))
        
        # Load again to verify persistence
        final_session = new_session_manager.get_session(session_id)
        self.assertEqual(len(final_session.conversation_history), original_history_length + 1)
    
    def test_multiple_session_persistence(self):
        """Test persistence of multiple concurrent sessions"""
        # Create multiple sessions with different states
        sessions_data = []
        
        for i in range(5):
            session_id = self.session_manager.create_new_session()
            
            # Set different affection levels
            session = self.session_manager.get_session(session_id)
            session.affection_level = 20 + (i * 15)  # 20, 35, 50, 65, 80
            self.session_manager.save_session(session_id)
            
            # Add different conversation histories
            for j in range(i + 1):  # Different number of messages per session
                user_input = f"Message {j} from session {i}"
                assistant_response = f"Response {j} to session {i}"
                
                self.session_manager.update_conversation_history(
                    session_id, user_input, assistant_response
                )
            
            sessions_data.append({
                'id': session_id,
                'affection': session.affection_level,
                'history_length': i + 1
            })
        
        # Simulate restart
        new_session_manager = SessionManager(self.test_dir)
        
        # Verify all sessions were restored correctly
        for session_data in sessions_data:
            restored_session = new_session_manager.get_session(session_data['id'])
            
            self.assertIsNotNone(restored_session)
            self.assertEqual(restored_session.affection_level, session_data['affection'])
            self.assertEqual(len(restored_session.conversation_history), session_data['history_length'])
    
    def test_session_cleanup_persistence(self):
        """Test that session cleanup works correctly with persistence"""
        # Create sessions with different ages
        current_time = datetime.now()
        
        # Recent session (should not be cleaned)
        recent_session_id = self.session_manager.create_new_session()
        recent_session = self.session_manager.get_session(recent_session_id)
        recent_session.last_interaction = current_time.isoformat()
        self.session_manager.save_session(recent_session_id)
        
        # Old session (should be cleaned)
        old_session_id = self.session_manager.create_new_session()
        old_session = self.session_manager.get_session(old_session_id)
        old_time = current_time - timedelta(days=35)
        old_session.last_interaction = old_time.isoformat()
        self.session_manager.save_session(old_session_id)
        
        # Verify both sessions exist initially
        self.assertIsNotNone(self.session_manager.get_session(recent_session_id))
        self.assertIsNotNone(self.session_manager.get_session(old_session_id))
        
        # Run cleanup
        cleaned_count = self.session_manager.cleanup_old_sessions(30)
        self.assertGreaterEqual(cleaned_count, 1)
        
        # Verify old session was cleaned but recent session remains
        self.assertIsNotNone(self.session_manager.get_session(recent_session_id))
        self.assertIsNone(self.session_manager.get_session(old_session_id))
        
        # Simulate restart and verify cleanup persisted
        new_session_manager = SessionManager(self.test_dir)
        
        self.assertIsNotNone(new_session_manager.get_session(recent_session_id))
        self.assertIsNone(new_session_manager.get_session(old_session_id))
    
    # SECTION 3: SYSTEM BEHAVIOR TESTS
    
    def test_various_user_input_patterns(self):
        """Test system behavior with various user input patterns"""
        session_id = self.session_manager.create_new_session()
        
        # Test different input patterns
        input_patterns = [
            # Empty and whitespace
            ("", "empty"),
            ("   ", "whitespace"),
            ("\n\t", "special_chars"),
            
            # Single words
            ("ありがとう", "single_positive"),
            ("うざい", "single_negative"),
            
            # Long messages
            ("今日は本当に素晴らしい一日でした。あなたのおかげで多くのことを学ぶことができて、心から感謝しています。", "long_positive"),
            ("もう本当にうんざりだ。何もかもがうまくいかないし、全部めんどくさい。", "long_negative"),
            
            # Mixed language
            ("Thank you ありがとう", "mixed_language"),
            ("Hello こんにちは", "mixed_greeting"),
            
            # Special characters and numbers
            ("123456", "numbers"),
            ("!@#$%^&*()", "special_symbols"),
            ("😊😊😊", "emoji"),
            
            # Repeated words
            ("ありがとうありがとうありがとう", "repeated_positive"),
            ("うざいうざいうざい", "repeated_negative")
        ]
        
        initial_affection = self.session_manager.get_affection_level(session_id)
        
        for user_input, pattern_type in input_patterns:
            try:
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, user_input
                )
                
                # Verify system handles all input types gracefully
                self.assertIsInstance(new_level, int)
                self.assertGreaterEqual(new_level, 0)
                self.assertLessEqual(new_level, 100)
                
                # Verify sentiment result is valid
                self.assertIsInstance(sentiment_result.sentiment_score, (int, float))
                self.assertIsInstance(sentiment_result.affection_delta, int)
                
                # Update conversation history
                mock_response = f"Response to {pattern_type} input"
                success = self.session_manager.update_conversation_history(
                    session_id, user_input, mock_response
                )
                self.assertTrue(success)
                
            except Exception as e:
                self.fail(f"System failed to handle input pattern '{pattern_type}': {str(e)}")
        
        # Verify all interactions were recorded
        session = self.session_manager.get_session(session_id)
        self.assertEqual(len(session.conversation_history), len(input_patterns))
        
        sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
        self.assertEqual(len(sentiment_history), len(input_patterns))
    
    def test_rapid_interaction_handling(self):
        """Test system behavior with rapid successive interactions"""
        session_id = self.session_manager.create_new_session()
        
        # Simulate rapid interactions
        rapid_inputs = [
            "こんにちは",
            "ありがとう",
            "好き",
            "嬉しい",
            "素晴らしい",
            "最高",
            "感謝",
            "助かる",
            "良い",
            "幸せ"
        ]
        
        start_time = time.time()
        
        # Process inputs rapidly
        for i, user_input in enumerate(rapid_inputs):
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                session_id, user_input
            )
            
            # Update conversation history
            mock_response = f"Rapid response {i}"
            success = self.session_manager.update_conversation_history(
                session_id, user_input, mock_response
            )
            self.assertTrue(success)
            
            # Verify system remains responsive
            self.assertIsInstance(new_level, int)
            self.assertGreaterEqual(new_level, 0)
            self.assertLessEqual(new_level, 100)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify processing completed in reasonable time (should be very fast)
        self.assertLess(processing_time, 5.0, "Rapid interaction processing took too long")
        
        # Verify all interactions were recorded correctly
        session = self.session_manager.get_session(session_id)
        self.assertEqual(len(session.conversation_history), len(rapid_inputs))
        
        sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
        self.assertEqual(len(sentiment_history), len(rapid_inputs))
        
        # Verify affection increased due to positive inputs
        final_affection = self.session_manager.get_affection_level(session_id)
        self.assertGreater(final_affection, 15)  # Should be higher than default
    
    # SECTION 4: FASTAPI AND GRADIO FUNCTIONALITY TESTS
    
    @patch('requests.post')
    def test_chat_function_integration(self, mock_post):
        """Test chat function integration with affection system"""
        # Mock LM Studio API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Mock assistant response"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Test chat function with session ID
        session_id = self.session_manager.create_new_session()
        user_input = "ありがとう、とても助かった"
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
        self.assertEqual(updated_history[0][0], user_input)
        self.assertEqual(updated_history[0][1], "Mock assistant response")
        
        # Verify affection was updated
        final_affection = self.session_manager.get_affection_level(session_id)
        self.assertGreater(final_affection, 15)  # Should be higher than default
        
        # Verify conversation history was updated
        session = self.session_manager.get_session(session_id)
        self.assertEqual(len(session.conversation_history), 1)
        
        # Verify API was called with dynamic prompt
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        
        # Verify system message includes dynamic prompt modifications
        system_message = request_data['messages'][0]
        self.assertEqual(system_message['role'], 'system')
        self.assertIn('麻理', system_message['content'])
    
    def test_on_submit_function_integration(self):
        """Test on_submit function integration"""
        # Test with new session
        msg = "こんにちは、麻理"
        history = []
        session_id = None
        relationship_info = {}
        
        with patch('app.chat') as mock_chat:
            mock_chat.return_value = ("Mock response", [("こんにちは、麻理", "Mock response")])
            
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
    
    def test_clear_history_function(self):
        """Test clear_history function"""
        # Call clear_history
        result = clear_history()
        
        # Verify return values
        empty_chatbot, empty_history, empty_session, empty_rel_info = result
        
        self.assertEqual(empty_chatbot, [])
        self.assertEqual(empty_history, [])
        self.assertIsNone(empty_session)
        self.assertEqual(empty_rel_info, {})
    
    # SECTION 5: ERROR HANDLING AND RECOVERY TESTS
    
    def test_system_recovery_from_errors(self):
        """Test system recovery from various error conditions"""
        session_id = self.session_manager.create_new_session()
        
        # Test recovery from file system errors
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # System should handle file permission errors gracefully
            try:
                new_level, _ = self.affection_tracker.update_affection_for_interaction(
                    session_id, "ありがとう"
                )
                # Should still work, just might not persist immediately
                self.assertIsInstance(new_level, int)
            except Exception as e:
                self.fail(f"System should handle file permission errors gracefully: {str(e)}")
        
        # Test recovery from corrupted data
        session = self.session_manager.get_session(session_id)
        if session:
            # Corrupt session data
            session.affection_level = "invalid_data"
            
            try:
                # System should handle corrupted data gracefully
                level = self.session_manager.get_affection_level(session_id)
                self.assertIsInstance(level, int)
                self.assertGreaterEqual(level, 0)
                self.assertLessEqual(level, 100)
            except Exception as e:
                self.fail(f"System should handle corrupted data gracefully: {str(e)}")
    
    def test_concurrent_access_handling(self):
        """Test handling of concurrent access to sessions"""
        session_id = self.session_manager.create_new_session()
        
        # Simulate concurrent updates
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def update_affection(input_text, thread_id):
            try:
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, f"{input_text} from thread {thread_id}"
                )
                results.put((thread_id, new_level, sentiment_result.affection_delta))
            except Exception as e:
                errors.put((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_affection, args=("ありがとう", i))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertTrue(errors.empty(), f"Concurrent access caused errors: {list(errors.queue)}")
        
        # Verify all updates were processed
        self.assertEqual(results.qsize(), 5)
        
        # Verify final session state is consistent
        session = self.session_manager.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertGreaterEqual(session.affection_level, 15)  # Should have increased
        
        # Verify sentiment history was recorded
        sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
        self.assertEqual(len(sentiment_history), 5)

if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
