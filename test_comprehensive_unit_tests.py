"""
Comprehensive Unit Tests for Mari AI Chat Affection System
Task 6.1: Create comprehensive unit tests for all affection system components
Tests Mari's personality consistency across affection levels, relationship progression, and edge cases
Requirements: 1.1, 1.2, 1.3, 2.3, 3.4
"""

import os
import shutil
import unittest
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

from affection_system import (
    SessionManager, AffectionTracker, initialize_affection_system,
    get_session_manager, get_affection_tracker
)
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult, SentimentType
from session_storage import UserSession, SessionStorage
from prompt_generator import PromptGenerator

class TestComprehensiveUnitTests(unittest.TestCase):
    """Comprehensive unit test suite for the affection system"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp(prefix="test_comprehensive_")
        
        # Initialize components
        self.session_manager = SessionManager(self.test_dir)
        self.affection_tracker = AffectionTracker(self.session_manager)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Create a test session
        self.test_session_id = self.session_manager.create_new_session()
        
        # Create a base prompt for testing
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

    # SECTION 1: SESSION MANAGER UNIT TESTS
    
    def test_session_manager_initialization(self):
        """Test SessionManager initialization"""
        # Test with default directory
        sm = SessionManager()
        self.assertIsNotNone(sm.storage)
        self.assertEqual(sm.storage_dir, "sessions")
        self.assertEqual(sm.current_sessions, {})
        
        # Test with custom directory
        custom_dir = os.path.join(self.test_dir, "custom_sessions")
        sm_custom = SessionManager(custom_dir)
        self.assertEqual(sm_custom.storage_dir, custom_dir)
        self.assertTrue(os.path.exists(custom_dir))
    
    def test_session_id_generation(self):
        """Test session ID generation"""
        session_id1 = self.session_manager.generate_session_id()
        session_id2 = self.session_manager.generate_session_id()
        
        # Should be unique
        self.assertNotEqual(session_id1, session_id2)
        
        # Should be valid UUID format
        import uuid
        try:
            uuid.UUID(session_id1)
            uuid.UUID(session_id2)
        except ValueError:
            self.fail("Generated session IDs are not valid UUIDs")
    
    def test_create_new_session(self):
        """Test new session creation"""
        # Test with auto-generated ID
        session_id = self.session_manager.create_new_session()
        self.assertIsNotNone(session_id)
        
        # Verify session exists in memory
        self.assertIn(session_id, self.session_manager.current_sessions)
        
        # Verify session has correct default values
        session = self.session_manager.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.affection_level, 15)
        self.assertEqual(session.conversation_history, [])
        self.assertIsNotNone(session.session_start_time)
        self.assertIsNotNone(session.last_interaction)
        
        # Test with custom ID
        custom_id = "test-custom-id"
        custom_session_id = self.session_manager.create_new_session(custom_id)
        self.assertEqual(custom_session_id, custom_id)
        
        # Verify session file was created
        session_file = os.path.join(self.test_dir, f"{session_id}.json")
        self.assertTrue(os.path.exists(session_file))
    
    def test_get_session(self):
        """Test session retrieval"""
        # Test getting existing session
        session = self.session_manager.get_session(self.test_session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, self.test_session_id)
        
        # Test getting non-existent session
        non_existent = self.session_manager.get_session("non-existent-id")
        self.assertIsNone(non_existent)
        
        # Test loading from storage
        # Remove from memory cache
        if self.test_session_id in self.session_manager.current_sessions:
            del self.session_manager.current_sessions[self.test_session_id]
        
        # Should still be able to load from storage
        session = self.session_manager.get_session(self.test_session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, self.test_session_id)
    
    def test_update_affection(self):
        """Test affection level updates"""
        # Test positive update
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        success = self.session_manager.update_affection(self.test_session_id, 10)
        self.assertTrue(success)
        
        new_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(new_level, initial_level + 10)
        
        # Test negative update
        success = self.session_manager.update_affection(self.test_session_id, -5)
        self.assertTrue(success)
        
        newer_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(newer_level, new_level - 5)
        
        # Test boundary conditions
        # Set to near maximum
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 95
        self.session_manager.save_session(self.test_session_id)
        
        # Try to exceed maximum
        self.session_manager.update_affection(self.test_session_id, 10)
        final_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(final_level, 100)  # Should be capped at 100
        
        # Set to near minimum
        session.affection_level = 5
        self.session_manager.save_session(self.test_session_id)
        
        # Try to go below minimum
        self.session_manager.update_affection(self.test_session_id, -10)
        final_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(final_level, 0)  # Should be capped at 0
        
        # Test update for non-existent session
        success = self.session_manager.update_affection("non-existent", 5)
        self.assertFalse(success)
    
    def test_conversation_history_update(self):
        """Test conversation history updates"""
        user_input = "Hello Mari"
        assistant_response = "What do you want?"
        
        # Test successful update
        success = self.session_manager.update_conversation_history(
            self.test_session_id, user_input, assistant_response
        )
        self.assertTrue(success)
        
        # Verify history was updated
        session = self.session_manager.get_session(self.test_session_id)
        self.assertEqual(len(session.conversation_history), 1)
        
        history_entry = session.conversation_history[0]
        self.assertEqual(history_entry["user"], user_input)
        self.assertEqual(history_entry["assistant"], assistant_response)
        self.assertIn("timestamp", history_entry)
        
        # Test update for non-existent session
        success = self.session_manager.update_conversation_history(
            "non-existent", "test", "test"
        )
        self.assertFalse(success)
    
    def test_session_cleanup(self):
        """Test old session cleanup"""
        # Create some test sessions with different ages
        old_session_id = self.session_manager.create_new_session()
        recent_session_id = self.session_manager.create_new_session()
        
        # Manually set old session's last interaction to 35 days ago
        old_session = self.session_manager.get_session(old_session_id)
        old_date = datetime.now() - timedelta(days=35)
        old_session.last_interaction = old_date.isoformat()
        self.session_manager.save_session(old_session_id)
        
        # Run cleanup for sessions older than 30 days
        cleaned_count = self.session_manager.cleanup_old_sessions(30)
        
        # Should have cleaned up at least the old session
        self.assertGreaterEqual(cleaned_count, 1)
        
        # Old session should no longer exist
        old_session_after_cleanup = self.session_manager.get_session(old_session_id)
        self.assertIsNone(old_session_after_cleanup)
        
        # Recent session should still exist
        recent_session_after_cleanup = self.session_manager.get_session(recent_session_id)
        self.assertIsNotNone(recent_session_after_cleanup)
    
    # SECTION 2: AFFECTION TRACKER UNIT TESTS
    
    def test_relationship_stage_mapping(self):
        """Test relationship stage determination"""
        test_cases = [
            (0, "hostile"),
            (15, "hostile"),
            (16, "distant"),
            (30, "distant"),
            (31, "cautious"),
            (50, "cautious"),
            (51, "friendly"),
            (70, "friendly"),
            (71, "warm"),
            (85, "warm"),
            (86, "close"),
            (100, "close")
        ]
        
        for affection, expected_stage in test_cases:
            actual_stage = self.affection_tracker.get_relationship_stage(affection)
            self.assertEqual(actual_stage, expected_stage,
                           f"Expected {expected_stage} for affection {affection}, got {actual_stage}")
    
    def test_relationship_descriptions(self):
        """Test relationship stage descriptions"""
        stages = ["hostile", "distant", "cautious", "friendly", "warm", "close"]
        
        for stage in stages:
            # Get affection level for this stage
            if stage == "hostile":
                affection = 10
            elif stage == "distant":
                affection = 25
            elif stage == "cautious":
                affection = 40
            elif stage == "friendly":
                affection = 60
            elif stage == "warm":
                affection = 80
            else:  # close
                affection = 95
            
            description = self.affection_tracker.get_relationship_description(affection)
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 10)  # Should be a meaningful description
    
    def test_mari_behavioral_state(self):
        """Test Mari's behavioral state mapping"""
        affection_levels = [10, 25, 40, 60, 80, 95]
        
        for level in affection_levels:
            state = self.affection_tracker.get_mari_behavioral_state(level)
            
            # Verify required fields are present
            required_fields = [
                "core_personality", "speech_patterns", "first_person",
                "stage", "stage_traits", "description"
            ]
            
            for field in required_fields:
                self.assertIn(field, state, f"Missing field {field} in behavioral state")
            
            # Verify core personality consistency
            self.assertEqual(state["first_person"], "あたし")
            self.assertIn("〜だろ", state["speech_patterns"])
            self.assertIn("〜じゃねーか", state["speech_patterns"])
            self.assertIn("うっせー", state["speech_patterns"])
            
            # Verify stage-specific traits
            stage_traits = state["stage_traits"]
            required_stage_fields = [
                "openness", "trust", "vulnerability", "communication_style",
                "emotional_expression", "characteristic_phrases", "relationship_dynamics"
            ]
            
            for field in required_stage_fields:
                self.assertIn(field, stage_traits, f"Missing stage trait {field}")
    
    def test_sentiment_analysis_integration(self):
        """Test sentiment analysis integration"""
        # Test positive sentiment
        positive_input = "ありがとう、とても助かった"
        sentiment_result = self.affection_tracker.analyze_user_sentiment(positive_input)
        
        self.assertIsInstance(sentiment_result, SentimentAnalysisResult)
        self.assertGreater(sentiment_result.sentiment_score, 0)
        self.assertGreater(sentiment_result.affection_delta, 0)
        
        # Test negative sentiment
        negative_input = "うるさい、バカ"
        sentiment_result = self.affection_tracker.analyze_user_sentiment(negative_input)
        
        self.assertLess(sentiment_result.sentiment_score, 0)
        self.assertLess(sentiment_result.affection_delta, 0)
        
        # Test neutral sentiment
        neutral_input = "今日は晴れています"
        sentiment_result = self.affection_tracker.analyze_user_sentiment(neutral_input)
        
        self.assertAlmostEqual(sentiment_result.sentiment_score, 0, delta=0.5)
        self.assertAlmostEqual(sentiment_result.affection_delta, 0, delta=1)
    
    def test_affection_delta_calculation(self):
        """Test affection delta calculation"""
        # Test various input types
        test_cases = [
            ("ありがとう", "positive"),
            ("うるさい", "negative"),
            ("今日は晴れ", "neutral"),
            ("", "neutral"),  # Empty input
            ("ありがとう、でもうざい", "mixed")  # Mixed sentiment
        ]
        
        for user_input, expected_type in test_cases:
            delta, sentiment_result = self.affection_tracker.calculate_affection_delta(user_input)
            
            self.assertIsInstance(delta, int)
            self.assertIsInstance(sentiment_result, SentimentAnalysisResult)
            
            if expected_type == "positive":
                self.assertGreater(delta, 0)
            elif expected_type == "negative":
                self.assertLess(delta, 0)
            elif expected_type == "neutral":
                self.assertAlmostEqual(delta, 0, delta=1)
    
    def test_affection_update_for_interaction(self):
        """Test affection updates for user interactions"""
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        
        # Test positive interaction
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, "ありがとう、とても助かった"
        )
        
        self.assertGreater(new_level, initial_level)
        self.assertGreater(sentiment_result.affection_delta, 0)
        
        # Verify sentiment history was recorded
        history = self.affection_tracker.get_sentiment_history(self.test_session_id)
        self.assertEqual(len(history), 1)
        
        history_entry = history[0]
        self.assertIn("timestamp", history_entry)
        self.assertIn("user_input", history_entry)
        self.assertIn("sentiment_score", history_entry)
        self.assertIn("affection_delta", history_entry)
    
    def test_gradual_affection_changes(self):
        """Test gradual affection change scheduling"""
        # Set initial affection
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 50
        self.session_manager.save_session(self.test_session_id)
        
        # Apply a large positive change
        very_positive = "あなたは私の人生で最高の出会いだよ。本当にありがとう。"
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        
        new_level, _ = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, very_positive
        )
        
        # Should have pending changes scheduled
        self.assertIn(self.test_session_id, self.affection_tracker.pending_affection_changes)
        pending_changes = self.affection_tracker.pending_affection_changes[self.test_session_id]
        self.assertGreater(len(pending_changes), 0)
        
        # Process pending changes
        self.affection_tracker._process_pending_affection_changes(self.test_session_id)
        
        # Affection should have increased further
        final_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertGreaterEqual(final_level, new_level)
    
    def test_sentiment_history_tracking(self):
        """Test sentiment history tracking"""
        interactions = [
            "こんにちは",
            "ありがとう",
            "うるさい",
            "あなたの話し方が好きだよ"
        ]
        
        # Process interactions
        for user_input in interactions:
            self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
        
        # Get history
        history = self.affection_tracker.get_sentiment_history(self.test_session_id)
        self.assertEqual(len(history), len(interactions))
        
        # Verify history structure
        for i, entry in enumerate(history):
            self.assertEqual(entry["user_input"], interactions[i])
            self.assertIn("timestamp", entry)
            self.assertIn("sentiment_score", entry)
            self.assertIn("interaction_type", entry)
            self.assertIn("affection_delta", entry)
            self.assertIn("detected_keywords", entry)
        
        # Test history limit
        limited_history = self.affection_tracker.get_sentiment_history(self.test_session_id, limit=2)
        self.assertEqual(len(limited_history), 2)
        self.assertEqual(limited_history, history[-2:])  # Should be the last 2 entries
    
    # SECTION 3: SENTIMENT ANALYZER UNIT TESTS
    
    def test_sentiment_analyzer_initialization(self):
        """Test SentimentAnalyzer initialization"""
        analyzer = SentimentAnalyzer()
        
        # Verify keyword dictionaries are loaded
        self.assertGreater(len(analyzer.positive_keywords), 0)
        self.assertGreater(len(analyzer.negative_keywords), 0)
        
        # Test specific keywords exist
        self.assertIn('ありがとう', analyzer.positive_keywords)
        self.assertIn('うざい', analyzer.negative_keywords)
    
    def test_sentiment_keyword_detection(self):
        """Test sentiment keyword detection"""
        # Test positive keywords
        positive_inputs = [
            "ありがとう",
            "素晴らしい",
            "好き",
            "嬉しい",
            "助かる"
        ]
        
        for input_text in positive_inputs:
            result = self.sentiment_analyzer.analyze_user_input(input_text)
            self.assertGreater(result.sentiment_score, 0,
                             f"Expected positive sentiment for '{input_text}'")
            self.assertIn(SentimentType.POSITIVE, result.sentiment_types)
        
        # Test negative keywords
        negative_inputs = [
            "うざい",
            "バカ",
            "きらい",
            "うるさい",
            "めんどくさい"
        ]
        
        for input_text in negative_inputs:
            result = self.sentiment_analyzer.analyze_user_input(input_text)
            self.assertLess(result.sentiment_score, 0,
                          f"Expected negative sentiment for '{input_text}'")
            self.assertIn(SentimentType.NEGATIVE, result.sentiment_types)
    
    def test_sentiment_intensity_calculation(self):
        """Test sentiment intensity calculation"""
        # Test different intensities
        test_cases = [
            ("ありがとう", 1),  # Single positive word
            ("ありがとう、とても助かった", 2),  # Multiple positive words
            ("本当にありがとう、素晴らしい、最高", 3),  # Many positive words
            ("うざい", -1),  # Single negative word
            ("うざい、バカ", -2),  # Multiple negative words
            ("うざい、バカ、きらい、うるさい", -4)  # Many negative words
        ]
        
        for input_text, expected_intensity in test_cases:
            result = self.sentiment_analyzer.analyze_user_input(input_text)
            if expected_intensity > 0:
                self.assertGreater(result.sentiment_score, 0)
                self.assertGreater(abs(result.sentiment_score), abs(expected_intensity) - 1)
            elif expected_intensity < 0:
                self.assertLess(result.sentiment_score, 0)
                self.assertGreater(abs(result.sentiment_score), abs(expected_intensity) - 1)
    
    def test_mixed_sentiment_handling(self):
        """Test handling of mixed sentiment inputs"""
        mixed_inputs = [
            "ありがとう、でもうざい",
            "好きだけど、めんどくさい",
            "素晴らしい、しかしバカ"
        ]
        
        for input_text in mixed_inputs:
            result = self.sentiment_analyzer.analyze_user_input(input_text)
            
            # Should detect both positive and negative sentiment
            self.assertIn(SentimentType.POSITIVE, result.sentiment_types)
            self.assertIn(SentimentType.NEGATIVE, result.sentiment_types)
            
            # Net sentiment should be close to neutral
            self.assertLessEqual(abs(result.sentiment_score), 2)
    
    # SECTION 4: PROMPT GENERATOR UNIT TESTS
    
    def test_prompt_generator_initialization(self):
        """Test PromptGenerator initialization"""
        generator = PromptGenerator(self.base_prompt)
        self.assertEqual(generator.base_prompt, self.base_prompt)
    
    def test_dynamic_prompt_generation(self):
        """Test dynamic prompt generation for different affection levels"""
        affection_levels = [10, 25, 40, 60, 80, 95]
        
        for level in affection_levels:
            prompt = self.prompt_generator.generate_dynamic_prompt(level)
            
            # Verify base prompt is included
            self.assertIn("麻理（まり）", prompt)
            self.assertIn("警戒心が強い", prompt)
            self.assertIn("一人称は「あたし」", prompt)
            
            # Verify relationship stage information is added
            stage = self.affection_tracker.get_relationship_stage(level)
            if stage == "hostile":
                self.assertIn("極端に警戒し、敵対的・攻撃的な態度", prompt)
            elif stage == "close":
                self.assertIn("深い信頼関係が形成され、素直な感情表現が増える", prompt)
    
    def test_character_consistency_validation(self):
        """Test character consistency validation"""
        # Test valid prompt (should pass)
        valid_prompt = self.base_prompt
        self.assertTrue(self.prompt_generator.validate_character_consistency(valid_prompt))
        
        # Test prompt missing core traits (should fail)
        invalid_prompt = "You are a helpful assistant."
        self.assertFalse(self.prompt_generator.validate_character_consistency(invalid_prompt))
        
        # Test prompt with some but not all traits
        partial_prompt = """
        あなたは「麻理（まり）」という人格を持ったAIです。
        - 警戒心が強い
        - 一人称は「あたし」
        """
        # This should still fail as it's missing many core traits
        self.assertFalse(self.prompt_generator.validate_character_consistency(partial_prompt))
    
    def test_relationship_context_generation(self):
        """Test relationship context generation"""
        affection_levels = [10, 25, 40, 60, 80, 95]
        
        for level in affection_levels:
            context = self.prompt_generator.get_relationship_context(level)
            
            self.assertIsInstance(context, str)
            self.assertGreater(len(context), 10)  # Should be meaningful context
            
            # Verify context matches relationship stage
            stage = self.affection_tracker.get_relationship_stage(level)
            if stage == "hostile":
                self.assertIn("警戒", context)
            elif stage == "close":
                self.assertIn("信頼", context)
    
    def test_affection_modified_prompt(self):
        """Test affection-modified prompt generation"""
        base_prompt = "Base system prompt"
        affection_levels = [10, 50, 90]
        
        for level in affection_levels:
            modified_prompt = self.prompt_generator.get_affection_modified_prompt(base_prompt, level)
            
            # Should contain base prompt
            self.assertIn(base_prompt, modified_prompt)
            
            # Should contain relationship-specific modifications
            self.assertGreater(len(modified_prompt), len(base_prompt))
    
    # SECTION 5: INTEGRATION TESTS
    
    def test_full_system_integration(self):
        """Test full system integration"""
        # Initialize system
        storage_dir = os.path.join(self.test_dir, "integration_test")
        session_mgr, affection_trk = initialize_affection_system(storage_dir, auto_load_sessions=False)
        
        # Verify global instances are set
        self.assertIsNotNone(get_session_manager())
        self.assertIsNotNone(get_affection_tracker())
        
        # Create session and test full flow
        session_id = session_mgr.create_new_session()
        
        # Test interaction flow
        user_input = "ありがとう、とても助かった"
        new_level, sentiment_result = affection_trk.update_affection_for_interaction(session_id, user_input)
        
        # Verify affection increased
        self.assertGreater(new_level, 15)  # Should be higher than default
        
        # Generate dynamic prompt
        prompt = self.prompt_generator.generate_dynamic_prompt(new_level)
        self.assertIn("麻理", prompt)
        
        # Update conversation history
        success = session_mgr.update_conversation_history(session_id, user_input, "Test response")
        self.assertTrue(success)
        
        # Verify session persistence
        session = session_mgr.get_session(session_id)
        self.assertEqual(len(session.conversation_history), 1)
    
    # SECTION 6: ERROR HANDLING AND EDGE CASES
    
    def test_corrupted_session_handling(self):
        """Test handling of corrupted session files"""
        # Create a session
        session_id = self.session_manager.create_new_session()
        
        # Corrupt the session file
        file_path = os.path.join(self.test_dir, f"{session_id}.json")
        with open(file_path, 'w') as f:
            f.write("This is not valid JSON")
        
        # Remove from memory cache
        if session_id in self.session_manager.current_sessions:
            del self.session_manager.current_sessions[session_id]
        
        # Try to load the corrupted session
        loaded_session = self.session_manager.get_session(session_id)
        
        # Should return None for corrupted session
        self.assertIsNone(loaded_session)
    
    def test_empty_input_handling(self):
        """Test handling of empty and whitespace-only inputs"""
        empty_inputs = ["", "   ", "\n", "\t", "  \n  \t  "]
        
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        
        for empty_input in empty_inputs:
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, empty_input
            )
            
            # Affection should not change for empty inputs
            self.assertEqual(new_level, initial_level)
            self.assertEqual(sentiment_result.affection_delta, 0)
    
    def test_boundary_affection_values(self):
        """Test affection system behavior at boundary values"""
        # Test at minimum affection (0)
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 0
        self.session_manager.save_session(self.test_session_id)
        
        # Apply negative input - should not go below 0
        new_level, _ = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, "うざい、バカ、消えろ"
        )
        self.assertGreaterEqual(new_level, 0)
        
        # Test at maximum affection (100)
        session.affection_level = 100
        self.session_manager.save_session(self.test_session_id)
        
        # Apply positive input - should not go above 100
        new_level, _ = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, "ありがとう、素晴らしい、最高"
        )
        self.assertLessEqual(new_level, 100)
    
    def test_concurrent_session_access(self):
        """Test concurrent access to the same session"""
        # This test simulates concurrent access patterns
        session_id = self.session_manager.create_new_session()
        
        # Simulate multiple rapid updates
        inputs = ["ありがとう", "うるさい", "好き", "バカ", "助かる"]
        
        for user_input in inputs:
            # Update affection
            new_level, _ = self.affection_tracker.update_affection_for_interaction(
                session_id, user_input
            )
            
            # Update conversation history
            success = self.session_manager.update_conversation_history(
                session_id, user_input, f"Response at level {new_level}"
            )
            self.assertTrue(success)
        
        # Verify final state is consistent
        session = self.session_manager.get_session(session_id)
        self.assertEqual(len(session.conversation_history), len(inputs))
        
        # Verify sentiment history matches
        sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
        self.assertEqual(len(sentiment_history), len(inputs))
    
    def test_large_conversation_history(self):
        """Test system behavior with large conversation histories"""
        # Create a session with many interactions
        session_id = self.session_manager.create_new_session()
        
        # Add many conversation entries
        for i in range(100):
            user_input = f"Message {i}"
            assistant_response = f"Response {i}"
            
            success = self.session_manager.update_conversation_history(
                session_id, user_input, assistant_response
            )
            self.assertTrue(success)
        
        # Verify all entries are stored
        session = self.session_manager.get_session(session_id)
        self.assertEqual(len(session.conversation_history), 100)
        
        # Verify session can still be saved and loaded
        self.assertTrue(self.session_manager.save_session(session_id))
        
        # Remove from memory and reload
        if session_id in self.session_manager.current_sessions:
            del self.session_manager.current_sessions[session_id]
        
        reloaded_session = self.session_manager.get_session(session_id)
        self.assertIsNotNone(reloaded_session)
        self.assertEqual(len(reloaded_session.conversation_history), 100)
    
    def test_invalid_affection_values(self):
        """Test handling of invalid affection values"""
        # Test with invalid session data
        session = self.session_manager.get_session(self.test_session_id)
        
        # Manually set invalid affection values
        session.affection_level = -10
        self.session_manager.save_session(self.test_session_id)
        
        # System should handle this gracefully
        level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertGreaterEqual(level, 0)
        
        # Test with extremely high value
        session.affection_level = 150
        self.session_manager.save_session(self.test_session_id)
        
        level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertLessEqual(level, 100)
    
    def test_memory_cleanup(self):
        """Test memory cleanup for session cache"""
        # Create many sessions
        session_ids = []
        for i in range(50):
            session_id = self.session_manager.create_new_session()
            session_ids.append(session_id)
        
        # Verify all sessions are in memory
        self.assertEqual(len(self.session_manager.current_sessions), len(session_ids) + 1)  # +1 for test_session_id
        
        # Run cleanup
        cleaned_count = self.session_manager.cleanup_old_sessions(0)  # Clean all sessions
        
        # Verify sessions were cleaned from storage
        self.assertGreater(cleaned_count, 0)
        
        # Memory cache should be updated
        remaining_sessions = len(self.session_manager.current_sessions)
        self.assertLess(remaining_sessions, len(session_ids))

if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
