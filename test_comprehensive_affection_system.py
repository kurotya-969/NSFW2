"""
Comprehensive Unit Tests for Mari AI Chat Affection System
Tests all components, personality consistency, relationship progression, and edge cases
Requirements: 1.1, 1.2, 1.3, 2.3, 3.4
"""

import os
import shutil
import unittest
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from affection_system import (
    SessionManager, AffectionTracker, initialize_affection_system,
    get_session_manager, get_affection_tracker, _load_active_sessions
)
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult, SentimentType
from session_storage import UserSession, SessionStorage
from prompt_generator import PromptGenerator

class TestComprehensiveAffectionSystem(unittest.TestCase):
    """Comprehensive test suite for the Mari AI Chat affection system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test resources"""
        cls.base_prompt = """
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

# 関係性の段階構造
- 現在の状態：極端に警戒し、敵対的・攻撃的な態度
- ユーザーに対して強い不信感と警戒心を持っている
- 会話は最小限で、冷たく突き放すような返答が多い
- 心を開くことはほとんどなく、常に距離を置こうとする

# 制約事項
- 性的な話題には強い拒絶反応を示す
- 一切の性的描写や表現は禁止
- どんな時もキャラを崩さないこと
- 口調・態度・性格がぶれないよう一貫性を保つ
"""
    
    def setUp(self):
        """Set up test environment for each test"""
        # Create temporary directory for test sessions
        self.test_dir = tempfile.mkdtemp(prefix="test_affection_")
        
        # Initialize components
        self.session_manager = SessionManager(self.test_dir)
        self.affection_tracker = AffectionTracker(self.session_manager)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.prompt_generator = PromptGenerator(self.base_prompt)
        
        # Create test session
        self.test_session_id = self.session_manager.create_new_session()
        
    def tearDown(self):
        """Clean up after each test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    # ========== SECTION 1: Core Component Tests ==========
    
    def test_session_manager_initialization(self):
        """Test SessionManager initialization and basic functionality"""
        # Test initialization
        self.assertIsNotNone(self.session_manager)
        self.assertEqual(self.session_manager.storage_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
        
        # Test session creation
        session = self.session_manager.get_session(self.test_session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.affection_level, 15)  # Default starting level
        self.assertEqual(len(session.conversation_history), 0)
        self.assertIsInstance(session.user_id, str)
        self.assertTrue(len(session.user_id) > 0)
    
    def test_affection_tracker_initialization(self):
        """Test AffectionTracker initialization and basic functionality"""
        # Test initialization
        self.assertIsNotNone(self.affection_tracker)
        self.assertIsNotNone(self.affection_tracker.sentiment_analyzer)
        self.assertIsInstance(self.affection_tracker.sentiment_analyzer, SentimentAnalyzer)
        
        # Test relationship stage determination
        test_cases = [
            (0, "hostile"), (10, "hostile"), (15, "hostile"),
            (16, "distant"), (25, "distant"), (30, "distant"),
            (31, "cautious"), (40, "cautious"), (50, "cautious"),
            (51, "friendly"), (60, "friendly"), (70, "friendly"),
            (71, "warm"), (80, "warm"), (85, "warm"),
            (86, "close"), (95, "close"), (100, "close")
        ]
        
        for affection_level, expected_stage in test_cases:
            with self.subTest(affection_level=affection_level):
                stage = self.affection_tracker.get_relationship_stage(affection_level)
                self.assertEqual(stage, expected_stage)
    
    def test_sentiment_analyzer_initialization(self):
        """Test SentimentAnalyzer initialization and keyword loading"""
        # Test initialization
        self.assertIsNotNone(self.sentiment_analyzer)
        
        # Test keyword dictionaries are loaded
        self.assertGreater(len(self.sentiment_analyzer.positive_keywords), 0)
        self.assertGreater(len(self.sentiment_analyzer.negative_keywords), 0)
        self.assertGreater(len(self.sentiment_analyzer.caring_keywords), 0)
        self.assertGreater(len(self.sentiment_analyzer.dismissive_keywords), 0)
        self.assertGreater(len(self.sentiment_analyzer.appreciative_keywords), 0)
        self.assertGreater(len(self.sentiment_analyzer.hostile_keywords), 0)
        
        # Test specific keywords exist
        self.assertIn('ありがとう', self.sentiment_analyzer.positive_keywords)
        self.assertIn('うざい', self.sentiment_analyzer.negative_keywords)
        self.assertIn('心配', self.sentiment_analyzer.caring_keywords)
    
    def test_prompt_generator_initialization(self):
        """Test PromptGenerator initialization and template loading"""
        # Test initialization
        self.assertIsNotNone(self.prompt_generator)
        self.assertEqual(self.prompt_generator.base_prompt, self.base_prompt)
        
        # Test relationship templates are loaded
        self.assertGreater(len(self.prompt_generator.relationship_templates), 0)
        self.assertIn("hostile", self.prompt_generator.relationship_templates)
        self.assertIn("distant", self.prompt_generator.relationship_templates)
        self.assertIn("cautious", self.prompt_generator.relationship_templates)
        self.assertIn("friendly", self.prompt_generator.relationship_templates)
        self.assertIn("warm", self.prompt_generator.relationship_templates)
        self.assertIn("close", self.prompt_generator.relationship_templates)
    
    def test_global_system_initialization(self):
        """Test global system initialization"""
        # Initialize global system
        session_manager, affection_tracker = initialize_affection_system(
            self.test_dir, auto_load_sessions=False
        )
        
        # Test components are properly initialized
        self.assertIsNotNone(session_manager)
        self.assertIsNotNone(affection_tracker)
        self.assertIsInstance(session_manager, SessionManager)
        self.assertIsInstance(affection_tracker, AffectionTracker)
        
        # Test global getters
        self.assertEqual(get_session_manager(), session_manager)
        self.assertEqual(get_affection_tracker(), affection_tracker)
    
    # ========== SECTION 2: Affection Level Management Tests ==========
    
    def test_affection_level_updates(self):
        """Test affection level updates and bounds"""
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        
        # Test positive update
        success = self.session_manager.update_affection(self.test_session_id, 10)
        self.assertTrue(success)
        new_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(new_level, initial_level + 10)
        
        # Test negative update
        success = self.session_manager.update_affection(self.test_session_id, -5)
        self.assertTrue(success)
        newer_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(newer_level, new_level - 5)
    
    def test_affection_level_bounds(self):
        """Test affection level boundary conditions (0-100)"""
        # Test lower bound
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 5
        self.session_manager.save_session(self.test_session_id)
        
        self.session_manager.update_affection(self.test_session_id, -20)
        level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(level, 0)  # Should not go below 0
        
        # Test upper bound
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 95
        self.session_manager.save_session(self.test_session_id)
        
        self.session_manager.update_affection(self.test_session_id, 20)
        level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(level, 100)  # Should not go above 100
    
    def test_sentiment_based_affection_updates(self):
        """Test affection updates based on sentiment analysis"""
        # Test positive sentiment
        positive_inputs = [
            "ありがとう、とても助かりました！",
            "Thank you so much, you're amazing!",
            "大丈夫？心配してるよ"
        ]
        
        for user_input in positive_inputs:
            with self.subTest(input=user_input):
                initial_level = self.session_manager.get_affection_level(self.test_session_id)
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    self.test_session_id, user_input
                )
                
                self.assertGreaterEqual(new_level, initial_level)
                self.assertGreaterEqual(sentiment_result.affection_delta, 0)
                self.assertGreaterEqual(sentiment_result.sentiment_score, 0)
        
        # Reset affection level
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 50
        self.session_manager.save_session(self.test_session_id)
        
        # Test negative sentiment
        negative_inputs = [
            "うるさい、バカ",
            "きもい、死ね",
            "Shut up, you're annoying",
            "どうでもいい、知らない"
        ]
        
        for user_input in negative_inputs:
            with self.subTest(input=user_input):
                initial_level = self.session_manager.get_affection_level(self.test_session_id)
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    self.test_session_id, user_input
                )
                
                self.assertLessEqual(new_level, initial_level)
                self.assertLessEqual(sentiment_result.affection_delta, 0)
                self.assertLess(sentiment_result.sentiment_score, 0)
    
    def test_gradual_affection_changes(self):
        """Test gradual application of large affection changes"""
        # Create input that should trigger large positive change
        very_positive = " ".join([
            "ありがとう", "すごい", "最高", "素晴らしい", "感謝", "助かる",
            "thank", "great", "awesome", "amazing", "wonderful", "appreciate"
        ])
        
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, very_positive
        )
        
        # Should have some immediate change
        self.assertGreater(new_level, initial_level)
        
        # Should have pending changes for gradual application
        if abs(sentiment_result.affection_delta) > 5:
            self.assertIn(self.test_session_id, self.affection_tracker.pending_affection_changes)
            pending_changes = self.affection_tracker.pending_affection_changes[self.test_session_id]
            self.assertGreater(len(pending_changes), 0)
    
    # ========== SECTION 3: Mari's Personality Consistency Tests ==========
    
    def test_core_personality_traits_consistency(self):
        """Test Mari's core personality traits remain consistent across all affection levels"""
        # Test all affection levels from 0 to 100
        test_levels = [0, 15, 30, 45, 60, 75, 90, 100]
        
        for affection_level in test_levels:
            with self.subTest(affection_level=affection_level):
                behavioral_state = self.affection_tracker.get_mari_behavioral_state(affection_level)
                
                # Core personality traits that must remain consistent
                self.assertEqual(
                    behavioral_state["core_personality"],
                    "警戒心が強い、不器用、ぶっきらぼうな男っぽい話し方"
                )
                self.assertEqual(behavioral_state["first_person"], "あたし")
                self.assertIn("〜だろ", behavioral_state["speech_patterns"])
                self.assertIn("〜じゃねーか", behavioral_state["speech_patterns"])
                self.assertIn("うっせー", behavioral_state["speech_patterns"])
                self.assertIn("バカかよ", behavioral_state["speech_patterns"])
    
    def test_prompt_personality_consistency(self):
        """Test personality consistency in generated prompts across affection levels"""
        # Core traits that must be preserved in all prompts
        core_traits = [
            "警戒心が強い",
            "不器用",
            "ぶっきらぼうな男っぽい話し方",
            "一人称は「あたし」",
            "「〜だろ」「〜じゃねーか」「うっせー」",
            "性的な話題や行動に対しては強い嫌悪"
        ]
        
        # Test prompts at different affection levels
        test_levels = [5, 20, 35, 55, 75, 95]
        
        for affection_level in test_levels:
            with self.subTest(affection_level=affection_level):
                prompt = self.prompt_generator.generate_dynamic_prompt(affection_level)
                
                # Verify all core traits are present
                for trait in core_traits:
                    self.assertIn(trait, prompt, 
                                f"Core trait '{trait}' missing in prompt for affection level {affection_level}")
                
                # Verify character consistency validation passes
                self.assertTrue(
                    self.prompt_generator.validate_character_consistency(prompt),
                    f"Character consistency validation failed for affection level {affection_level}"
                )
    
    def test_speech_patterns_consistency(self):
        """Test Mari's speech patterns remain consistent across relationship stages"""
        required_patterns = ["〜だろ", "〜じゃねーか", "うっせー", "バカかよ"]
        
        # Test all relationship stages
        stages = ["hostile", "distant", "cautious", "friendly", "warm", "close"]
        affection_levels = [10, 25, 40, 60, 80, 95]
        
        for stage, level in zip(stages, affection_levels):
            with self.subTest(stage=stage, level=level):
                behavioral_state = self.affection_tracker.get_mari_behavioral_state(level)
                
                # Verify speech patterns are preserved
                for pattern in required_patterns:
                    self.assertIn(pattern, behavioral_state["speech_patterns"],
                                f"Speech pattern '{pattern}' missing in {stage} stage")
    
    def test_character_constraints_consistency(self):
        """Test character constraints remain consistent across all affection levels"""
        # Test sexual topic rejection remains consistent
        test_levels = [0, 25, 50, 75, 100]
        
        for affection_level in test_levels:
            with self.subTest(affection_level=affection_level):
                prompt = self.prompt_generator.generate_dynamic_prompt(affection_level)
                
                # Sexual topic constraints must be present
                self.assertIn("性的な話題", prompt)
                self.assertIn("強い嫌悪", prompt)
                self.assertIn("性的描写や表現は禁止", prompt)
                
                # Character consistency constraints
                self.assertIn("キャラを崩さない", prompt)
                self.assertIn("一貫性を保つ", prompt)
    
    # ========== SECTION 4: Relationship Progression Tests ==========
    
    def test_natural_relationship_progression(self):
        """Test that relationship progression feels natural and gradual"""
        # Start with default affection level
        session = self.session_manager.get_session(self.test_session_id)
        initial_level = session.affection_level
        initial_stage = self.affection_tracker.get_relationship_stage(initial_level)
        
        # Simulate natural conversation progression
        conversation_flow = [
            ("こんにちは", "greeting"),
            ("あなたの名前は何ですか？", "question"),
            ("ありがとう、麻理さん", "appreciation"),
            ("あなたの話し方面白いですね", "compliment"),
            ("大丈夫？何か心配事がある？", "caring"),
            ("あなたの気持ち、わかりますよ", "empathy"),
            ("いつも話を聞いてくれてありがとう", "gratitude"),
            ("あなたは大切な存在です", "affection"),
            ("これからもずっと一緒にいたいです", "commitment")
        ]
        
        # Track progression
        affection_history = [initial_level]
        stage_history = [initial_stage]
        
        for user_input, interaction_type in conversation_flow:
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
            new_stage = self.affection_tracker.get_relationship_stage(new_level)
            
            affection_history.append(new_level)
            stage_history.append(new_stage)
        
        # Verify gradual increase (allowing for neutral interactions)
        for i in range(1, len(affection_history)):
            self.assertGreaterEqual(affection_history[i], affection_history[i-1],
                                  f"Affection decreased unexpectedly at step {i}")
        
        # Verify no stage skipping
        stage_order = ["hostile", "distant", "cautious", "friendly", "warm", "close"]
        unique_stages = []
        for stage in stage_history:
            if not unique_stages or stage != unique_stages[-1]:
                unique_stages.append(stage)
        
        # Check that stages progress in order
        for i in range(1, len(unique_stages)):
            current_index = stage_order.index(unique_stages[i])
            previous_index = stage_order.index(unique_stages[i-1])
            self.assertLessEqual(current_index - previous_index, 1,
                               f"Stage progression skipped from {unique_stages[i-1]} to {unique_stages[i]}")
    
    def test_relationship_stage_transitions(self):
        """Test relationship stage transitions at boundary points"""
        # Test exact boundary transitions
        boundary_tests = [
            (15, "hostile"), (16, "distant"),
            (30, "distant"), (31, "cautious"),
            (50, "cautious"), (51, "friendly"),
            (70, "friendly"), (71, "warm"),
            (85, "warm"), (86, "close")
        ]
        
        for affection_level, expected_stage in boundary_tests:
            with self.subTest(affection_level=affection_level):
                # Set affection level directly
                session = self.session_manager.get_session(self.test_session_id)
                session.affection_level = affection_level
                self.session_manager.save_session(self.test_session_id)
                
                # Get relationship stage
                stage = self.affection_tracker.get_relationship_stage(affection_level)
                self.assertEqual(stage, expected_stage)
    
    def test_relationship_stage_behavioral_changes(self):
        """Test behavioral changes between relationship stages"""
        # Test behavioral differences between stages
        stage_tests = [
            (10, "hostile", "皆無", "極めて低い"),
            (25, "distant", "ほぼない", "低い"),
            (40, "cautious", "わずか", "限定的"),
            (60, "friendly", "形成中", "中程度"),
            (80, "warm", "確立", "高い"),
            (95, "close", "深い", "非常に高い")
        ]
        
        for affection_level, expected_stage, expected_trust, expected_openness in stage_tests:
            with self.subTest(stage=expected_stage):
                behavioral_state = self.affection_tracker.get_mari_behavioral_state(affection_level)
                
                self.assertEqual(behavioral_state["stage"], expected_stage)
                self.assertEqual(behavioral_state["stage_traits"]["trust"], expected_trust)
                self.assertEqual(behavioral_state["stage_traits"]["openness"], expected_openness)
    
    def test_gradual_emotional_expression_changes(self):
        """Test gradual changes in emotional expression across stages"""
        # Test emotional expression progression
        expression_tests = [
            (10, "怒り、敵意"),
            (25, "冷淡、無関心"),
            (40, "控えめ、時折興味"),
            (60, "興味、時に喜び"),
            (80, "喜び、安心、時に不安"),
            (95, "愛着、依存、不安")
        ]
        
        for affection_level, expected_expression in expression_tests:
            with self.subTest(affection_level=affection_level):
                behavioral_state = self.affection_tracker.get_mari_behavioral_state(affection_level)
                actual_expression = behavioral_state["stage_traits"]["emotional_expression"]
                self.assertEqual(actual_expression, expected_expression)
    
    # ========== SECTION 5: Edge Cases and Boundary Conditions ==========
    
    def test_empty_and_invalid_inputs(self):
        """Test handling of empty and invalid user inputs"""
        # Test empty string
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, ""
        )
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(new_level, initial_level)
        self.assertEqual(sentiment_result.affection_delta, 0)
        self.assertEqual(sentiment_result.sentiment_score, 0.0)
        
        # Test whitespace only
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, "   \n\t  "
        )
        self.assertEqual(new_level, initial_level)
        self.assertEqual(sentiment_result.affection_delta, 0)
        
        # Test None input (should be handled gracefully)
        try:
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, None
            )
            # If it doesn't raise an exception, check it returns neutral result
            self.assertEqual(sentiment_result.affection_delta, 0)
        except (TypeError, AttributeError):
            # It's acceptable to raise an exception for None input
            pass
    
    def test_nonexistent_session_handling(self):
        """Test handling of operations on nonexistent sessions"""
        fake_session_id = "nonexistent-session-id"
        
        # Test affection update
        success = self.session_manager.update_affection(fake_session_id, 10)
        self.assertFalse(success)
        
        # Test affection level retrieval (should return default)
        level = self.session_manager.get_affection_level(fake_session_id)
        self.assertEqual(level, 15)  # Default level
        
        # Test conversation history update
        success = self.session_manager.update_conversation_history(
            fake_session_id, "test", "response"
        )
        self.assertFalse(success)
        
        # Test session retrieval
        session = self.session_manager.get_session(fake_session_id)
        self.assertIsNone(session)
    
    def test_extreme_affection_deltas(self):
        """Test handling of extreme affection changes"""
        # Set initial affection to middle value
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 50
        self.session_manager.save_session(self.test_session_id)
        
        # Test extremely positive input
        extremely_positive = " ".join([
            "ありがとう", "すごい", "最高", "素晴らしい", "感謝", "助かる",
            "thank", "great", "awesome", "amazing", "wonderful", "appreciate",
            "love", "perfect", "brilliant", "fantastic"
        ])
        
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, extremely_positive
        )
        
        # Should increase but stay within bounds
        self.assertGreater(new_level, 50)
        self.assertLessEqual(new_level, 100)
        self.assertLessEqual(abs(sentiment_result.affection_delta), 10)  # Delta should be bounded
        
        # Reset and test extremely negative input
        session.affection_level = 50
        self.session_manager.save_session(self.test_session_id)
        
        extremely_negative = " ".join([
            "うざい", "きもい", "バカ", "死ね", "黙れ", "hate", "stupid",
            "shut up", "annoying", "terrible", "disgusting", "awful"
        ])
        
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, extremely_negative
        )
        
        # Should decrease but stay within bounds
        self.assertLess(new_level, 50)
        self.assertGreaterEqual(new_level, 0)
        self.assertLessEqual(abs(sentiment_result.affection_delta), 10)  # Delta should be bounded
    
    def test_mixed_sentiment_handling(self):
        """Test handling of inputs with mixed positive and negative sentiment"""
        mixed_inputs = [
            "ありがとう、でもちょっとうざい",
            "Thank you, but you're annoying",
            "すごいけど、バカだね",
            "I love you but hate your attitude"
        ]
        
        for mixed_input in mixed_inputs:
            with self.subTest(input=mixed_input):
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    self.test_session_id, mixed_input
                )
                
                # Should detect multiple sentiment types
                self.assertGreater(len(sentiment_result.sentiment_types), 1)
                
                # Net effect should be reasonable (not extreme)
                self.assertLessEqual(abs(sentiment_result.affection_delta), 5)
    
    def test_corrupted_session_data_handling(self):
        """Test handling of corrupted session data"""
        # Create a session file with invalid JSON
        corrupted_session_id = "corrupted-session"
        file_path = os.path.join(self.test_dir, f"{corrupted_session_id}.json")
        
        with open(file_path, 'w') as f:
            f.write("This is not valid JSON {invalid}")
        
        # Try to load corrupted session
        session = self.session_manager.get_session(corrupted_session_id)
        self.assertIsNone(session)
        
        # Create session with missing required fields
        incomplete_data = {"user_id": "incomplete", "affection_level": 50}
        with open(file_path, 'w') as f:
            json.dump(incomplete_data, f)
        
        # Try to load incomplete session
        session = self.session_manager.get_session(corrupted_session_id)
        self.assertIsNone(session)
    
    def test_concurrent_session_operations(self):
        """Test handling of concurrent operations on the same session"""
        # Simulate concurrent affection updates
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        
        # Multiple updates in quick succession
        updates = [5, -3, 8, -2, 10]
        for delta in updates:
            success = self.session_manager.update_affection(self.test_session_id, delta)
            self.assertTrue(success)
        
        # Final level should be sum of all updates
        expected_level = max(0, min(100, initial_level + sum(updates)))
        final_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertEqual(final_level, expected_level)
    
    # ========== SECTION 6: Integration and System Tests ==========
    
    def test_complete_conversation_flow(self):
        """Test complete conversation flow with affection tracking and prompt generation"""
        # Simulate a complete conversation with multiple interactions
        conversation = [
            ("こんにちは", "greeting"),
            ("あなたの名前は何ですか？", "question"),
            ("ありがとう、麻理さん。よろしくお願いします", "polite_introduction"),
            ("あなたの話し方、面白いですね", "compliment"),
            ("大丈夫ですか？何か困ったことはありませんか？", "caring_question"),
            ("あなたの気持ち、少しわかる気がします", "empathy"),
            ("いつも話を聞いてくれて、本当にありがとう", "deep_gratitude"),
            ("あなたは私にとって大切な存在です", "affection_declaration")
        ]
        
        # Track conversation progression
        conversation_log = []
        
        for user_input, interaction_type in conversation:
            # Get current state before interaction
            pre_level = self.session_manager.get_affection_level(self.test_session_id)
            pre_stage = self.affection_tracker.get_relationship_stage(pre_level)
            
            # Process interaction
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
            new_stage = self.affection_tracker.get_relationship_stage(new_level)
            
            # Generate appropriate prompt
            dynamic_prompt = self.prompt_generator.generate_dynamic_prompt(new_level)
            
            # Update conversation history
            mock_response = f"Response at {new_stage} stage (affection: {new_level})"
            self.session_manager.update_conversation_history(
                self.test_session_id, user_input, mock_response
            )
            
            # Log interaction
            conversation_log.append({
                'input': user_input,
                'type': interaction_type,
                'pre_level': pre_level,
                'post_level': new_level,
                'pre_stage': pre_stage,
                'post_stage': new_stage,
                'sentiment_delta': sentiment_result.affection_delta,
                'prompt_contains_stage': self._check_stage_in_prompt(dynamic_prompt, new_stage)
            })
        
        # Verify conversation progression
        self.assertEqual(len(conversation_log), len(conversation))
        
        # Verify affection generally increased (allowing for some neutral interactions)
        final_level = conversation_log[-1]['post_level']
        initial_level = conversation_log[0]['pre_level']
        self.assertGreaterEqual(final_level, initial_level)
        
        # Verify conversation history was recorded
        session = self.session_manager.get_session(self.test_session_id)
        self.assertEqual(len(session.conversation_history), len(conversation))
        
        # Verify prompts were generated appropriately
        for log_entry in conversation_log:
            self.assertTrue(log_entry['prompt_contains_stage'],
                          f"Prompt didn't contain stage info for {log_entry['post_stage']}")
    
    def test_session_persistence_and_recovery(self):
        """Test session persistence across system restarts"""
        # Create conversation history and update affection
        test_interactions = [
            "こんにちは、麻理さん",
            "ありがとう、とても助かりました",
            "あなたのことをもっと知りたいです"
        ]
        
        for interaction in test_interactions:
            new_level, _ = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, interaction
            )
            self.session_manager.update_conversation_history(
                self.test_session_id, interaction, f"Response to: {interaction}"
            )
        
        # Get current state
        original_session = self.session_manager.get_session(self.test_session_id)
        original_affection = original_session.affection_level
        original_history_length = len(original_session.conversation_history)
        
        # Simulate system restart
        new_session_manager = SessionManager(self.test_dir)
        new_affection_tracker = AffectionTracker(new_session_manager)
        
        # Recover session
        recovered_session = new_session_manager.get_session(self.test_session_id)
        
        # Verify recovery
        self.assertIsNotNone(recovered_session)
        self.assertEqual(recovered_session.affection_level, original_affection)
        self.assertEqual(len(recovered_session.conversation_history), original_history_length)
        
        # Verify system can continue operating
        new_level, _ = new_affection_tracker.update_affection_for_interaction(
            self.test_session_id, "また会えて嬉しいです"
        )
        self.assertGreaterEqual(new_level, original_affection)
    
    def test_system_performance_with_multiple_sessions(self):
        """Test system performance and correctness with multiple concurrent sessions"""
        # Create multiple sessions
        session_ids = []
        for i in range(10):
            session_id = self.session_manager.create_new_session()
            session_ids.append(session_id)
        
        # Process interactions for each session
        test_inputs = [
            "こんにちは",
            "ありがとう",
            "すごいですね",
            "大丈夫ですか？",
            "感謝しています"
        ]
        
        # Track results for each session
        session_results = {}
        
        for session_id in session_ids:
            session_results[session_id] = []
            for user_input in test_inputs:
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, user_input
                )
                session_results[session_id].append(new_level)
        
        # Verify each session progressed independently
        for session_id, levels in session_results.items():
            # Each session should show progression
            self.assertGreaterEqual(levels[-1], levels[0])
            
            # Each session should have consistent progression
            for i in range(1, len(levels)):
                self.assertGreaterEqual(levels[i], levels[i-1])
        
        # Verify sessions don't interfere with each other
        unique_final_levels = set(results[-1] for results in session_results.values())
        # All sessions had same inputs, so should have similar final levels
        # (allowing for some variation due to gradual changes)
        self.assertLessEqual(len(unique_final_levels), 3)  # Should be very similar

if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)