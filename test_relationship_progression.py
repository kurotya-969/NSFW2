"""
Test script for relationship progression in Mari AI Chat
Tests that relationship progression feels natural and gradual
Requirements: 2.3, 3.4
"""

import os
import shutil
import unittest
from datetime import datetime
from affection_system import SessionManager, AffectionTracker

class TestRelationshipProgression(unittest.TestCase):
    """Test cases for relationship progression"""
    
    def setUp(self):
        """Set up the test environment"""
        # Clean up any existing test sessions
        self.test_dir = "test_relationship_progression"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Initialize components
        self.session_manager = SessionManager(self.test_dir)
        self.affection_tracker = AffectionTracker(self.session_manager)
        
        # Create a test session
        self.test_session_id = self.session_manager.create_new_session()
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_gradual_relationship_progression(self):
        """Test that relationship progression feels natural and gradual"""
        # Start with default affection level
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 15  # Start at hostile
        self.session_manager.save_session(self.test_session_id)
        
        # Define a series of interactions that should gradually increase affection
        interactions = [
            # Simple neutral interactions
            "こんにちは",
            "今日の天気はどう？",
            "何してるの？",
            
            # Slightly positive interactions
            "ありがとう",
            "それは良かった",
            "面白いね",
            
            # More positive interactions
            "あなたの話し方が好きだよ",
            "あなたと話すの楽しい",
            "いつも助けてくれてありがとう",
            
            # Caring interactions
            "大丈夫？心配してるよ",
            "無理しないでね",
            "あなたの気持ち、わかるよ",
            
            # Appreciative interactions
            "あなたのおかげで助かったよ",
            "いつも話を聞いてくれてありがとう",
            "あなたは本当に頼りになるね",
            
            # Deep connection interactions
            "あなたは大切な存在だよ",
            "あなたといると安心する",
            "これからもずっと一緒にいようね",
            
            # Very strong positive interactions
            "あなたは私の人生で最高の出会いだよ",
            "あなたがいてくれて本当に幸せ"
        ]
        
        # Track affection levels and stages
        affection_levels = [session.affection_level]
        stages = [self.affection_tracker.get_relationship_stage(session.affection_level)]
        
        # Process each interaction
        for user_input in interactions:
            new_level, _ = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
            new_stage = self.affection_tracker.get_relationship_stage(new_level)
            
            affection_levels.append(new_level)
            stages.append(new_stage)
            
            # Update conversation history
            self.session_manager.update_conversation_history(
                self.test_session_id,
                user_input,
                f"Response at {new_stage} stage with affection {new_level}"
            )
        
        # Verify affection increases gradually
        for i in range(1, len(affection_levels)):
            self.assertGreaterEqual(affection_levels[i], affection_levels[i-1],
                                  f"Affection decreased from {affection_levels[i-1]} to {affection_levels[i]} after input: {interactions[i-1]}")
        
        # Force affection to higher levels to test all stages
        session = self.session_manager.get_session(self.test_session_id)
        
        # Test friendly stage
        session.affection_level = 60
        self.session_manager.save_session(self.test_session_id)
        friendly_stage = self.affection_tracker.get_relationship_stage(60)
        self.assertEqual(friendly_stage, "friendly")
        
        # Test warm stage
        session.affection_level = 80
        self.session_manager.save_session(self.test_session_id)
        warm_stage = self.affection_tracker.get_relationship_stage(80)
        self.assertEqual(warm_stage, "warm")
        
        # Test close stage
        session.affection_level = 95
        self.session_manager.save_session(self.test_session_id)
        close_stage = self.affection_tracker.get_relationship_stage(95)
        self.assertEqual(close_stage, "close")
        
        # Verify stage transitions are smooth (no skipping stages)
        stage_order = ["hostile", "distant", "cautious", "friendly", "warm", "close"]
        stage_transitions = []
        
        current_stage = stages[0]
        for i in range(1, len(stages)):
            if stages[i] != current_stage:
                stage_transitions.append((current_stage, stages[i]))
                current_stage = stages[i]
        
        for old_stage, new_stage in stage_transitions:
            old_index = stage_order.index(old_stage)
            new_index = stage_order.index(new_stage)
            self.assertEqual(new_index, old_index + 1, 
                           f"Stage transition from {old_stage} to {new_stage} skipped intermediate stages")
    
    def test_relationship_regression(self):
        """Test that relationship can regress with negative interactions"""
        # Start with a high affection level
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 80  # Start at warm
        self.session_manager.save_session(self.test_session_id)
        
        initial_stage = self.affection_tracker.get_relationship_stage(session.affection_level)
        self.assertEqual(initial_stage, "warm")
        
        # Define a series of increasingly negative interactions
        negative_interactions = [
            "うるさいな",
            "めんどくさい",
            "知らないよ",
            "うざい",
            "バカじゃないの？",
            "黙れ",
            "消えろ"
        ]
        
        # Track affection levels and stages
        affection_levels = [session.affection_level]
        stages = [initial_stage]
        
        # Process each interaction
        for user_input in negative_interactions:
            new_level, _ = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
            new_stage = self.affection_tracker.get_relationship_stage(new_level)
            
            affection_levels.append(new_level)
            stages.append(new_stage)
        
        # Verify affection decreases gradually
        for i in range(1, len(affection_levels)):
            self.assertLessEqual(affection_levels[i], affection_levels[i-1],
                               f"Affection increased from {affection_levels[i-1]} to {affection_levels[i]} after negative input: {negative_interactions[i-1]}")
        
        # Force affection to lower levels to test regression through all stages
        session = self.session_manager.get_session(self.test_session_id)
        
        # Test friendly stage regression
        session.affection_level = 60
        self.session_manager.save_session(self.test_session_id)
        friendly_stage = self.affection_tracker.get_relationship_stage(60)
        self.assertEqual(friendly_stage, "friendly")
        
        # Test cautious stage regression
        session.affection_level = 40
        self.session_manager.save_session(self.test_session_id)
        cautious_stage = self.affection_tracker.get_relationship_stage(40)
        self.assertEqual(cautious_stage, "cautious")
        
        # Test significant affection decrease with very negative inputs
        session.affection_level = 80  # Reset to warm
        self.session_manager.save_session(self.test_session_id)
        
        # Apply extremely negative inputs
        extremely_negative = [
            "うざい、きもい、バカ",
            "死ね、黙れ、消えろ",
            "hate you, stupid, shut up"
        ]
        
        for input_text in extremely_negative:
            self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, input_text
            )
        
        # Verify significant decrease
        final_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertLess(final_level, 60, "Affection didn't decrease significantly after multiple negative interactions")
        
        # Verify final affection is significantly lower than initial
        self.assertLess(affection_levels[-1], affection_levels[0] - 20,
                      "Affection didn't decrease significantly after multiple negative interactions")
    
    def test_mixed_interaction_pattern(self):
        """Test relationship changes with mixed positive and negative interactions"""
        # Start with a middle affection level
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 50  # Start at cautious
        self.session_manager.save_session(self.test_session_id)
        
        # Define a series of alternating positive and negative interactions
        mixed_interactions = [
            ("ありがとう", "positive"),
            ("うるさいな", "negative"),
            ("あなたの話し方が好きだよ", "positive"),
            ("めんどくさい", "negative"),
            ("大丈夫？心配してるよ", "positive"),
            ("知らないよ", "negative"),
            ("いつも助けてくれてありがとう", "positive"),
            ("うざい", "negative"),
            ("あなたは大切な存在だよ", "positive")
        ]
        
        # Track affection levels
        affection_levels = [session.affection_level]
        
        # Process each interaction
        for user_input, interaction_type in mixed_interactions:
            new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
            affection_levels.append(new_level)
            
            # Verify sentiment matches expected type
            if interaction_type == "positive":
                self.assertGreaterEqual(sentiment_result.affection_delta, 0,
                                      f"Expected positive sentiment for '{user_input}' but got delta {sentiment_result.affection_delta}")
            else:
                self.assertLessEqual(sentiment_result.affection_delta, 0,
                                   f"Expected negative sentiment for '{user_input}' but got delta {sentiment_result.affection_delta}")
        
        # Verify affection changes match interaction pattern
        for i in range(1, len(affection_levels)):
            interaction_type = mixed_interactions[i-1][1]
            if interaction_type == "positive":
                self.assertGreaterEqual(affection_levels[i], affection_levels[i-1],
                                      f"Affection didn't increase after positive input: {mixed_interactions[i-1][0]}")
            else:
                self.assertLessEqual(affection_levels[i], affection_levels[i-1],
                                   f"Affection didn't decrease after negative input: {mixed_interactions[i-1][0]}")
    
    def test_relationship_recovery(self):
        """Test relationship can recover after negative interactions"""
        # Start with a middle affection level
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 50  # Start at cautious
        initial_level = session.affection_level
        self.session_manager.save_session(self.test_session_id)
        
        # First apply negative interactions
        negative_interactions = [
            "うるさいな",
            "めんどくさい",
            "知らないよ",
            "うざい"
        ]
        
        # Process negative interactions
        for user_input in negative_interactions:
            new_level, _ = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
        
        # Verify affection decreased
        mid_level = new_level
        self.assertLess(mid_level, initial_level)
        
        # Now apply positive interactions to recover
        positive_interactions = [
            "ごめんなさい、さっきは言い過ぎた",
            "あなたの話し方が好きだよ",
            "大丈夫？心配してるよ",
            "いつも助けてくれてありがとう",
            "あなたは大切な存在だよ"
        ]
        
        # Process positive interactions
        for user_input in positive_interactions:
            new_level, _ = self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
        
        # Verify affection recovered
        final_level = new_level
        self.assertGreater(final_level, mid_level)
        self.assertGreaterEqual(final_level, initial_level,
                              "Affection didn't recover to at least initial level after positive interactions")

if __name__ == "__main__":
    unittest.main()