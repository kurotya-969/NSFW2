"""
Comprehensive Tests for Mari AI Chat Affection System
Tests edge cases and boundary conditions for the affection system
Requirements: 1.1, 1.2, 1.3, 2.3, 3.4
"""

import os
import shutil
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from affection_system import (
    SessionManager, AffectionTracker, initialize_affection_system,
    get_session_manager, get_affection_tracker
)
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult, SentimentType
from session_storage import UserSession, SessionStorage
from prompt_generator import PromptGenerator

class TestAffectionSystemEdgeCases(unittest.TestCase):
    """Test cases for edge cases and boundary conditions in the affection system"""
    
    def setUp(self):
        """Set up the test environment"""
        # Clean up any existing test sessions
        self.test_dir = "test_affection_edge_cases"
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
    
    # SECTION 4: EDGE CASES AND BOUNDARY TESTS
    
    def test_affection_boundary_conditions(self):
        """Test affection level boundary conditions (0-100)"""
        # Test lower boundary
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 5
        self.session_manager.save_session(self.test_session_id)
        
        # Apply very negative interaction
        self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, "死ね、黙れ、消えろ、hate you, stupid, shut up"
        )
        
        # Verify affection doesn't go below 0
        self.assertGreaterEqual(self.session_manager.get_affection_level(self.test_session_id), 0)
        
        # Test upper boundary
        session.affection_level = 95
        self.session_manager.save_session(self.test_session_id)
        
        # Apply very positive interaction
        self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, "あなたは私の人生で最高の出会いだよ。本当にありがとう。あなたがいてくれて本当に幸せ。"
        )
        
        # Process all pending changes
        while self.affection_tracker.pending_affection_changes.get(self.test_session_id, []):
            self.affection_tracker._process_pending_affection_changes(self.test_session_id)
        
        # Verify affection doesn't go above 100
        self.assertLessEqual(self.session_manager.get_affection_level(self.test_session_id), 100)
    
    def test_empty_and_neutral_inputs(self):
        """Test handling of empty and neutral inputs"""
        # Set initial affection
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 50
        initial_level = session.affection_level
        self.session_manager.save_session(self.test_session_id)
        
        # Test empty input
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, ""
        )
        self.assertEqual(new_level, initial_level)
        self.assertEqual(sentiment_result.affection_delta, 0)
        
        # Test neutral input
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, "今日は晴れています"
        )
        self.assertEqual(new_level, initial_level)
        self.assertAlmostEqual(sentiment_result.affection_delta, 0, delta=1)
    
    def test_mixed_sentiment_inputs(self):
        """Test handling of inputs with mixed sentiment"""
        # Set initial affection
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 50
        initial_level = session.affection_level
        self.session_manager.save_session(self.test_session_id)
        
        # Test mixed positive and negative input
        mixed_input = "ありがとう、でもちょっとうざい"
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, mixed_input
        )
        
        # Verify both sentiment types were detected
        self.assertIn(SentimentType.POSITIVE, sentiment_result.sentiment_types)
        self.assertIn(SentimentType.NEGATIVE, sentiment_result.sentiment_types)
        
        # The net effect should be close to neutral
        self.assertAlmostEqual(new_level, initial_level, delta=3)
    
    def test_invalid_session_handling(self):
        """Test handling of invalid or non-existent sessions"""
        # Try to update affection for non-existent session
        success = self.session_manager.update_affection("non-existent-id", 5)
        self.assertFalse(success)
        
        # Try to get affection level for non-existent session
        level = self.session_manager.get_affection_level("non-existent-id")
        self.assertEqual(level, 15)  # Should return default value
        
        # Try to update conversation history for non-existent session
        success = self.session_manager.update_conversation_history(
            "non-existent-id", "Hello", "Response"
        )
        self.assertFalse(success)
    
    def test_session_persistence_recovery(self):
        """Test session persistence and recovery after system restart"""
        # Create a session with some data
        session_id = self.session_manager.create_new_session()
        
        # Update affection and conversation history
        self.session_manager.update_affection(session_id, 20)
        self.session_manager.update_conversation_history(
            session_id, "Hello Mari", "What do you want? Don't bother me."
        )
        
        # Get the current affection level
        original_affection = self.session_manager.get_affection_level(session_id)
        
        # Simulate system restart by creating new instances
        new_session_manager = SessionManager(self.test_dir)
        
        # Load the session
        loaded_session = new_session_manager.get_session(session_id)
        
        # Verify session data was recovered
        self.assertIsNotNone(loaded_session)
        self.assertEqual(loaded_session.affection_level, original_affection)
        self.assertEqual(len(loaded_session.conversation_history), 1)
        self.assertEqual(loaded_session.conversation_history[0]["user"], "Hello Mari")
    
    def test_corrupted_session_handling(self):
        """Test handling of corrupted session files"""
        # Create a session
        session_id = self.session_manager.create_new_session()
        
        # Corrupt the session file
        file_path = os.path.join(self.test_dir, f"{session_id}.json")
        with open(file_path, 'w') as f:
            f.write("This is not valid JSON")
        
        # Try to load the corrupted session
        loaded_session = self.session_manager.get_session(session_id)
        
        # Should return None for corrupted session
        self.assertIsNone(loaded_session)
        
        # Verify we can create a new session with the same ID
        new_session_id = self.session_manager.create_new_session(session_id)
        self.assertEqual(new_session_id, session_id)
        
        # Verify the new session is valid
        new_session = self.session_manager.get_session(session_id)
        self.assertIsNotNone(new_session)
        self.assertEqual(new_session.affection_level, 15)  # Default starting level
    
    def test_session_cleanup(self):
        """Test session cleanup functionality"""
        # Create multiple sessions with different timestamps
        current_time = datetime.now()
        
        # Create a recent session
        recent_session_id = self.session_manager.create_new_session()
        recent_session = self.session_manager.get_session(recent_session_id)
        recent_session.last_interaction = current_time.isoformat()
        self.session_manager.save_session(recent_session_id)
        
        # Create an old session (31 days old)
        old_session_id = self.session_manager.create_new_session()
        old_session = self.session_manager.get_session(old_session_id)
        old_time = current_time - timedelta(days=31)
        old_session.last_interaction = old_time.isoformat()
        self.session_manager.save_session(old_session_id)
        
        # Run cleanup (30 days threshold)
        cleaned_count = self.session_manager.cleanup_old_sessions(days_old=30)
        
        # Verify only old session was cleaned up
        self.assertEqual(cleaned_count, 1)
        self.assertIsNotNone(self.session_manager.get_session(recent_session_id))
        self.assertIsNone(self.session_manager.get_session(old_session_id))
    
    def test_sentiment_history_tracking(self):
        """Test sentiment history tracking functionality"""
        # Apply multiple interactions
        interactions = [
            "こんにちは",
            "ありがとう",
            "うるさい",
            "あなたの話し方が好きだよ"
        ]
        
        # Process each interaction
        for user_input in interactions:
            self.affection_tracker.update_affection_for_interaction(
                self.test_session_id, user_input
            )
        
        # Get sentiment history
        history = self.affection_tracker.get_sentiment_history(self.test_session_id)
        
        # Verify history was tracked
        self.assertEqual(len(history), len(interactions))
        
        # Verify history contains expected fields
        for entry in history:
            self.assertIn("timestamp", entry)
            self.assertIn("user_input", entry)
            self.assertIn("sentiment_score", entry)
            self.assertIn("interaction_type", entry)
            self.assertIn("affection_delta", entry)
            self.assertIn("detected_keywords", entry)
    
    def test_gradual_affection_changes(self):
        """Test that large affection changes are applied gradually"""
        # Start with a middle affection level
        session = self.session_manager.get_session(self.test_session_id)
        session.affection_level = 50  # Start at cautious
        self.session_manager.save_session(self.test_session_id)
        
        # Apply a very positive interaction that would cause a large affection increase
        very_positive = "あなたは私の人生で最高の出会いだよ。本当にありがとう。あなたがいてくれて本当に幸せ。"
        
        # Get initial affection level
        initial_level = self.session_manager.get_affection_level(self.test_session_id)
        
        # Update affection
        new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
            self.test_session_id, very_positive
        )
        
        # Verify immediate change is not too large
        self.assertLessEqual(new_level - initial_level, 10,
                           "Immediate affection change was too large")
        
        # Verify there are pending gradual changes
        self.assertGreater(len(self.affection_tracker.pending_affection_changes.get(self.test_session_id, [])), 0,
                         "No pending gradual affection changes were scheduled")
        
        # Process pending changes
        self.affection_tracker._process_pending_affection_changes(self.test_session_id)
        
        # Verify final level is higher but still not the full change
        final_level = self.session_manager.get_affection_level(self.test_session_id)
        self.assertGreater(final_level, new_level,
                         "Affection didn't increase after processing pending changes")

if __name__ == "__main__":
    unittest.main()
"""