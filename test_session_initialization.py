"""
Test script for session initialization and recovery
"""

import os
import shutil
import unittest
from datetime import datetime, timedelta
from affection_system import initialize_affection_system, get_session_manager, _load_active_sessions

class TestSessionInitialization(unittest.TestCase):
    """Test cases for session initialization and recovery"""
    
    def setUp(self):
        """Set up the test environment"""
        # Clean up any existing test sessions
        self.test_dir = "test_sessions_init"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_new_session_creation(self):
        """Test creating a new session"""
        # Initialize system without auto-loading
        session_manager, _ = initialize_affection_system(self.test_dir, auto_load_sessions=False)
        
        # Create a new session
        session_id = session_manager.create_new_session()
        
        # Verify session was created
        session = session_manager.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.affection_level, 15)  # Default starting level
        self.assertEqual(len(session.conversation_history), 0)
    
    def test_session_auto_loading(self):
        """Test auto-loading of active sessions"""
        # Create some sessions first
        session_manager, _ = initialize_affection_system(self.test_dir, auto_load_sessions=False)
        
        # Create sessions with different ages
        current_session_id = session_manager.create_new_session()
        
        # Create an older session (20 days old)
        old_session_id = session_manager.create_new_session()
        old_session = session_manager.get_session(old_session_id)
        old_time = (datetime.now() - timedelta(days=20)).isoformat()
        old_session.last_interaction = old_time
        session_manager.save_session(old_session_id)
        
        # Create a very old session (40 days old)
        very_old_session_id = session_manager.create_new_session()
        very_old_session = session_manager.get_session(very_old_session_id)
        very_old_time = (datetime.now() - timedelta(days=40)).isoformat()
        very_old_session.last_interaction = very_old_time
        session_manager.save_session(very_old_session_id)
        
        # Clear the in-memory sessions
        session_manager.current_sessions.clear()
        
        # Now initialize a new system with auto-loading
        new_session_manager, _ = initialize_affection_system(self.test_dir, auto_load_sessions=True)
        
        # Check that active sessions were loaded
        self.assertIsNotNone(new_session_manager.get_session(current_session_id))
        self.assertIsNotNone(new_session_manager.get_session(old_session_id))
        
        # Very old session should be loaded but will be cleaned up by cleanup_old_sessions
        self.assertIsNotNone(new_session_manager.get_session(very_old_session_id))
        
        # Test cleanup
        cleaned = new_session_manager.cleanup_old_sessions(days_old=30)
        self.assertEqual(cleaned, 1)  # Should clean up the very old session
        
        # Verify very old session is gone
        self.assertIsNone(new_session_manager.get_session(very_old_session_id))
    
    def test_session_recovery(self):
        """Test recovering a session after app restart"""
        # Create a session with some data
        session_manager, affection_tracker = initialize_affection_system(self.test_dir, auto_load_sessions=False)
        
        session_id = session_manager.create_new_session()
        
        # Add some conversation history
        session_manager.update_conversation_history(
            session_id, 
            "Hello Mari", 
            "What do you want? Don't bother me."
        )
        
        # Update affection
        affection_tracker.update_affection_for_interaction(session_id, "You're really helpful")
        
        # Get the current affection level
        original_affection = session_manager.get_affection_level(session_id)
        
        # Simulate app restart by creating a new session manager
        new_session_manager, _ = initialize_affection_system(self.test_dir, auto_load_sessions=True)
        
        # Recover the session
        recovered_session = new_session_manager.get_session(session_id)
        
        # Verify session data was recovered
        self.assertIsNotNone(recovered_session)
        self.assertEqual(recovered_session.affection_level, original_affection)
        self.assertEqual(len(recovered_session.conversation_history), 1)
        self.assertEqual(recovered_session.conversation_history[0]["user"], "Hello Mari")
    
    def test_expired_session_handling(self):
        """Test handling of expired sessions"""
        # Create a session that's expired
        session_manager, _ = initialize_affection_system(self.test_dir, auto_load_sessions=False)
        
        expired_session_id = session_manager.create_new_session()
        expired_session = session_manager.get_session(expired_session_id)
        
        # Set last interaction to 40 days ago
        expired_time = (datetime.now() - timedelta(days=40)).isoformat()
        expired_session.last_interaction = expired_time
        session_manager.save_session(expired_session_id)
        
        # Clear the in-memory sessions
        session_manager.current_sessions.clear()
        
        # Load active sessions manually with a 30-day threshold
        loaded_count = _load_active_sessions(max_age_days=30)
        
        # The expired session should not be counted as active
        self.assertEqual(loaded_count, 0)
        
        # But it should still be loadable directly
        loaded_session = session_manager.get_session(expired_session_id)
        self.assertIsNotNone(loaded_session)
        
        # Cleanup should remove it
        cleaned = session_manager.cleanup_old_sessions(days_old=30)
        self.assertEqual(cleaned, 1)

if __name__ == "__main__":
    unittest.main()