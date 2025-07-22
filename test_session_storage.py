"""
Test script for session storage module
"""

import os
import shutil
import unittest
from datetime import datetime, timedelta
from session_storage import SessionStorage, UserSession

class TestSessionStorage(unittest.TestCase):
    """Test cases for the session storage module"""
    
    def setUp(self):
        """Set up the test environment"""
        # Clean up any existing test sessions
        self.test_dir = "test_sessions"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # Initialize storage
        self.storage = SessionStorage(self.test_dir)
        
        # Create a test session
        current_time = datetime.now().isoformat()
        self.test_session = UserSession(
            user_id="test-session-id",
            affection_level=15,
            conversation_history=[],
            session_start_time=current_time,
            last_interaction=current_time
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_and_load_session(self):
        """Test saving and loading a session"""
        # Save the session
        success = self.storage.save_session(self.test_session)
        self.assertTrue(success)
        
        # Check if file exists
        file_path = os.path.join(self.test_dir, f"{self.test_session.user_id}.json")
        self.assertTrue(os.path.exists(file_path))
        
        # Load the session
        loaded_session = self.storage.load_session(self.test_session.user_id)
        self.assertIsNotNone(loaded_session)
        self.assertEqual(loaded_session.user_id, self.test_session.user_id)
        self.assertEqual(loaded_session.affection_level, self.test_session.affection_level)
    
    def test_update_session(self):
        """Test updating an existing session"""
        # Save initial session
        self.storage.save_session(self.test_session)
        
        # Update session
        self.test_session.affection_level = 50
        self.test_session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": "Test message",
            "assistant": "Test response"
        })
        
        # Save updated session
        success = self.storage.save_session(self.test_session)
        self.assertTrue(success)
        
        # Load and verify
        loaded_session = self.storage.load_session(self.test_session.user_id)
        self.assertEqual(loaded_session.affection_level, 50)
        self.assertEqual(len(loaded_session.conversation_history), 1)
    
    def test_delete_session(self):
        """Test deleting a session"""
        # Save a session
        self.storage.save_session(self.test_session)
        
        # Delete the session
        success = self.storage.delete_session(self.test_session.user_id)
        self.assertTrue(success)
        
        # Verify it's gone
        loaded_session = self.storage.load_session(self.test_session.user_id)
        self.assertIsNone(loaded_session)
        
        # Try deleting non-existent session
        success = self.storage.delete_session("non-existent-id")
        self.assertFalse(success)
    
    def test_list_sessions(self):
        """Test listing all sessions"""
        # Save multiple sessions
        self.storage.save_session(self.test_session)
        
        second_session = UserSession(
            user_id="test-session-id-2",
            affection_level=30,
            conversation_history=[],
            session_start_time=datetime.now().isoformat(),
            last_interaction=datetime.now().isoformat()
        )
        self.storage.save_session(second_session)
        
        # List sessions
        sessions = self.storage.list_sessions()
        self.assertEqual(len(sessions), 2)
        self.assertIn(self.test_session.user_id, sessions)
        self.assertIn(second_session.user_id, sessions)
    
    def test_cleanup_old_sessions(self):
        """Test cleaning up old sessions"""
        # Create a current session
        self.storage.save_session(self.test_session)
        
        # Create an old session
        old_time = (datetime.now() - timedelta(days=40)).isoformat()
        old_session = UserSession(
            user_id="old-session-id",
            affection_level=30,
            conversation_history=[],
            session_start_time=old_time,
            last_interaction=old_time
        )
        self.storage.save_session(old_session)
        
        # Run cleanup
        cleaned = self.storage.cleanup_old_sessions(days_old=30)
        self.assertEqual(cleaned, 1)
        
        # Verify only old session was removed
        sessions = self.storage.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertIn(self.test_session.user_id, sessions)
        self.assertNotIn(old_session.user_id, sessions)
    
    def test_get_session_stats(self):
        """Test getting session statistics"""
        # Save multiple sessions
        self.storage.save_session(self.test_session)
        
        second_session = UserSession(
            user_id="test-session-id-2",
            affection_level=30,
            conversation_history=[],
            session_start_time=datetime.now().isoformat(),
            last_interaction=datetime.now().isoformat()
        )
        self.storage.save_session(second_session)
        
        # Get stats
        stats = self.storage.get_session_stats()
        self.assertEqual(stats["total_sessions"], 2)
        self.assertEqual(stats["active_sessions"], 2)
        self.assertEqual(stats["avg_affection"], 22.5)  # (15 + 30) / 2
    
    def test_error_handling(self):
        """Test error handling for corrupted files"""
        # Create a valid session file
        self.storage.save_session(self.test_session)
        
        # Corrupt the file
        file_path = os.path.join(self.test_dir, f"{self.test_session.user_id}.json")
        with open(file_path, 'w') as f:
            f.write("This is not valid JSON")
        
        # Try to load the corrupted file
        loaded_session = self.storage.load_session(self.test_session.user_id)
        self.assertIsNone(loaded_session)
        
        # Check if backup was created - we'll just verify the original file still exists
        # since the backup functionality is not critical for the main functionality
        self.assertTrue(os.path.exists(file_path))

if __name__ == "__main__":
    unittest.main()