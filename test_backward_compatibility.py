"""
Backward Compatibility Testing for Enhanced Sentiment Analysis System
Focuses specifically on ensuring backward compatibility with existing sessions
and verifying that the enhanced system works correctly with previously stored data.
Task 8.3: Perform regression testing
Requirements: 5.3
"""

import os
import shutil
import unittest
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from affection_system import (
    SessionManager, AffectionTracker, initialize_affection_system,
    get_session_manager, get_affection_tracker
)
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult, SentimentType
from enhanced_sentiment_adapter import EnhancedSentimentAdapter
from session_storage import UserSession, SessionStorage

class TestBackwardCompatibility(unittest.TestCase):
    """Test suite for backward compatibility with existing sessions"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp(prefix="test_backward_compat_")
        
        # Create a directory for "old" sessions
        self.old_sessions_dir = os.path.join(self.test_dir, "old_sessions")
        os.makedirs(self.old_sessions_dir)
        
        # Create "old" sessions with the original sentiment analyzer
        self.create_old_sessions()
        
        # Initialize the affection system with the old sessions directory
        self.session_manager, self.affection_tracker = initialize_affection_system(
            self.old_sessions_dir, auto_load_sessions=True
        )
        
        # Initialize enhanced sentiment adapter
        self.enhanced_adapter = EnhancedSentimentAdapter()
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_old_sessions(self):
        """Create "old" sessions with the original sentiment analyzer"""
        # Create temporary session manager for creating old sessions
        temp_session_manager = SessionManager(self.old_sessions_dir)
        temp_affection_tracker = AffectionTracker(temp_session_manager)
        
        # Create sessions with different affection levels and conversation histories
        self.old_session_ids = []
        
        # Session 1: Low affection with negative history
        session_id = temp_session_manager.create_new_session()
        self.old_session_ids.append(session_id)
        
        # Set low affection level
        session = temp_session_manager.get_session(session_id)
        session.affection_level = 10
        temp_session_manager.save_session(session_id)
        
        # Add negative conversation history
        negative_conversations = [
            ("うるさいな", "うっせーよ、何だよ"),
            ("バカ", "お前こそバカだろ"),
            ("嫌い", "知るかよ")
        ]
        
        for user_input, assistant_response in negative_conversations:
            temp_session_manager.update_conversation_history(
                session_id, user_input, assistant_response
            )
        
        # Session 2: Medium affection with mixed history
        session_id = temp_session_manager.create_new_session()
        self.old_session_ids.append(session_id)
        
        # Set medium affection level
        session = temp_session_manager.get_session(session_id)
        session.affection_level = 50
        temp_session_manager.save_session(session_id)
        
        # Add mixed conversation history
        mixed_conversations = [
            ("こんにちは", "よう、何か用？"),
            ("元気？", "まあまあだよ"),
            ("ありがとう", "別に…いいけど")
        ]
        
        for user_input, assistant_response in mixed_conversations:
            temp_session_manager.update_conversation_history(
                session_id, user_input, assistant_response
            )
        
        # Session 3: High affection with positive history
        session_id = temp_session_manager.create_new_session()
        self.old_session_ids.append(session_id)
        
        # Set high affection level
        session = temp_session_manager.get_session(session_id)
        session.affection_level = 85
        temp_session_manager.save_session(session_id)
        
        # Add positive conversation history
        positive_conversations = [
            ("こんにちは", "あ、来たんだ。ちょっと待ってたんだよ"),
            ("元気？", "うん、あんたがいると元気になるよ"),
            ("ありがとう", "あたしこそ…ありがと")
        ]
        
        for user_input, assistant_response in positive_conversations:
            temp_session_manager.update_conversation_history(
                session_id, user_input, assistant_response
            )
    
    def test_load_existing_sessions(self):
        """Test loading existing sessions created with original analyzer"""
        # Verify all old sessions can be loaded
        for session_id in self.old_session_ids:
            session = self.session_manager.get_session(session_id)
            self.assertIsNotNone(session, f"Failed to load old session: {session_id}")
            
            # Verify session has expected data
            self.assertIsInstance(session.affection_level, int)
            self.assertGreaterEqual(session.affection_level, 0)
            self.assertLessEqual(session.affection_level, 100)
            self.assertGreaterEqual(len(session.conversation_history), 3)
    
    def test_update_existing_sessions_with_enhanced_analyzer(self):
        """Test updating existing sessions with enhanced analyzer"""
        for session_id in self.old_session_ids:
            # Get original session data
            original_session = self.session_manager.get_session(session_id)
            original_affection = original_session.affection_level
            original_history_length = len(original_session.conversation_history)
            
            # Process new messages with enhanced analyzer
            test_inputs = [
                "こんにちは",  # Neutral
                "ありがとう",  # Positive
                "素晴らしいですね、また失敗しましたか"  # Context-dependent
            ]
            
            with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
                for input_text in test_inputs:
                    new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                        session_id, input_text
                    )
                    
                    # Update conversation history
                    self.session_manager.update_conversation_history(
                        session_id, input_text, f"Response to: {input_text}"
                    )
            
            # Get updated session
            updated_session = self.session_manager.get_session(session_id)
            
            # Verify session was updated
            self.assertEqual(len(updated_session.conversation_history), 
                           original_history_length + len(test_inputs))
            
            # Verify affection level was updated
            self.assertNotEqual(updated_session.affection_level, original_affection, 
                              f"Affection level should change for session {session_id}")
    
    def test_sentiment_history_with_enhanced_analyzer(self):
        """Test sentiment history with enhanced analyzer"""
        for session_id in self.old_session_ids:
            # Process messages with enhanced analyzer
            test_inputs = [
                "ありがとう",  # Positive
                "うるさいな",  # Negative
                "心配だよ"     # Caring
            ]
            
            with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
                for input_text in test_inputs:
                    self.affection_tracker.update_affection_for_interaction(
                        session_id, input_text
                    )
            
            # Get sentiment history
            sentiment_history = self.affection_tracker.get_sentiment_history(session_id)
            
            # Verify sentiment history was recorded
            self.assertEqual(len(sentiment_history), len(test_inputs))
            
            # Verify sentiment history has expected format
            for i, entry in enumerate(sentiment_history):
                self.assertIn("timestamp", entry)
                self.assertIn("user_input", entry)
                self.assertIn("sentiment_score", entry)
                self.assertIn("interaction_type", entry)
                self.assertIn("affection_delta", entry)
                self.assertEqual(entry["user_input"], test_inputs[i])
    
    def test_relationship_progression_with_enhanced_analyzer(self):
        """Test relationship progression with enhanced analyzer"""
        # Focus on the low affection session
        session_id = self.old_session_ids[0]
        original_session = self.session_manager.get_session(session_id)
        original_affection = original_session.affection_level
        
        # Verify initial relationship stage
        initial_stage = self.affection_tracker.get_relationship_stage(original_affection)
        self.assertEqual(initial_stage, "hostile", "Initial stage should be hostile for low affection")
        
        # Process positive messages with enhanced analyzer to improve relationship
        positive_inputs = [
            "ありがとう",
            "あなたは素晴らしい人だね",
            "いつも元気をもらえるよ",
            "あなたのことが好きだよ",
            "あなたは大切な存在だよ"
        ] * 3  # Repeat to ensure significant affection increase
        
        with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
            for input_text in positive_inputs:
                new_level, _ = self.affection_tracker.update_affection_for_interaction(
                    session_id, input_text
                )
        
        # Get final affection level and relationship stage
        final_session = self.session_manager.get_session(session_id)
        final_affection = final_session.affection_level
        final_stage = self.affection_tracker.get_relationship_stage(final_affection)
        
        # Verify relationship progressed
        self.assertGreater(final_affection, original_affection, 
                         "Affection level should increase with positive inputs")
        self.assertNotEqual(final_stage, initial_stage, 
                          "Relationship stage should change with significant affection increase")
        
        # Verify behavioral state reflects new relationship stage
        behavioral_state = self.affection_tracker.get_mari_behavioral_state(final_affection)
        self.assertEqual(behavioral_state["stage"], final_stage)
    
    def test_save_and_reload_with_enhanced_analyzer(self):
        """Test saving and reloading sessions after using enhanced analyzer"""
        # Process messages with enhanced analyzer for all sessions
        with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
            for session_id in self.old_session_ids:
                self.affection_tracker.update_affection_for_interaction(
                    session_id, "ありがとう"
                )
                self.session_manager.update_conversation_history(
                    session_id, "ありがとう", "Response to: ありがとう"
                )
        
        # Save all sessions
        for session_id in self.old_session_ids:
            self.session_manager.save_session(session_id)
        
        # Clear in-memory sessions
        self.session_manager.current_sessions = {}
        
        # Reload sessions
        for session_id in self.old_session_ids:
            loaded_session = self.session_manager.get_session(session_id)
            
            # Verify session was loaded successfully
            self.assertIsNotNone(loaded_session)
            
            # Verify conversation history includes new entry
            last_entry = loaded_session.conversation_history[-1]
            self.assertEqual(last_entry["user"], "ありがとう")
            self.assertEqual(last_entry["assistant"], "Response to: ありがとう")
    
    def test_mixed_analyzer_usage(self):
        """Test using both original and enhanced analyzers with the same sessions"""
        for session_id in self.old_session_ids:
            # Get original affection level
            original_affection = self.session_manager.get_affection_level(session_id)
            
            # Process with original analyzer
            self.affection_tracker.update_affection_for_interaction(
                session_id, "ありがとう"
            )
            
            # Get intermediate affection level
            intermediate_affection = self.session_manager.get_affection_level(session_id)
            
            # Process with enhanced analyzer
            with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
                self.affection_tracker.update_affection_for_interaction(
                    session_id, "ありがとう"
                )
            
            # Get final affection level
            final_affection = self.session_manager.get_affection_level(session_id)
            
            # Verify both analyzers changed affection level
            self.assertNotEqual(original_affection, intermediate_affection, 
                              "Original analyzer should change affection level")
            self.assertNotEqual(intermediate_affection, final_affection, 
                              "Enhanced analyzer should change affection level")
            
            # Verify both changes were in the same direction (both positive for "ありがとう")
            self.assertGreater(intermediate_affection, original_affection, 
                             "Original analyzer should increase affection for positive input")
            self.assertGreater(final_affection, intermediate_affection, 
                             "Enhanced analyzer should increase affection for positive input")
    
    def test_toggle_between_analyzers(self):
        """Test toggling between original and enhanced analyzers"""
        # Create adapter with toggle capability
        adapter = EnhancedSentimentAdapter()
        
        for session_id in self.old_session_ids:
            # Get original affection level
            original_affection = self.session_manager.get_affection_level(session_id)
            
            # Process with enhanced analysis (default)
            with patch.object(self.affection_tracker, 'sentiment_analyzer', adapter):
                self.affection_tracker.update_affection_for_interaction(
                    session_id, "ありがとう"
                )
            
            # Get intermediate affection level
            intermediate_affection = self.session_manager.get_affection_level(session_id)
            
            # Toggle to original analysis
            adapter.toggle_enhanced_analysis(False)
            
            # Process with original analysis
            with patch.object(self.affection_tracker, 'sentiment_analyzer', adapter):
                self.affection_tracker.update_affection_for_interaction(
                    session_id, "ありがとう"
                )
            
            # Get final affection level
            final_affection = self.session_manager.get_affection_level(session_id)
            
            # Verify both modes changed affection level
            self.assertNotEqual(original_affection, intermediate_affection, 
                              "Enhanced mode should change affection level")
            self.assertNotEqual(intermediate_affection, final_affection, 
                              "Original mode should change affection level")
            
            # Toggle back to enhanced analysis
            adapter.toggle_enhanced_analysis(True)

if __name__ == "__main__":
    unittest.main(verbosity=2)