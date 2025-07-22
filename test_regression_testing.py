"""
Regression Testing for Enhanced Sentiment Analysis System
Verifies existing functionality remains intact, tests backward compatibility with current sessions,
and validates performance with enhanced analysis.
Task 8.3: Perform regression testing
Requirements: 5.3
"""

import os
import shutil
import unittest
import tempfile
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from affection_system import (
    SessionManager, AffectionTracker, initialize_affection_system,
    get_session_manager, get_affection_tracker
)
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult, SentimentType
from enhanced_sentiment_adapter import EnhancedSentimentAdapter
from context_sentiment_detector import ContextSentimentDetector
from session_storage import UserSession, SessionStorage

class TestRegressionTesting(unittest.TestCase):
    """Regression test suite for the enhanced sentiment analysis system"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp(prefix="test_regression_")
        
        # Initialize the affection system
        self.session_manager, self.affection_tracker = initialize_affection_system(
            self.test_dir, auto_load_sessions=False
        )
        
        # Initialize enhanced sentiment adapter
        self.enhanced_adapter = EnhancedSentimentAdapter()
        
        # Create test sessions with different affection levels
        self.session_ids = {}
        self.create_test_sessions()
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_test_sessions(self):
        """Create test sessions with different affection levels"""
        # Create sessions with different affection levels
        affection_levels = {
            "hostile": 5,
            "distant": 20,
            "cautious": 40,
            "friendly": 60,
            "warm": 80,
            "close": 95
        }
        
        for stage, level in affection_levels.items():
            session_id = self.session_manager.create_new_session()
            
            # Set affection level directly
            session = self.session_manager.get_session(session_id)
            session.affection_level = level
            self.session_manager.save_session(session_id)
            
            # Store session ID for later use
            self.session_ids[stage] = session_id
            
            # Add some conversation history
            self.session_manager.update_conversation_history(
                session_id, 
                f"Test message for {stage} stage", 
                f"Test response for {stage} stage"
            )
    
    # SECTION 1: VERIFY EXISTING FUNCTIONALITY REMAINS INTACT
    
    def test_basic_sentiment_analysis_functionality(self):
        """Test that basic sentiment analysis functionality remains intact"""
        # Test inputs with different sentiment characteristics
        test_inputs = {
            "ありがとう": {"expected_type": "positive", "expected_delta_sign": 1},
            "うるさいな": {"expected_type": "negative", "expected_delta_sign": -1},
            "こんにちは": {"expected_type": "neutral", "expected_delta_sign": 0},
            "心配だよ": {"expected_type": "caring", "expected_delta_sign": 1},
            "どうでもいい": {"expected_type": "dismissive", "expected_delta_sign": -1},
            "感謝してる": {"expected_type": "appreciative", "expected_delta_sign": 1},
            "てめえ": {"expected_type": "hostile", "expected_delta_sign": -1}
        }
        
        # Test with original sentiment analyzer
        original_analyzer = SentimentAnalyzer()
        
        for input_text, expected in test_inputs.items():
            # Get original analysis
            original_result = original_analyzer.analyze_user_input(input_text)
            
            # Verify basic functionality
            self.assertEqual(original_result.interaction_type, expected["expected_type"], 
                           f"Original analyzer failed for input: {input_text}")
            
            if expected["expected_delta_sign"] > 0:
                self.assertGreater(original_result.affection_delta, 0, 
                                 f"Original analyzer should return positive delta for: {input_text}")
            elif expected["expected_delta_sign"] < 0:
                self.assertLess(original_result.affection_delta, 0, 
                              f"Original analyzer should return negative delta for: {input_text}")
            else:
                self.assertEqual(original_result.affection_delta, 0, 
                               f"Original analyzer should return zero delta for: {input_text}")
            
            # Get enhanced analysis
            enhanced_result = self.enhanced_adapter.analyze_user_input(input_text)
            
            # Verify enhanced analyzer maintains same basic functionality
            self.assertEqual(enhanced_result.interaction_type, expected["expected_type"], 
                           f"Enhanced analyzer failed for input: {input_text}")
            
            if expected["expected_delta_sign"] > 0:
                self.assertGreater(enhanced_result.affection_delta, 0, 
                                 f"Enhanced analyzer should return positive delta for: {input_text}")
            elif expected["expected_delta_sign"] < 0:
                self.assertLess(enhanced_result.affection_delta, 0, 
                              f"Enhanced analyzer should return negative delta for: {input_text}")
    
    def test_affection_tracking_functionality(self):
        """Test that affection tracking functionality remains intact"""
        # Test with both original and enhanced analyzers
        session_id_original = self.session_manager.create_new_session()
        session_id_enhanced = self.session_manager.create_new_session()
        
        # Test inputs that should affect affection
        test_inputs = [
            "ありがとう",  # Positive
            "うるさいな",  # Negative
            "心配だよ",    # Caring
            "感謝してる"   # Appreciative
        ]
        
        # Process with original analyzer
        original_levels = [self.session_manager.get_affection_level(session_id_original)]
        for input_text in test_inputs:
            new_level, _ = self.affection_tracker.update_affection_for_interaction(
                session_id_original, input_text
            )
            original_levels.append(new_level)
        
        # Process with enhanced analyzer
        enhanced_levels = [self.session_manager.get_affection_level(session_id_enhanced)]
        with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
            for input_text in test_inputs:
                new_level, _ = self.affection_tracker.update_affection_for_interaction(
                    session_id_enhanced, input_text
                )
                enhanced_levels.append(new_level)
        
        # Verify both analyzers change affection levels
        self.assertNotEqual(original_levels[0], original_levels[-1], 
                          "Original analyzer should change affection level")
        self.assertNotEqual(enhanced_levels[0], enhanced_levels[-1], 
                          "Enhanced analyzer should change affection level")
        
        # Verify both analyzers produce valid affection levels
        for level in original_levels + enhanced_levels:
            self.assertGreaterEqual(level, 0, "Affection level should be >= 0")
            self.assertLessEqual(level, 100, "Affection level should be <= 100")
    
    def test_relationship_stage_functionality(self):
        """Test that relationship stage functionality remains intact"""
        # Test relationship stage determination
        stages = ["hostile", "distant", "cautious", "friendly", "warm", "close"]
        
        for stage in stages:
            session_id = self.session_ids[stage]
            affection_level = self.session_manager.get_affection_level(session_id)
            
            # Get relationship stage
            relationship_stage = self.affection_tracker.get_relationship_stage(affection_level)
            
            # Verify stage matches expected
            self.assertEqual(relationship_stage, stage, 
                           f"Expected relationship stage {stage} for affection level {affection_level}")
            
            # Get behavioral state
            behavioral_state = self.affection_tracker.get_mari_behavioral_state(affection_level)
            
            # Verify behavioral state contains expected keys
            self.assertIn("core_personality", behavioral_state)
            self.assertIn("speech_patterns", behavioral_state)
            self.assertIn("stage", behavioral_state)
            self.assertIn("stage_traits", behavioral_state)
            
            # Verify stage in behavioral state matches expected
            self.assertEqual(behavioral_state["stage"], stage)
    
    def test_conversation_history_functionality(self):
        """Test that conversation history functionality remains intact"""
        # Test conversation history tracking
        session_id = self.session_manager.create_new_session()
        
        # Add conversation history
        test_conversations = [
            ("Hello", "Hi there"),
            ("How are you?", "I'm fine"),
            ("What's your name?", "My name is Mari")
        ]
        
        for user_input, assistant_response in test_conversations:
            self.session_manager.update_conversation_history(
                session_id, user_input, assistant_response
            )
        
        # Get session and verify history
        session = self.session_manager.get_session(session_id)
        self.assertEqual(len(session.conversation_history), len(test_conversations))
        
        # Verify history entries have expected format
        for i, entry in enumerate(session.conversation_history):
            self.assertIn("timestamp", entry)
            self.assertIn("user", entry)
            self.assertIn("assistant", entry)
            self.assertEqual(entry["user"], test_conversations[i][0])
            self.assertEqual(entry["assistant"], test_conversations[i][1])
    
    # SECTION 2: TEST BACKWARD COMPATIBILITY WITH CURRENT SESSIONS
    
    def test_backward_compatibility_with_existing_sessions(self):
        """Test backward compatibility with existing sessions"""
        # For each test session, verify enhanced analyzer works with existing data
        for stage, session_id in self.session_ids.items():
            # Get original session data
            original_session = self.session_manager.get_session(session_id)
            original_affection = original_session.affection_level
            
            # Process a new message with enhanced analyzer
            test_input = "こんにちは"  # Neutral greeting
            
            with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
                new_level, sentiment_result = self.affection_tracker.update_affection_for_interaction(
                    session_id, test_input
                )
                
                # Explicitly update conversation history
                self.session_manager.update_conversation_history(
                    session_id, test_input, f"Response to: {test_input}"
                )
            
            # Verify session still exists and can be loaded
            updated_session = self.session_manager.get_session(session_id)
            self.assertIsNotNone(updated_session)
            
            # Verify affection level was updated properly
            self.assertIsInstance(new_level, int)
            self.assertGreaterEqual(new_level, 0)
            self.assertLessEqual(new_level, 100)
            
            # Verify conversation history was preserved and updated
            self.assertEqual(len(updated_session.conversation_history), 2)  # Original + new entry
            self.assertEqual(updated_session.conversation_history[0]["user"], 
                           f"Test message for {stage} stage")
            self.assertEqual(updated_session.conversation_history[0]["assistant"], 
                           f"Test response for {stage} stage")
            self.assertEqual(updated_session.conversation_history[1]["user"], test_input)
    
    def test_session_persistence_with_enhanced_analysis(self):
        """Test session persistence with enhanced analysis"""
        # Create a new session
        session_id = self.session_manager.create_new_session()
        
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
                
                # Update conversation history
                self.session_manager.update_conversation_history(
                    session_id, input_text, f"Response to: {input_text}"
                )
        
        # Save session explicitly
        self.session_manager.save_session(session_id)
        
        # Clear in-memory sessions
        self.session_manager.current_sessions = {}
        
        # Try to load the session from storage
        loaded_session = self.session_manager.get_session(session_id)
        
        # Verify session was loaded successfully
        self.assertIsNotNone(loaded_session)
        self.assertEqual(len(loaded_session.conversation_history), len(test_inputs))
        
        # Verify conversation history was preserved
        for i, input_text in enumerate(test_inputs):
            self.assertEqual(loaded_session.conversation_history[i]["user"], input_text)
            self.assertEqual(loaded_session.conversation_history[i]["assistant"], 
                           f"Response to: {input_text}")
    
    def test_session_cleanup_with_enhanced_analysis(self):
        """Test session cleanup with enhanced analysis"""
        # Create a session with old timestamp
        session_id = self.session_manager.create_new_session()
        session = self.session_manager.get_session(session_id)
        
        # Set last interaction to 31 days ago
        old_date = (datetime.now() - timedelta(days=31)).isoformat()
        session.last_interaction = old_date
        self.session_manager.save_session(session_id)
        
        # Process a message with enhanced analyzer
        with patch.object(self.affection_tracker, 'sentiment_analyzer', self.enhanced_adapter):
            self.affection_tracker.update_affection_for_interaction(
                session_id, "こんにちは"
            )
        
        # Run cleanup
        cleaned_count = self.session_manager.cleanup_old_sessions(days_old=30)
        
        # Verify session was cleaned up
        self.assertEqual(cleaned_count, 1)
        
        # Verify session no longer exists
        loaded_session = self.session_manager.get_session(session_id)
        self.assertIsNone(loaded_session)
    
    # SECTION 3: VALIDATE PERFORMANCE WITH ENHANCED ANALYSIS
    
    def test_performance_comparison(self):
        """Test performance comparison between original and enhanced analyzers"""
        # Test inputs with different complexity
        test_inputs = [
            "こんにちは",  # Simple
            "ありがとう",  # Simple positive
            "うるさいな",  # Simple negative
            "今日はとても良い天気ですね",  # Longer simple
            "素晴らしいですね、また失敗しましたか",  # Context-dependent
            "良くないね",  # Negated sentiment
            "嬉しいけど、少し不安もある"  # Mixed emotions
        ]
        
        # Measure time for original analyzer
        original_analyzer = SentimentAnalyzer()
        original_times = []
        
        for input_text in test_inputs:
            start_time = time.time()
            original_analyzer.analyze_user_input(input_text)
            end_time = time.time()
            original_times.append(end_time - start_time)
        
        # Measure time for enhanced analyzer
        enhanced_times = []
        
        for input_text in test_inputs:
            start_time = time.time()
            self.enhanced_adapter.analyze_user_input(input_text)
            end_time = time.time()
            enhanced_times.append(end_time - start_time)
        
        # Calculate average times
        avg_original = sum(original_times) / len(original_times)
        avg_enhanced = sum(enhanced_times) / len(enhanced_times)
        
        # Log performance comparison
        print(f"\nPerformance comparison:")
        print(f"Original analyzer average time: {avg_original:.6f} seconds")
        print(f"Enhanced analyzer average time: {avg_enhanced:.6f} seconds")
        
        # Avoid division by zero
        if avg_original > 0:
            print(f"Performance ratio: {avg_enhanced / avg_original:.2f}x")
        else:
            print("Performance ratio: N/A (original time too small to measure)")
        
        # Enhanced analyzer is expected to be slower due to additional processing
        # but should still be within reasonable limits for interactive use
        self.assertLess(avg_enhanced, 1.0, 
                      "Enhanced analyzer should process inputs in under 1 second on average")
    
    def test_fallback_mechanism_performance(self):
        """Test fallback mechanism performance"""
        # Create a mock that raises an exception
        mock_context_detector = MagicMock()
        mock_context_detector.analyze_with_context.side_effect = Exception("Test error")
        
        # Create adapter with mock
        adapter = EnhancedSentimentAdapter()
        adapter.context_sentiment_detector = mock_context_detector
        
        # Test inputs - use inputs that are not in the keywords_to_match list
        test_inputs = [
            "テスト",
            "サンプル",
            "例"
        ]
        
        # Measure fallback performance
        fallback_times = []
        
        for input_text in test_inputs:
            start_time = time.time()
            result = adapter.analyze_user_input(input_text)
            end_time = time.time()
            fallback_times.append(end_time - start_time)
            
            # Verify result is valid despite error
            self.assertIsInstance(result, SentimentAnalysisResult)
            self.assertIsInstance(result.sentiment_score, float)
            self.assertIsInstance(result.affection_delta, int)
        
        # Calculate average fallback time
        avg_fallback = sum(fallback_times) / len(fallback_times)
        
        # Log fallback performance
        print(f"\nFallback performance:")
        print(f"Average fallback time: {avg_fallback:.6f} seconds")
        
        # Fallback should be reasonably fast
        self.assertLess(avg_fallback, 0.5, 
                      "Fallback mechanism should process inputs in under 0.5 seconds on average")
        
        # Verify fallback statistics
        fallback_stats = adapter.get_fallback_stats()
        self.assertEqual(fallback_stats["total_attempts"], len(test_inputs))
        self.assertEqual(fallback_stats["successful_attempts"], len(test_inputs))
    
    def test_toggle_performance(self):
        """Test performance when toggling between original and enhanced analysis"""
        # Test input
        test_input = "こんにちは"
        
        # Create adapter
        adapter = EnhancedSentimentAdapter()
        
        # Measure time with enhanced analysis enabled
        start_time = time.time()
        adapter.analyze_user_input(test_input)
        enhanced_time = time.time() - start_time
        
        # Toggle to original analysis
        adapter.toggle_enhanced_analysis(False)
        
        # Measure time with original analysis
        start_time = time.time()
        adapter.analyze_user_input(test_input)
        original_time = time.time() - start_time
        
        # Toggle back to enhanced analysis
        adapter.toggle_enhanced_analysis(True)
        
        # Measure time with enhanced analysis again
        start_time = time.time()
        adapter.analyze_user_input(test_input)
        enhanced_time2 = time.time() - start_time
        
        # Log toggle performance
        print(f"\nToggle performance:")
        print(f"Enhanced analysis time: {enhanced_time:.6f} seconds")
        print(f"Original analysis time: {original_time:.6f} seconds")
        print(f"Enhanced analysis time (after toggle): {enhanced_time2:.6f} seconds")
        
        # Original analysis should be faster or equal
        # Note: In some environments, the timing might be too small to measure accurately
        if original_time > 0 and enhanced_time > 0:
            self.assertLessEqual(original_time, enhanced_time, 
                              "Original analysis should be faster than enhanced analysis")
        else:
            # Skip the test if timing is too small to measure
            print("Timing too small to measure accurately, skipping comparison")
        
        # Enhanced analysis after toggle should be similar to initial enhanced analysis
        # Avoid division by zero
        if enhanced_time > 0:
            ratio = enhanced_time2 / enhanced_time
            self.assertGreater(ratio, 0.5, "Enhanced analysis performance should be consistent after toggle")
            self.assertLess(ratio, 2.0, "Enhanced analysis performance should be consistent after toggle")
        else:
            # Skip the test if timing is too small to measure
            print("Timing too small to measure accurately, skipping ratio comparison")

if __name__ == "__main__":
    unittest.main(verbosity=2)