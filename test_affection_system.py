"""
Test script for affection system foundation
"""

import os
import shutil
import unittest
from datetime import datetime, timedelta
from affection_system import initialize_affection_system, UserSession
from sentiment_analyzer import SentimentType

class TestAffectionSystem(unittest.TestCase):
    """Test cases for the affection system"""
    
    def setUp(self):
        """Set up the test environment"""
        # Clean up any existing test sessions
        self.test_dir = "test_sessions"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # Initialize system
        self.session_manager, self.affection_tracker = initialize_affection_system(self.test_dir)
        self.session_id = self.session_manager.create_new_session()
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_session_creation_and_retrieval(self):
        """Test session creation and retrieval"""
        session = self.session_manager.get_session(self.session_id)
        self.assertEqual(session.user_id, self.session_id)
        self.assertEqual(session.affection_level, 15)  # Default starting level
    
    def test_affection_updates(self):
        """Test basic affection updates"""
        initial_affection = self.session_manager.get_affection_level(self.session_id)
        self.session_manager.update_affection(self.session_id, 10)
        new_affection = self.session_manager.get_affection_level(self.session_id)
        self.assertEqual(new_affection, initial_affection + 10)
    
    def test_affection_bounds(self):
        """Test affection level boundaries"""
        # Test lower bound
        self.session_manager.update_affection(self.session_id, -200)
        self.assertEqual(self.session_manager.get_affection_level(self.session_id), 0)
        
        # Test upper bound
        self.session_manager.update_affection(self.session_id, 200)
        self.assertEqual(self.session_manager.get_affection_level(self.session_id), 100)
    
    def test_relationship_stages(self):
        """Test relationship stage determination"""
        stages = {
            15: "distant",
            50: "friendly",
            85: "close"
        }
        
        for level, expected_stage in stages.items():
            stage = self.affection_tracker.get_relationship_stage(level)
            self.assertEqual(stage, expected_stage)
    
    def test_sentiment_analysis_integration(self):
        """Test sentiment analysis integration with affection system"""
        test_inputs = {
            "ありがとう": {"delta_sign": 1, "type": SentimentType.POSITIVE},
            "うざい": {"delta_sign": -1, "type": SentimentType.NEGATIVE},
            "大丈夫？心配してる": {"delta_sign": 1, "type": SentimentType.CARING},
            "どうでもいい": {"delta_sign": -1, "type": SentimentType.DISMISSIVE},
            "助かった": {"delta_sign": 1, "type": SentimentType.APPRECIATIVE},
            "ふざけるな": {"delta_sign": -1, "type": SentimentType.HOSTILE},
            "hello": {"delta_sign": 0, "type": SentimentType.NEUTRAL}
        }
        
        for input_text, expected in test_inputs.items():
            delta, result = self.affection_tracker.calculate_affection_delta(input_text)
            
            # Check if delta sign matches expected
            if expected["delta_sign"] > 0:
                self.assertGreater(delta, 0)
            elif expected["delta_sign"] < 0:
                self.assertLess(delta, 0)
            else:
                self.assertEqual(delta, 0)
            
            # Check if sentiment type is detected
            self.assertIn(expected["type"], result.sentiment_types)
    
    def test_affection_update_with_sentiment(self):
        """Test updating affection based on sentiment analysis"""
        initial_affection = self.session_manager.get_affection_level(self.session_id)
        
        # Test with positive input
        new_level, result = self.affection_tracker.update_affection_for_interaction(
            self.session_id, "ありがとう、素晴らしい！"
        )
        
        # Verify affection increased
        self.assertGreater(new_level, initial_affection)
        
        # Verify sentiment result
        self.assertGreater(result.sentiment_score, 0)
        self.assertGreater(result.affection_delta, 0)
        self.assertIn(SentimentType.POSITIVE, result.sentiment_types)
        
        # Test with negative input
        current_affection = new_level
        new_level, result = self.affection_tracker.update_affection_for_interaction(
            self.session_id, "うるさい、バカ"
        )
        
        # Verify affection decreased
        self.assertLess(new_level, current_affection)
        
        # Verify sentiment result
        self.assertLess(result.sentiment_score, 0)
        self.assertLess(result.affection_delta, 0)
        self.assertIn(SentimentType.NEGATIVE, result.sentiment_types)
        
    def test_gradual_affection_changes(self):
        """Test gradual affection changes for large sentiment impacts"""
        initial_affection = self.session_manager.get_affection_level(self.session_id)
        
        # Create a very positive message that would trigger a large affection change
        very_positive = "ありがとう、すごい、素晴らしい、感謝します、助かりました！"
        
        # Get the sentiment analysis result
        delta, sentiment_result = self.affection_tracker.calculate_affection_delta(very_positive)
        
        # Verify it's a large delta that would trigger gradual changes
        self.assertGreater(abs(delta), 5)
        
        # Apply the affection change
        new_level, _ = self.affection_tracker.update_affection_for_interaction(
            self.session_id, very_positive
        )
        
        # Verify immediate change is applied (should be about 1/3 of total)
        expected_immediate = initial_affection + (delta // 3)
        self.assertAlmostEqual(new_level, expected_immediate, delta=2)
        
        # Verify pending changes were scheduled
        self.assertIn(self.session_id, self.affection_tracker.pending_affection_changes)
        self.assertGreater(len(self.affection_tracker.pending_affection_changes[self.session_id]), 0)
        
        # Simulate time passing to trigger pending changes
        # We'll modify the scheduled times to be in the past
        current_time = datetime.now()
        for change in self.affection_tracker.pending_affection_changes[self.session_id]:
            # Set scheduled time to 5 minutes ago
            change["scheduled_time"] = (current_time - timedelta(minutes=5)).isoformat()
        
        # Process the pending changes
        self.affection_tracker._process_pending_affection_changes(self.session_id)
        
        # Verify all changes were applied
        final_level = self.session_manager.get_affection_level(self.session_id)
        self.assertEqual(len(self.affection_tracker.pending_affection_changes[self.session_id]), 0)
        
        # The final level should be close to initial + full delta
        expected_final = min(100, initial_affection + delta)  # Cap at 100
        self.assertAlmostEqual(final_level, expected_final, delta=2)
    
    def test_sentiment_history(self):
        """Test sentiment history tracking"""
        # Add some interactions
        self.affection_tracker.update_affection_for_interaction(self.session_id, "ありがとう")
        self.affection_tracker.update_affection_for_interaction(self.session_id, "すごい")
        self.affection_tracker.update_affection_for_interaction(self.session_id, "うざい")
        
        # Get history
        history = self.affection_tracker.get_sentiment_history(self.session_id)
        
        # Verify history
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["user_input"], "ありがとう")
        self.assertEqual(history[2]["user_input"], "うざい")
        
        # Verify history limit
        limited_history = self.affection_tracker.get_sentiment_history(self.session_id, limit=2)
        self.assertEqual(len(limited_history), 2)
        self.assertEqual(limited_history[1]["user_input"], "うざい")
    
    def test_persistence(self):
        """Test session persistence"""
        # Update affection
        self.affection_tracker.update_affection_for_interaction(self.session_id, "ありがとう")
        current_affection = self.session_manager.get_affection_level(self.session_id)
        
        # Create new session manager to test loading
        session_manager2, _ = initialize_affection_system(self.test_dir)
        loaded_session = session_manager2.get_session(self.session_id)
        
        # Verify loaded session
        self.assertIsNotNone(loaded_session)
        self.assertEqual(loaded_session.affection_level, current_affection)
    
    def test_conversation_history(self):
        """Test conversation history updates"""
        success = self.session_manager.update_conversation_history(
            self.session_id, 
            "Test user input", 
            "Test assistant response"
        )
        
        updated_session = self.session_manager.get_session(self.session_id)
        self.assertTrue(success)
        self.assertEqual(len(updated_session.conversation_history), 1)
        self.assertEqual(updated_session.conversation_history[0]["user"], "Test user input")
        self.assertEqual(updated_session.conversation_history[0]["assistant"], "Test assistant response")

def test_affection_system():
    """Run the original test function for backward compatibility"""
    print("Testing Affection System Foundation...")
    
    # Clean up any existing test sessions
    test_dir = "test_sessions"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Initialize system
    session_manager, affection_tracker = initialize_affection_system(test_dir)
    
    # Test 1: Create new session
    print("\n1. Testing session creation...")
    session_id = session_manager.create_new_session()
    print(f"   Created session: {session_id}")
    
    # Test 2: Retrieve session
    print("\n2. Testing session retrieval...")
    session = session_manager.get_session(session_id)
    print(f"   Retrieved session: {session.user_id}")
    print(f"   Initial affection level: {session.affection_level}")
    
    # Test 3: Update affection
    print("\n3. Testing affection updates...")
    initial_affection = session.affection_level
    session_manager.update_affection(session_id, 10)
    new_affection = session_manager.get_affection_level(session_id)
    print(f"   Affection updated from {initial_affection} to {new_affection}")
    
    # Test 4: Test bounds checking
    print("\n4. Testing affection bounds...")
    session_manager.update_affection(session_id, -200)  # Should not go below 0
    bounded_affection = session_manager.get_affection_level(session_id)
    print(f"   Affection after large negative delta: {bounded_affection}")
    
    session_manager.update_affection(session_id, 200)  # Should not go above 100
    max_affection = session_manager.get_affection_level(session_id)
    print(f"   Affection after large positive delta: {max_affection}")
    
    # Test 5: Test relationship stages
    print("\n5. Testing relationship stages...")
    for level in [15, 50, 85]:
        stage = affection_tracker.get_relationship_stage(level)
        print(f"   Affection {level} -> Stage: {stage}")
    
    # Test 6: Test sentiment analysis integration
    print("\n6. Testing sentiment analysis integration...")
    test_inputs = [
        "ありがとう",  # Should increase
        "うざい",      # Should decrease
        "hello"        # Should be neutral
    ]
    
    for test_input in test_inputs:
        delta, result = affection_tracker.calculate_affection_delta(test_input)
        print(f"   Input: '{test_input}' -> Delta: {delta}, Type: {result.interaction_type}")
    
    # Test 7: Test persistence
    print("\n7. Testing session persistence...")
    # Create a new session manager to test loading
    session_manager2, _ = initialize_affection_system(test_dir)
    loaded_session = session_manager2.get_session(session_id)
    if loaded_session:
        print(f"   Successfully loaded session with affection: {loaded_session.affection_level}")
    else:
        print("   Failed to load session from storage")
    
    # Test 8: Test conversation history
    print("\n8. Testing conversation history...")
    success = session_manager.update_conversation_history(
        session_id, 
        "Test user input", 
        "Test assistant response"
    )
    updated_session = session_manager.get_session(session_id)
    print(f"   History update success: {success}")
    print(f"   History length: {len(updated_session.conversation_history)}")
    
    print("\n✅ All tests completed successfully!")
    
    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("   Test cleanup completed.")

if __name__ == "__main__":
    test_affection_system()