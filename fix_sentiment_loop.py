"""
Fix for the "じゃあな！" loop issue in the Mari AI Chat system
This script helps you break out of a negative sentiment loop by directly modifying the session
"""

import os
import sys
from datetime import datetime
from affection_system import initialize_affection_system, get_session_manager, get_affection_tracker

def list_sessions():
    """List all available sessions with their affection levels"""
    session_manager, affection_tracker = initialize_affection_system("sessions")
    
    session_ids = session_manager.list_sessions()
    if not session_ids:
        print("No sessions found.")
        return []
    
    print(f"Available sessions ({len(session_ids)}):")
    for i, session_id in enumerate(session_ids):
        session = session_manager.get_session(session_id)
        if session:
            affection_level = session.affection_level
            relationship_stage = affection_tracker.get_relationship_stage(affection_level)
            print(f"{i+1}. ID: {session_id}")
            print(f"   Affection Level: {affection_level}")
            print(f"   Relationship Stage: {relationship_stage}")
            print(f"   Last Interaction: {session.last_interaction}")
            print()
    
    return session_ids

def fix_negative_loop(session_id, new_affection_level=None):
    """Fix a negative sentiment loop by resetting affection or setting to a specific level"""
    session_manager, affection_tracker = initialize_affection_system("sessions")
    
    # Get the session
    session = session_manager.get_session(session_id)
    if not session:
        print(f"Session {session_id} not found.")
        return False
    
    # Get current affection level
    current_level = session.affection_level
    current_stage = affection_tracker.get_relationship_stage(current_level)
    
    print(f"Current affection level: {current_level} ({current_stage})")
    
    # If no new level specified, increase by 20 points
    if new_affection_level is None:
        new_affection_level = min(100, current_level + 20)
    
    # Update affection level
    session.affection_level = new_affection_level
    session.last_interaction = datetime.now().isoformat()
    
    # Save the session
    success = session_manager.storage.save_session(session)
    
    if success:
        new_stage = affection_tracker.get_relationship_stage(new_affection_level)
        print(f"Successfully updated affection level from {current_level} to {new_affection_level}")
        print(f"Relationship stage changed from '{current_stage}' to '{new_stage}'")
        
        # Clear any pending affection changes
        if session_id in affection_tracker.pending_affection_changes:
            affection_tracker.pending_affection_changes[session_id] = []
            print("Cleared pending affection changes")
        
        # Clear sentiment history for this session
        if session_id in affection_tracker.sentiment_history:
            affection_tracker.sentiment_history[session_id] = []
            print("Cleared sentiment history")
        
        return True
    else:
        print(f"Failed to update session {session_id}")
        return False

def reset_sentiment_history(session_id):
    """Reset the sentiment history for a session to break out of negative loops"""
    session_manager, affection_tracker = initialize_affection_system("sessions")
    
    # Check if session exists
    session = session_manager.get_session(session_id)
    if not session:
        print(f"Session {session_id} not found.")
        return False
    
    # Clear sentiment history
    if session_id in affection_tracker.sentiment_history:
        affection_tracker.sentiment_history[session_id] = []
        print(f"Cleared sentiment history for session {session_id}")
        return True
    else:
        print(f"No sentiment history found for session {session_id}")
        return False

def print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python fix_sentiment_loop.py list")
    print("  python fix_sentiment_loop.py fix <session_id> [new_affection_level]")
    print("  python fix_sentiment_loop.py reset <session_id>")
    print()
    print("Examples:")
    print("  python fix_sentiment_loop.py list")
    print("  python fix_sentiment_loop.py fix abc123")
    print("  python fix_sentiment_loop.py fix abc123 50")
    print("  python fix_sentiment_loop.py reset abc123")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        list_sessions()
    
    elif command == "fix" and len(sys.argv) >= 3:
        session_id = sys.argv[2]
        new_level = None
        
        if len(sys.argv) >= 4:
            try:
                new_level = int(sys.argv[3])
                if new_level < 0 or new_level > 100:
                    print("Affection level must be between 0 and 100")
                    sys.exit(1)
            except ValueError:
                print("Affection level must be an integer")
                sys.exit(1)
        
        fix_negative_loop(session_id, new_level)
    
    elif command == "reset" and len(sys.argv) >= 3:
        session_id = sys.argv[2]
        reset_sentiment_history(session_id)
    
    else:
        print_usage()
        sys.exit(1)