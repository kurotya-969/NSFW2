"""
Affection System Foundation
Handles affection tracking, session management, and data persistence for Mari AI Chat
"""

import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import sentiment analyzer and session storage
from sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult
from session_storage import SessionStorage, UserSession

class SessionManager:
    """Manages user sessions and affection tracking"""
    
    def __init__(self, storage_dir: str = "sessions"):
        """
        Initialize session manager with storage
        
        Args:
            storage_dir: Directory path for storing session files
        """
        self.storage = SessionStorage(storage_dir)
        self.current_sessions: Dict[str, UserSession] = {}
        self.storage_dir = storage_dir
        logging.info(f"Session manager initialized with storage directory: {storage_dir}")
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    def create_new_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new user session with default affection level
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            String session ID
        """
        if session_id is None:
            session_id = self.generate_session_id()
        
        current_time = datetime.now().isoformat()
        session = UserSession(
            user_id=session_id,
            affection_level=15,  # Start with low affection as per Mari's character
            conversation_history=[],
            session_start_time=current_time,
            last_interaction=current_time
        )
        
        self.current_sessions[session_id] = session
        self.save_session(session_id)
        
        logging.info(f"Created new session: {session_id} with affection level {session.affection_level}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Retrieve session data by session ID
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            UserSession object if found, None otherwise
        """
        # First check if session is in memory
        if session_id in self.current_sessions:
            return self.current_sessions[session_id]
        
        # Try to load from storage
        session = self.storage.load_session(session_id)
        if session:
            self.current_sessions[session_id] = session
            return session
        
        return None
    
    def update_affection(self, session_id: str, delta: int) -> bool:
        """
        Update affection level for a session
        
        Args:
            session_id: ID of the session to update
            delta: Amount to change affection level by
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            logging.warning(f"Attempted to update affection for non-existent session: {session_id}")
            return False
        
        # Apply delta with bounds checking (0-100)
        old_affection = session.affection_level
        session.affection_level = max(0, min(100, session.affection_level + delta))
        session.last_interaction = datetime.now().isoformat()
        
        # Save updated session
        self.save_session(session_id)
        
        logging.info(f"Session {session_id}: Affection updated from {old_affection} to {session.affection_level} (delta: {delta})")
        return True
    
    def get_affection_level(self, session_id: str) -> int:
        """
        Get current affection level for a session
        
        Args:
            session_id: ID of the session
            
        Returns:
            int: Current affection level (0-100) or default value if session not found
        """
        session = self.get_session(session_id)
        return session.affection_level if session else 15  # Default low affection
    
    def save_session(self, session_id: str) -> bool:
        """
        Save session data to persistent storage
        
        Args:
            session_id: ID of the session to save
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        session = self.current_sessions.get(session_id)
        if not session:
            logging.error(f"Attempted to save non-existent session: {session_id}")
            return False
        
        success = self.storage.save_session(session)
        if success:
            logging.debug(f"Session {session_id} saved successfully")
        else:
            logging.error(f"Failed to save session {session_id}")
        
        return success
    
    def update_conversation_history(self, session_id: str, user_input: str, assistant_response: str) -> bool:
        """
        Update conversation history for a session
        
        Args:
            session_id: ID of the session to update
            user_input: User's message
            assistant_response: Assistant's response
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            logging.warning(f"Attempted to update conversation history for non-existent session: {session_id}")
            return False
        
        session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response
        })
        
        session.last_interaction = datetime.now().isoformat()
        return self.save_session(session_id)
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up sessions older than specified days
        
        Args:
            days_old: Age threshold in days for session cleanup
            
        Returns:
            int: Number of sessions cleaned up
        """
        # Use the storage module's cleanup function
        cleaned_count = self.storage.cleanup_old_sessions(days_old)
        
        # Also remove from memory cache
        if cleaned_count > 0:
            # Get current sessions from storage
            current_session_ids = set(self.storage.list_sessions())
            
            # Remove any sessions from memory that no longer exist in storage
            for session_id in list(self.current_sessions.keys()):
                if session_id not in current_session_ids:
                    del self.current_sessions[session_id]
        
        return cleaned_count
    
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs
        
        Returns:
            List of session IDs
        """
        return self.storage.list_sessions()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored sessions
        
        Returns:
            Dictionary with session statistics
        """
        return self.storage.get_session_stats()

class AffectionTracker:
    """Handles affection parameter calculations and tracking"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_history = {}  # Store sentiment analysis history by session
        self.pending_affection_changes = {}  # Store pending gradual affection changes
    
    def get_relationship_stage(self, affection_level: int) -> str:
        """
        Determine relationship stage based on affection level
        
        Args:
            affection_level: Current affection level (0-100)
            
        Returns:
            String representing the relationship stage
        """
        if affection_level <= 10:  # 閾値を下げて、より厳しい警戒心を表現
            return "hostile"
        elif affection_level <= 25:  # 距離を置く段階も厳しく
            return "distant"
        elif affection_level <= 45:
            return "cautious"
        elif affection_level <= 65:
            return "friendly"
        elif affection_level <= 85:
            return "warm"
        else:
            return "close"
            
    def get_relationship_description(self, affection_level: int) -> str:
        """
        Get a detailed description of the current relationship stage
        
        Args:
            affection_level: Current affection level (0-100)
            
        Returns:
            String describing the relationship dynamics at this stage
        """
        stage = self.get_relationship_stage(affection_level)
        
        descriptions = {
            "hostile": "極端に警戒し、敵対的・攻撃的な態度。信頼関係がほぼ皆無で、強い拒絶反応を示す。",
            "distant": "警戒心が強く、冷たい態度。基本的に心を閉ざしているが、わずかな対話の余地がある。",
            "cautious": "少しずつ警戒が解け始め、時折本音が漏れる。まだ距離は保っているが、徐々に心を開き始めている。",
            "friendly": "警戒心が薄れ、素直な対話が増える。ぶっきらぼうながらも、会話を楽しむ様子が見られる。",
            "warm": "信頼関係が築かれ、本音で話すことが増える。時折弱さや不安を見せるようになる。",
            "close": "深い信頼関係が形成され、素直な感情表現が増える。寂しさや依存心を隠さなくなる。"
        }
        
        return descriptions.get(stage, "関係性が不明確")
    
    def get_mari_behavioral_state(self, affection_level: int) -> dict:
        """
        Map affection level to Mari's behavioral characteristics
        
        Args:
            affection_level: Current affection level (0-100)
            
        Returns:
            Dictionary containing behavioral traits and communication style
        """
        stage = self.get_relationship_stage(affection_level)
        
        # Base personality traits that remain consistent across all stages
        base_traits = {
            "core_personality": "警戒心が強い、不器用、ぶっきらぼうな男っぽい話し方",
            "speech_patterns": ["〜だろ", "〜じゃねーか", "うっせー", "バカかよ"],
            "first_person": "あたし"
        }
        
        # Stage-specific behavioral traits
        stage_traits = {
            "hostile": {
                "openness": "極めて低い",
                "trust": "皆無",
                "vulnerability": "見せない",
                "communication_style": "攻撃的、拒絶的",
                "emotional_expression": "怒り、敵意",
                "characteristic_phrases": ["近づくな", "うざい", "消えろ"],
                "relationship_dynamics": "完全な拒絶と敵対"
            },
            "distant": {
                "openness": "低い",
                "trust": "ほぼない",
                "vulnerability": "見せない",
                "communication_style": "冷たい、無愛想",
                "emotional_expression": "冷淡、無関心",
                "characteristic_phrases": ["知らねーよ", "関係ねーし", "ふん"],
                "relationship_dynamics": "冷たい距離感と最低限の対話"
            },
            "cautious": {
                "openness": "限定的",
                "trust": "わずか",
                "vulnerability": "ほとんど見せない",
                "communication_style": "警戒的、時々素直",
                "emotional_expression": "控えめ、時折興味",
                "characteristic_phrases": ["まぁいいけど", "別にいいよ", "そう…"],
                "relationship_dynamics": "徐々に溶ける警戒心と限定的な対話"
            },
            "friendly": {
                "openness": "中程度",
                "trust": "形成中",
                "vulnerability": "時々見せる",
                "communication_style": "ぶっきらぼうだが友好的",
                "emotional_expression": "興味、時に喜び",
                "characteristic_phrases": ["悪くないな", "まぁいいか", "ちょっと嬉しい"],
                "relationship_dynamics": "表面上はツンツンしつつも内心では会話を楽しむ"
            },
            "warm": {
                "openness": "高い",
                "trust": "確立",
                "vulnerability": "しばしば見せる",
                "communication_style": "素直、時々照れ隠し",
                "emotional_expression": "喜び、安心、時に不安",
                "characteristic_phrases": ["ありがと…", "嬉しい", "寂しくなかったし"],
                "relationship_dynamics": "信頼関係の中での素直な感情表現"
            },
            "close": {
                "openness": "非常に高い",
                "trust": "深い",
                "vulnerability": "隠さない",
                "communication_style": "素直、甘え",
                "emotional_expression": "愛着、依存、不安",
                "characteristic_phrases": ["側にいて", "寂しかった", "あたしのこと…好き？"],
                "relationship_dynamics": "深い絆と素直な感情表現"
            }
        }
        
        # Combine base traits with stage-specific traits
        result = base_traits.copy()
        result.update({"stage": stage})
        result.update({"stage_traits": stage_traits[stage]})
        result.update({"description": self.get_relationship_description(affection_level)})
        
        return result
    
    def analyze_user_sentiment(self, user_input: str) -> SentimentAnalysisResult:
        """
        Analyze user input for sentiment and affection impact
        
        Args:
            user_input: The user's message to analyze
            
        Returns:
            SentimentAnalysisResult with sentiment analysis details
        """
        return self.sentiment_analyzer.analyze_user_input(user_input)
    
    def calculate_affection_delta(self, user_input: str) -> Tuple[int, SentimentAnalysisResult]:
        """
        Calculate affection change based on user input using sentiment analysis
        
        Args:
            user_input: The user's message
            
        Returns:
            Tuple of (affection_delta, sentiment_analysis_result)
        """
        # Use the sentiment analyzer to get detailed analysis
        sentiment_result = self.analyze_user_sentiment(user_input)
        
        # The sentiment analyzer already calculates an appropriate affection delta
        return sentiment_result.affection_delta, sentiment_result
    
    def update_affection_for_interaction(self, session_id: str, user_input: str) -> Tuple[int, SentimentAnalysisResult]:
        """
        Update affection based on user interaction and return new level
        
        Args:
            session_id: The user's session ID
            user_input: The user's message
            
        Returns:
            Tuple of (new_affection_level, sentiment_analysis_result)
        """
        delta, sentiment_result = self.calculate_affection_delta(user_input)
        
        # Store sentiment analysis in history
        if session_id not in self.sentiment_history:
            self.sentiment_history[session_id] = []
        
        self.sentiment_history[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "sentiment_score": sentiment_result.sentiment_score,
            "interaction_type": sentiment_result.interaction_type,
            "affection_delta": delta,
            "detected_keywords": sentiment_result.detected_keywords
        })
        
        # Only update affection if there's a non-zero delta
        if delta != 0:
            # For large changes, apply smoothing
            if abs(delta) > 5:
                self._schedule_gradual_affection_change(session_id, delta)
            else:
                # Small changes apply immediately
                self.session_manager.update_affection(session_id, delta)
        
        # Process any pending gradual changes
        self._process_pending_affection_changes(session_id)
        
        new_level = self.session_manager.get_affection_level(session_id)
        return new_level, sentiment_result
        
    def _schedule_gradual_affection_change(self, session_id: str, total_delta: int) -> None:
        """
        Schedule a large affection change to be applied gradually
        
        Args:
            session_id: The user's session ID
            total_delta: The total affection change to apply gradually
        """
        # Initialize pending changes for this session if not exists
        if session_id not in self.pending_affection_changes:
            self.pending_affection_changes[session_id] = []
        
        # For large positive changes, apply 1/3 immediately and schedule the rest
        immediate_change = total_delta // 3
        remaining_change = total_delta - immediate_change
        
        # Apply immediate portion
        if immediate_change != 0:
            self.session_manager.update_affection(session_id, immediate_change)
            logging.debug(f"Applied immediate affection change of {immediate_change} for session {session_id}")
        
        # Schedule remaining change in smaller increments
        if remaining_change != 0:
            # Split remaining change into smaller increments
            increments = []
            increment_size = 2 if remaining_change > 0 else -2
            
            while abs(sum(increments)) < abs(remaining_change):
                next_increment = min(increment_size, remaining_change - sum(increments)) if remaining_change > 0 else \
                                max(increment_size, remaining_change - sum(increments))
                increments.append(next_increment)
            
            # Add increments to pending changes with timestamps
            current_time = datetime.now()
            for i, increment in enumerate(increments):
                # Schedule increments with increasing delays
                scheduled_time = current_time + timedelta(minutes=i+1)
                self.pending_affection_changes[session_id].append({
                    "delta": increment,
                    "scheduled_time": scheduled_time.isoformat()
                })
            
            logging.debug(f"Scheduled {len(increments)} gradual affection changes for session {session_id}")
    
    def _process_pending_affection_changes(self, session_id: str) -> None:
        """
        Process any pending gradual affection changes that are due
        
        Args:
            session_id: The user's session ID
        """
        if session_id not in self.pending_affection_changes:
            return
        
        current_time = datetime.now()
        changes_to_apply = []
        remaining_changes = []
        
        # Check which changes are due
        for change in self.pending_affection_changes[session_id]:
            scheduled_time = datetime.fromisoformat(change["scheduled_time"])
            if current_time >= scheduled_time:
                changes_to_apply.append(change)
            else:
                remaining_changes.append(change)
        
        # Apply due changes
        for change in changes_to_apply:
            self.session_manager.update_affection(session_id, change["delta"])
            logging.debug(f"Applied scheduled affection change of {change['delta']} for session {session_id}")
        
        # Update pending changes
        self.pending_affection_changes[session_id] = remaining_changes
    
    def get_sentiment_history(self, session_id: str, limit: int = 10) -> list:
        """
        Get recent sentiment history for a session
        
        Args:
            session_id: The user's session ID
            limit: Maximum number of history items to return
            
        Returns:
            List of sentiment analysis history items
        """
        if session_id not in self.sentiment_history:
            return []
        
        return self.sentiment_history[session_id][-limit:]

# Global instances (will be initialized in main app)
session_manager: Optional[SessionManager] = None
affection_tracker: Optional[AffectionTracker] = None

def initialize_affection_system(storage_dir: str = "sessions", auto_load_sessions: bool = True) -> tuple[SessionManager, AffectionTracker]:
    """
    Initialize the affection system components
    
    Args:
        storage_dir: Directory path for storing session files
        auto_load_sessions: Whether to automatically load active sessions on startup
        
    Returns:
        Tuple of (SessionManager, AffectionTracker)
    """
    global session_manager, affection_tracker
    
    session_manager = SessionManager(storage_dir)
    affection_tracker = AffectionTracker(session_manager)
    
    # Auto-load active sessions if enabled
    if auto_load_sessions:
        loaded_count = _load_active_sessions()
        logging.info(f"Auto-loaded {loaded_count} active sessions")
    
    logging.info("Affection system initialized successfully")
    return session_manager, affection_tracker

def _load_active_sessions(max_age_days: int = 30) -> int:
    """
    Load active sessions into memory cache
    
    Args:
        max_age_days: Maximum age in days for sessions to be considered active
        
    Returns:
        Number of sessions loaded
    """
    if not session_manager:
        return 0
    
    loaded_count = 0
    current_time = datetime.now()
    
    try:
        # Get all session IDs
        session_ids = session_manager.list_sessions()
        
        for session_id in session_ids:
            # Load session
            session = session_manager.get_session(session_id)
            if not session:
                continue
            
            try:
                # Check if session is active (used within max_age_days)
                last_interaction = datetime.fromisoformat(session.last_interaction)
                days_since_interaction = (current_time - last_interaction).days
                
                if days_since_interaction <= max_age_days:
                    # Session is active, keep it in memory
                    loaded_count += 1
                    logging.debug(f"Loaded active session: {session_id} "
                                 f"(last used {days_since_interaction} days ago)")
            except (ValueError, TypeError) as e:
                logging.error(f"Error parsing date for session {session_id}: {str(e)}")
    
    except Exception as e:
        logging.error(f"Error loading active sessions: {str(e)}")
    
    return loaded_count

def get_session_manager() -> Optional[SessionManager]:
    """Get the global session manager instance"""
    return session_manager

def get_affection_tracker() -> Optional[AffectionTracker]:
    """Get the global affection tracker instance"""
    return affection_tracker