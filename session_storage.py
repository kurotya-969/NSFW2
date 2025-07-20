"""
Session Storage Module for Mari AI Chat
Handles file-based session persistence, serialization, and deserialization
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

@dataclass
class UserSession:
    """Data structure for tracking user session information"""
    user_id: str
    affection_level: int  # 0-100
    conversation_history: list
    session_start_time: str
    last_interaction: str
    user_metadata: Dict[str, Any] = None  # ユーザー情報を保存
    stage_transitions: List[Dict[str, Any]] = None  # 段階変化の履歴
    engagement_metrics: Dict[str, Any] = None  # エンゲージメント指標
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Create UserSession from dictionary"""
        return cls(**data)

class SessionStorage:
    """Handles file-based session storage operations"""
    
    def __init__(self, storage_dir: str = "sessions"):
        """
        Initialize session storage
        
        Args:
            storage_dir: Directory path for storing session files
        """
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist"""
        if not os.path.exists(self.storage_dir):
            try:
                os.makedirs(self.storage_dir)
                logging.info(f"Created session storage directory: {self.storage_dir}")
            except OSError as e:
                logging.error(f"Failed to create storage directory: {str(e)}")
                raise
    
    def save_session(self, session: UserSession) -> bool:
        """
        Save session data to persistent storage
        
        Args:
            session: UserSession object to save
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        if not session or not session.user_id:
            logging.error("Attempted to save invalid session")
            return False
        
        try:
            file_path = os.path.join(self.storage_dir, f"{session.user_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            
            logging.debug(f"Session {session.user_id} saved to {file_path}")
            return True
        
        except (IOError, OSError, TypeError) as e:
            logging.error(f"Failed to save session {session.user_id}: {str(e)}")
            return False
    
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """
        Load session data from persistent storage
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            UserSession object if found, None otherwise
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            
            if not os.path.exists(file_path):
                logging.debug(f"Session file not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = UserSession.from_dict(data)
            logging.debug(f"Session {session_id} loaded from {file_path}")
            return session
        
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse session file for {session_id}: {str(e)}")
            # Backup corrupted file for analysis
            self._backup_corrupted_file(file_path)
            return None
        
        except (IOError, OSError) as e:
            logging.error(f"Failed to load session {session_id}: {str(e)}")
            return None
        
        except (KeyError, TypeError) as e:
            logging.error(f"Invalid session data format for {session_id}: {str(e)}")
            self._backup_corrupted_file(file_path)
            return None
    
    def _backup_corrupted_file(self, file_path: str) -> None:
        """
        Create a backup of a corrupted session file
        
        Args:
            file_path: Path to the corrupted file
        """
        if not os.path.exists(file_path):
            return
            
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            backup_path = f"{file_path}.corrupted.{timestamp}"
            shutil.copy2(file_path, backup_path)
            logging.info(f"Created backup of corrupted session file: {backup_path}")
        except Exception as e:
            logging.error(f"Failed to backup corrupted file: {str(e)}")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session file
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            
            if not os.path.exists(file_path):
                logging.debug(f"Session file not found for deletion: {file_path}")
                return False
            
            os.remove(file_path)
            logging.info(f"Deleted session file: {file_path}")
            return True
        
        except (IOError, OSError) as e:
            logging.error(f"Failed to delete session {session_id}: {str(e)}")
            return False
    
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs
        
        Returns:
            List of session IDs
        """
        try:
            session_ids = []
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    session_ids.append(filename[:-5])  # Remove .json extension
            return session_ids
        
        except (IOError, OSError) as e:
            logging.error(f"Failed to list sessions: {str(e)}")
            return []
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up sessions older than specified days
        
        Args:
            days_old: Age threshold in days for session cleanup
            
        Returns:
            int: Number of sessions cleaned up
        """
        cleaned_count = 0
        current_time = datetime.now()
        
        try:
            for session_id in self.list_sessions():
                session = self.load_session(session_id)
                if not session:
                    continue
                
                try:
                    last_interaction = datetime.fromisoformat(session.last_interaction)
                    days_since_interaction = (current_time - last_interaction).days
                    
                    if days_since_interaction > days_old:
                        if self.delete_session(session_id):
                            cleaned_count += 1
                            logging.info(f"Cleaned up old session: {session_id} "
                                        f"({days_since_interaction} days old)")
                except (ValueError, TypeError) as e:
                    logging.error(f"Error parsing date for session {session_id}: {str(e)}")
        
        except Exception as e:
            logging.error(f"Error during session cleanup: {str(e)}")
        
        return cleaned_count
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored sessions
        
        Returns:
            Dictionary with session statistics
        """
        try:
            session_ids = self.list_sessions()
            total_count = len(session_ids)
            
            if total_count == 0:
                return {
                    "total_sessions": 0,
                    "active_sessions": 0,
                    "oldest_session": None,
                    "newest_session": None,
                    "avg_affection": 0
                }
            
            current_time = datetime.now()
            active_count = 0
            oldest_time = current_time
            newest_time = datetime(1970, 1, 1)
            total_affection = 0
            valid_affection_count = 0
            
            for session_id in session_ids:
                session = self.load_session(session_id)
                if not session:
                    continue
                
                try:
                    last_interaction = datetime.fromisoformat(session.last_interaction)
                    days_since_interaction = (current_time - last_interaction).days
                    
                    if days_since_interaction <= 30:  # Consider active if used in last 30 days
                        active_count += 1
                    
                    if last_interaction < oldest_time:
                        oldest_time = last_interaction
                    
                    if last_interaction > newest_time:
                        newest_time = last_interaction
                    
                    total_affection += session.affection_level
                    valid_affection_count += 1
                
                except (ValueError, TypeError):
                    continue
            
            avg_affection = total_affection / valid_affection_count if valid_affection_count > 0 else 0
            
            return {
                "total_sessions": total_count,
                "active_sessions": active_count,
                "oldest_session": oldest_time.isoformat() if oldest_time < current_time else None,
                "newest_session": newest_time.isoformat() if newest_time > datetime(1970, 1, 1) else None,
                "avg_affection": round(avg_affection, 2)
            }
        
        except Exception as e:
            logging.error(f"Error getting session stats: {str(e)}")
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "oldest_session": None,
                "newest_session": None,
                "avg_affection": 0,
                "error": str(e)
            }