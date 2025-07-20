"""
Usage Statistics Module for Mari AI Chat
Collects and analyzes user session data for statistical insights
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import calendar

class UsageStatistics:
    """Handles collection and analysis of usage statistics"""
    
    def __init__(self, storage_dir: str = "sessions"):
        """
        Initialize usage statistics module
        
        Args:
            storage_dir: Directory path for session storage
        """
        self.storage_dir = storage_dir
        self.stats_file = os.path.join(storage_dir, "usage_stats.json")
        self._ensure_stats_file()
    
    def _ensure_stats_file(self) -> None:
        """Create statistics file if it doesn't exist"""
        if not os.path.exists(self.stats_file):
            try:
                # Create empty stats structure
                initial_stats = {
                    "daily_users": {},
                    "hourly_distribution": {str(i): 0 for i in range(24)},
                    "total_sessions": 0,
                    "total_conversations": 0,
                    "avg_session_duration": 0,
                    "avg_conversation_turns": 0,
                    "last_updated": datetime.now().isoformat()
                }
                
                with open(self.stats_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_stats, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Created usage statistics file: {self.stats_file}")
            except OSError as e:
                logging.error(f"Failed to create statistics file: {str(e)}")
    
    def record_session_activity(self, session_id: str, activity_type: str = "interaction") -> None:
        """
        Record session activity for statistics
        
        Args:
            session_id: User session ID
            activity_type: Type of activity ("start", "interaction", "end")
        """
        try:
            # Get current date and hour
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            hour_str = str(now.hour)
            
            # Load current stats
            stats = self._load_stats()
            
            # Update daily users
            if date_str not in stats["daily_users"]:
                stats["daily_users"][date_str] = {"count": 0, "sessions": []}
            
            # Add session to daily users if not already counted
            if session_id not in stats["daily_users"][date_str]["sessions"]:
                stats["daily_users"][date_str]["sessions"].append(session_id)
                stats["daily_users"][date_str]["count"] = len(stats["daily_users"][date_str]["sessions"])
            
            # Update hourly distribution
            stats["hourly_distribution"][hour_str] = stats["hourly_distribution"].get(hour_str, 0) + 1
            
            # Update total counts
            if activity_type == "start":
                stats["total_sessions"] += 1
            elif activity_type == "interaction":
                stats["total_conversations"] += 1
            
            # Update last updated timestamp
            stats["last_updated"] = now.isoformat()
            
            # Save updated stats
            self._save_stats(stats)
            
        except Exception as e:
            logging.error(f"Failed to record session activity: {str(e)}")
    
    def update_session_metrics(self, session_id: str, duration_seconds: float, conversation_turns: int) -> None:
        """
        Update session metrics for statistics
        
        Args:
            session_id: User session ID
            duration_seconds: Session duration in seconds
            conversation_turns: Number of conversation turns
        """
        try:
            # Load current stats
            stats = self._load_stats()
            
            # Update average session duration
            current_total = stats["avg_session_duration"] * (stats["total_sessions"] - 1)
            stats["avg_session_duration"] = (current_total + duration_seconds) / stats["total_sessions"]
            
            # Update average conversation turns
            current_total_turns = stats["avg_conversation_turns"] * (stats["total_sessions"] - 1)
            stats["avg_conversation_turns"] = (current_total_turns + conversation_turns) / stats["total_sessions"]
            
            # Save updated stats
            self._save_stats(stats)
            
        except Exception as e:
            logging.error(f"Failed to update session metrics: {str(e)}")
    
    def _load_stats(self) -> Dict[str, Any]:
        """
        Load statistics from file
        
        Returns:
            Dictionary of usage statistics
        """
        try:
            if not os.path.exists(self.stats_file):
                self._ensure_stats_file()
                
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except (json.JSONDecodeError, IOError, OSError) as e:
            logging.error(f"Failed to load statistics: {str(e)}")
            # Return empty stats structure
            return {
                "daily_users": {},
                "hourly_distribution": {str(i): 0 for i in range(24)},
                "total_sessions": 0,
                "total_conversations": 0,
                "avg_session_duration": 0,
                "avg_conversation_turns": 0,
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_stats(self, stats: Dict[str, Any]) -> None:
        """
        Save statistics to file
        
        Args:
            stats: Dictionary of usage statistics
        """
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
                
        except (IOError, OSError) as e:
            logging.error(f"Failed to save statistics: {str(e)}")
    
    def get_daily_users(self, days: int = 30) -> Dict[str, int]:
        """
        Get daily unique users for the specified number of days
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary mapping dates to unique user counts
        """
        try:
            stats = self._load_stats()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Filter daily users within date range
            daily_users = {}
            for date_str, data in stats["daily_users"].items():
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    if start_date <= date <= end_date:
                        daily_users[date_str] = data["count"]
                except ValueError:
                    continue
            
            return daily_users
            
        except Exception as e:
            logging.error(f"Failed to get daily users: {str(e)}")
            return {}
    
    def get_hourly_distribution(self) -> Dict[str, int]:
        """
        Get hourly activity distribution
        
        Returns:
            Dictionary mapping hours to activity counts
        """
        try:
            stats = self._load_stats()
            return stats["hourly_distribution"]
            
        except Exception as e:
            logging.error(f"Failed to get hourly distribution: {str(e)}")
            return {str(i): 0 for i in range(24)}
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics
        
        Returns:
            Dictionary of summary statistics
        """
        try:
            stats = self._load_stats()
            
            # Calculate additional metrics
            total_users = sum(data["count"] for data in stats["daily_users"].values())
            active_days = len(stats["daily_users"])
            avg_users_per_day = total_users / active_days if active_days > 0 else 0
            
            # Get most active hour
            most_active_hour = max(stats["hourly_distribution"].items(), key=lambda x: x[1])[0]
            
            return {
                "total_sessions": stats["total_sessions"],
                "total_conversations": stats["total_conversations"],
                "avg_session_duration": stats["avg_session_duration"],
                "avg_conversation_turns": stats["avg_conversation_turns"],
                "total_unique_users": total_users,
                "active_days": active_days,
                "avg_users_per_day": avg_users_per_day,
                "most_active_hour": most_active_hour,
                "last_updated": stats["last_updated"]
            }
            
        except Exception as e:
            logging.error(f"Failed to get summary statistics: {str(e)}")
            return {
                "total_sessions": 0,
                "total_conversations": 0,
                "avg_session_duration": 0,
                "avg_conversation_turns": 0,
                "total_unique_users": 0,
                "active_days": 0,
                "avg_users_per_day": 0,
                "most_active_hour": "0",
                "last_updated": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_monthly_report(self, year: int = None, month: int = None) -> Dict[str, Any]:
        """
        Get monthly usage report
        
        Args:
            year: Year for report (default: current year)
            month: Month for report (default: current month)
            
        Returns:
            Dictionary with monthly report data
        """
        try:
            # Default to current year and month
            if year is None or month is None:
                now = datetime.now()
                year = now.year
                month = now.month
            
            # Load stats
            stats = self._load_stats()
            
            # Get days in month
            days_in_month = calendar.monthrange(year, month)[1]
            
            # Initialize daily data
            daily_data = {f"{year}-{month:02d}-{day:02d}": 0 for day in range(1, days_in_month + 1)}
            
            # Fill in actual data
            for date_str, data in stats["daily_users"].items():
                if date_str.startswith(f"{year}-{month:02d}-"):
                    daily_data[date_str] = data["count"]
            
            # Calculate monthly totals
            total_users = sum(daily_data.values())
            active_days = sum(1 for count in daily_data.values() if count > 0)
            avg_users_per_day = total_users / active_days if active_days > 0 else 0
            
            return {
                "year": year,
                "month": month,
                "month_name": calendar.month_name[month],
                "daily_users": daily_data,
                "total_users": total_users,
                "active_days": active_days,
                "avg_users_per_day": avg_users_per_day
            }
            
        except Exception as e:
            logging.error(f"Failed to get monthly report: {str(e)}")
            return {
                "year": year or datetime.now().year,
                "month": month or datetime.now().month,
                "month_name": calendar.month_name[month or datetime.now().month],
                "daily_users": {},
                "total_users": 0,
                "active_days": 0,
                "avg_users_per_day": 0,
                "error": str(e)
            }
    
    def export_data_csv(self, start_date: str = None, end_date: str = None) -> str:
        """
        Export statistics data as CSV
        
        Args:
            start_date: Start date in format "YYYY-MM-DD" (default: 30 days ago)
            end_date: End date in format "YYYY-MM-DD" (default: today)
            
        Returns:
            CSV data as string
        """
        try:
            # Default date range
            if not end_date:
                end_date_obj = datetime.now()
                end_date = end_date_obj.strftime("%Y-%m-%d")
            else:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                
            if not start_date:
                start_date_obj = end_date_obj - timedelta(days=30)
                start_date = start_date_obj.strftime("%Y-%m-%d")
            else:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            
            # Load stats
            stats = self._load_stats()
            
            # Prepare CSV data
            csv_lines = ["date,unique_users"]
            
            # Generate date range
            current_date = start_date_obj
            while current_date <= end_date_obj:
                date_str = current_date.strftime("%Y-%m-%d")
                user_count = stats["daily_users"].get(date_str, {}).get("count", 0)
                csv_lines.append(f"{date_str},{user_count}")
                current_date += timedelta(days=1)
            
            return "\n".join(csv_lines)
            
        except Exception as e:
            logging.error(f"Failed to export data as CSV: {str(e)}")
            return "Error exporting data"

# Global instance
usage_stats = None

def initialize_usage_statistics(storage_dir: str = "sessions") -> UsageStatistics:
    """
    Initialize the usage statistics module
    
    Args:
        storage_dir: Directory path for session storage
        
    Returns:
        UsageStatistics instance
    """
    global usage_stats
    usage_stats = UsageStatistics(storage_dir)
    logging.info("Usage statistics module initialized")
    return usage_stats

def get_usage_statistics() -> Optional[UsageStatistics]:
    """Get the global usage statistics instance"""
    return usage_stats