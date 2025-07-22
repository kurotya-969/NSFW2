"""
使用統計システムのテスト
セッション情報と使用統計の収集・分析機能のテスト
"""

import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from usage_statistics import UsageStatistics, initialize_usage_statistics, get_usage_statistics

class TestUsageStatistics(unittest.TestCase):
    """使用統計システムのテストケース"""
    
    def setUp(self):
        """テスト前の準備"""
        # テスト用のディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        # 統計モジュールを初期化
        self.stats = UsageStatistics(self.test_dir)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テスト用のディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    def test_stats_file_creation(self):
        """統計ファイルの作成をテスト"""
        # 統計ファイルのパスを取得
        stats_file = os.path.join(self.test_dir, "usage_stats.json")
        
        # ファイルが存在するか確認
        self.assertTrue(os.path.exists(stats_file))
        
        # ファイルを読み込み
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        
        # 基本的な構造を確認
        self.assertIn("daily_users", stats_data)
        self.assertIn("hourly_distribution", stats_data)
        self.assertIn("total_sessions", stats_data)
        self.assertIn("total_conversations", stats_data)
        self.assertIn("avg_session_duration", stats_data)
        self.assertIn("avg_conversation_turns", stats_data)
        self.assertIn("last_updated", stats_data)
    
    def test_record_session_activity(self):
        """セッションアクティビティの記録をテスト"""
        # テスト用のセッションID
        session_id = "test_session_123"
        
        # セッション開始を記録
        self.stats.record_session_activity(session_id, "start")
        
        # 統計を読み込み
        stats_data = self.stats._load_stats()
        
        # 総セッション数が増加しているか確認
        self.assertEqual(stats_data["total_sessions"], 1)
        
        # 現在の日付が記録されているか確認
        today = datetime.now().strftime("%Y-%m-%d")
        self.assertIn(today, stats_data["daily_users"])
        self.assertEqual(stats_data["daily_users"][today]["count"], 1)
        self.assertIn(session_id, stats_data["daily_users"][today]["sessions"])
        
        # 現在の時間帯が記録されているか確認
        hour = str(datetime.now().hour)
        self.assertGreaterEqual(stats_data["hourly_distribution"][hour], 1)
        
        # 会話を記録
        self.stats.record_session_activity(session_id, "interaction")
        
        # 統計を再読み込み
        stats_data = self.stats._load_stats()
        
        # 総会話数が増加しているか確認
        self.assertEqual(stats_data["total_conversations"], 1)
        
        # 同じセッションIDでの記録で、ユニークユーザー数は変わらないことを確認
        self.assertEqual(stats_data["daily_users"][today]["count"], 1)
        
        # 別のセッションIDで記録
        self.stats.record_session_activity("another_session", "start")
        
        # 統計を再読み込み
        stats_data = self.stats._load_stats()
        
        # ユニークユーザー数が増加しているか確認
        self.assertEqual(stats_data["daily_users"][today]["count"], 2)
    
    def test_update_session_metrics(self):
        """セッションメトリクスの更新をテスト"""
        # テスト用のセッションID
        session_id = "test_session_123"
        
        # セッション開始を記録
        self.stats.record_session_activity(session_id, "start")
        
        # セッションメトリクスを更新
        duration_seconds = 600  # 10分
        conversation_turns = 15
        
        self.stats.update_session_metrics(session_id, duration_seconds, conversation_turns)
        
        # 統計を読み込み
        stats_data = self.stats._load_stats()
        
        # 平均セッション時間が更新されているか確認
        self.assertEqual(stats_data["avg_session_duration"], duration_seconds)
        
        # 平均会話ターン数が更新されているか確認
        self.assertEqual(stats_data["avg_conversation_turns"], conversation_turns)
        
        # 別のセッションメトリクスを追加
        self.stats.record_session_activity("another_session", "start")
        self.stats.update_session_metrics("another_session", 1200, 25)
        
        # 統計を再読み込み
        stats_data = self.stats._load_stats()
        
        # 平均値が正しく計算されているか確認
        expected_avg_duration = (600 + 1200) / 2
        expected_avg_turns = (15 + 25) / 2
        
        self.assertEqual(stats_data["avg_session_duration"], expected_avg_duration)
        self.assertEqual(stats_data["avg_conversation_turns"], expected_avg_turns)
    
    def test_get_daily_users(self):
        """日別ユーザー数の取得をテスト"""
        # 過去の日付でユーザーを記録
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        two_days_ago = today - timedelta(days=2)
        
        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        two_days_ago_str = two_days_ago.strftime("%Y-%m-%d")
        
        # 統計データを直接操作
        stats_data = self.stats._load_stats()
        stats_data["daily_users"] = {
            today_str: {"count": 5, "sessions": ["s1", "s2", "s3", "s4", "s5"]},
            yesterday_str: {"count": 3, "sessions": ["s6", "s7", "s8"]},
            two_days_ago_str: {"count": 2, "sessions": ["s9", "s10"]}
        }
        self.stats._save_stats(stats_data)
        
        # 過去7日間のデータを取得
        daily_users = self.stats.get_daily_users(7)
        
        # 結果を確認
        self.assertEqual(len(daily_users), 3)
        self.assertEqual(daily_users[today_str], 5)
        self.assertEqual(daily_users[yesterday_str], 3)
        self.assertEqual(daily_users[two_days_ago_str], 2)
        
        # 過去1日間のデータを取得
        daily_users = self.stats.get_daily_users(1)
        
        # 結果を確認
        self.assertEqual(len(daily_users), 1)
        self.assertEqual(daily_users[today_str], 5)
    
    def test_get_hourly_distribution(self):
        """時間帯別分布の取得をテスト"""
        # 時間帯別データを直接操作
        stats_data = self.stats._load_stats()
        stats_data["hourly_distribution"] = {
            "9": 10,
            "10": 15,
            "11": 20,
            "12": 25,
            "13": 30,
            "14": 20,
            "15": 15,
            "16": 10,
            "17": 5
        }
        self.stats._save_stats(stats_data)
        
        # 時間帯別分布を取得
        hourly_distribution = self.stats.get_hourly_distribution()
        
        # 結果を確認
        self.assertEqual(hourly_distribution["9"], 10)
        self.assertEqual(hourly_distribution["13"], 30)
        self.assertEqual(hourly_distribution["17"], 5)
    
    def test_get_summary_statistics(self):
        """サマリー統計の取得をテスト"""
        # 統計データを直接操作
        stats_data = self.stats._load_stats()
        
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        stats_data["daily_users"] = {
            today: {"count": 5, "sessions": ["s1", "s2", "s3", "s4", "s5"]},
            yesterday: {"count": 3, "sessions": ["s6", "s7", "s8"]}
        }
        stats_data["hourly_distribution"] = {
            "9": 10,
            "10": 15,
            "13": 30,
            "17": 5
        }
        stats_data["total_sessions"] = 100
        stats_data["total_conversations"] = 500
        stats_data["avg_session_duration"] = 600
        stats_data["avg_conversation_turns"] = 15
        
        self.stats._save_stats(stats_data)
        
        # サマリー統計を取得
        summary = self.stats.get_summary_statistics()
        
        # 結果を確認
        self.assertEqual(summary["total_sessions"], 100)
        self.assertEqual(summary["total_conversations"], 500)
        self.assertEqual(summary["avg_session_duration"], 600)
        self.assertEqual(summary["avg_conversation_turns"], 15)
        self.assertEqual(summary["total_unique_users"], 8)  # 5 + 3
        self.assertEqual(summary["active_days"], 2)
        self.assertEqual(summary["avg_users_per_day"], 4)  # (5 + 3) / 2
        self.assertEqual(summary["most_active_hour"], "13")
    
    def test_get_monthly_report(self):
        """月次レポートの取得をテスト"""
        # 現在の年月を取得
        now = datetime.now()
        year = now.year
        month = now.month
        
        # 月内の日付を生成
        days_in_month = [f"{year}-{month:02d}-{day:02d}" for day in range(1, 29)]
        
        # 統計データを直接操作
        stats_data = self.stats._load_stats()
        stats_data["daily_users"] = {}
        
        # 月内の各日にランダムなユーザー数を設定
        for i, day in enumerate(days_in_month):
            user_count = (i % 5) + 1  # 1〜5のユーザー数
            stats_data["daily_users"][day] = {
                "count": user_count,
                "sessions": [f"s{j}" for j in range(user_count)]
            }
        
        self.stats._save_stats(stats_data)
        
        # 月次レポートを取得
        report = self.stats.get_monthly_report(year, month)
        
        # 結果を確認
        self.assertEqual(report["year"], year)
        self.assertEqual(report["month"], month)
        self.assertGreaterEqual(len(report["daily_users"]), 28)
        
        # 総ユーザー数と平均を確認
        total_users = sum(data for data in report["daily_users"].values())
        self.assertEqual(report["total_users"], total_users)
        
        active_days = sum(1 for count in report["daily_users"].values() if count > 0)
        self.assertEqual(report["active_days"], active_days)
        
        if active_days > 0:
            avg_users = total_users / active_days
            self.assertEqual(report["avg_users_per_day"], avg_users)
    
    def test_export_data_csv(self):
        """CSVデータエクスポートをテスト"""
        # 過去の日付でユーザーを記録
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)]
        
        # 統計データを直接操作
        stats_data = self.stats._load_stats()
        stats_data["daily_users"] = {}
        
        for i, date in enumerate(dates):
            user_count = (i % 5) + 1  # 1〜5のユーザー数
            stats_data["daily_users"][date] = {
                "count": user_count,
                "sessions": [f"s{j}" for j in range(user_count)]
            }
        
        self.stats._save_stats(stats_data)
        
        # CSVデータをエクスポート
        csv_data = self.stats.export_data_csv(dates[-1], dates[0])
        
        # 結果を確認
        lines = csv_data.strip().split("\n")
        self.assertEqual(lines[0], "date,unique_users")
        self.assertEqual(len(lines), 11)  # ヘッダー + 10日分
        
        # 各行のフォーマットを確認
        for i in range(1, len(lines)):
            parts = lines[i].split(",")
            self.assertEqual(len(parts), 2)
            self.assertTrue(parts[0] in dates)
            self.assertTrue(parts[1].isdigit())

if __name__ == "__main__":
    unittest.main()