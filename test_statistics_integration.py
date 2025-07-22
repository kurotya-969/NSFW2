"""
統計システムの統合テスト
データ収集、分析、管理者インターフェースの統合テスト
"""

import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import gradio as gr
from usage_statistics import UsageStatistics, initialize_usage_statistics, get_usage_statistics
from affection_system import initialize_affection_system, get_session_manager, get_affection_tracker
from admin_interface import create_admin_interface

class TestStatisticsIntegration(unittest.TestCase):
    """統計システムの統合テストケース"""
    
    def setUp(self):
        """テスト前の準備"""
        # テスト用のディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        # 統計モジュールを初期化
        self.stats = initialize_usage_statistics(self.test_dir)
        # アフェクションシステムを初期化
        self.session_manager, self.affection_tracker = initialize_affection_system(self.test_dir)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テスト用のディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    def test_session_activity_tracking(self):
        """セッションアクティビティの追跡をテスト"""
        # テスト用のセッションを作成
        session_id = self.session_manager.create_new_session()
        
        # セッション開始を記録
        self.stats.record_session_activity(session_id, "start")
        
        # 会話を記録
        for i in range(5):
            user_input = f"テストメッセージ {i}"
            assistant_response = f"テスト応答 {i}"
            
            # 会話履歴を更新
            self.session_manager.update_conversation_history(session_id, user_input, assistant_response)
            
            # 会話アクティビティを記録
            self.stats.record_session_activity(session_id, "interaction")
        
        # セッションメトリクスを更新
        duration_seconds = 300  # 5分
        conversation_turns = 5
        self.stats.update_session_metrics(session_id, duration_seconds, conversation_turns)
        
        # 統計を確認
        summary = self.stats.get_summary_statistics()
        
        # 結果を確認
        self.assertGreaterEqual(summary["total_sessions"], 1)
        self.assertGreaterEqual(summary["total_conversations"], 5)
        self.assertEqual(summary["avg_session_duration"], duration_seconds)
        self.assertEqual(summary["avg_conversation_turns"], conversation_turns)
    
    def test_multiple_session_tracking(self):
        """複数セッションの追跡をテスト"""
        # 複数のセッションを作成
        session_ids = [self.session_manager.create_new_session() for _ in range(3)]
        
        # 各セッションのアクティビティを記録
        for i, session_id in enumerate(session_ids):
            # セッション開始を記録
            self.stats.record_session_activity(session_id, "start")
            
            # 会話を記録
            for j in range(i + 1):  # セッションごとに異なる会話数
                user_input = f"セッション {i} メッセージ {j}"
                assistant_response = f"セッション {i} 応答 {j}"
                
                # 会話履歴を更新
                self.session_manager.update_conversation_history(session_id, user_input, assistant_response)
                
                # 会話アクティビティを記録
                self.stats.record_session_activity(session_id, "interaction")
            
            # セッションメトリクスを更新
            duration_seconds = (i + 1) * 100  # セッションごとに異なる時間
            conversation_turns = i + 1
            self.stats.update_session_metrics(session_id, duration_seconds, conversation_turns)
        
        # 統計を確認
        summary = self.stats.get_summary_statistics()
        
        # 結果を確認
        self.assertGreaterEqual(summary["total_sessions"], 3)
        self.assertGreaterEqual(summary["total_conversations"], 6)  # 0+1+2+3=6
        
        # 平均値を確認
        expected_avg_duration = (100 + 200 + 300) / 3
        expected_avg_turns = (1 + 2 + 3) / 3
        
        self.assertEqual(summary["avg_session_duration"], expected_avg_duration)
        self.assertEqual(summary["avg_conversation_turns"], expected_avg_turns)
        
        # 日別ユーザー数を確認
        today = datetime.now().strftime("%Y-%m-%d")
        daily_users = self.stats.get_daily_users(1)
        self.assertEqual(daily_users[today], 3)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_chart_generation(self, mock_savefig, mock_figure):
        """チャート生成をテスト"""
        # matplotlibをモック化
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_figure.return_value = mock_fig
        
        # テスト用のデータを作成
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        
        # 統計データを直接操作
        stats_data = self.stats._load_stats()
        stats_data["daily_users"] = {}
        
        for i, date in enumerate(dates):
            user_count = i + 1
            stats_data["daily_users"][date] = {
                "count": user_count,
                "sessions": [f"s{j}" for j in range(user_count)]
            }
        
        self.stats._save_stats(stats_data)
        
        # 管理者インターフェースの関数を抽出
        from admin_interface import create_daily_users_chart, create_hourly_distribution_chart
        
        # 日別ユーザー数のチャートを作成
        daily_users = self.stats.get_daily_users(7)
        daily_chart = create_daily_users_chart(daily_users)
        
        # 時間帯別分布のチャートを作成
        hourly_distribution = self.stats.get_hourly_distribution()
        hourly_chart = create_hourly_distribution_chart(hourly_distribution)
        
        # チャートが作成されたことを確認
        self.assertIsNotNone(daily_chart)
        self.assertIsNotNone(hourly_chart)
    
    def test_csv_export_integration(self):
        """CSVエクスポートの統合テストをテスト"""
        # テスト用のデータを作成
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)]
        
        # 統計データを直接操作
        stats_data = self.stats._load_stats()
        stats_data["daily_users"] = {}
        
        for i, date in enumerate(dates):
            user_count = (i % 5) + 1
            stats_data["daily_users"][date] = {
                "count": user_count,
                "sessions": [f"s{j}" for j in range(user_count)]
            }
        
        self.stats._save_stats(stats_data)
        
        # 管理者インターフェースの関数を抽出
        from admin_interface import export_data
        
        # データをエクスポート
        start_date = dates[-1]
        end_date = dates[0]
        csv_file = export_data(start_date, end_date)
        
        # ファイルが作成されたことを確認
        self.assertIsNotNone(csv_file)
        self.assertTrue(os.path.exists(csv_file))
        
        # ファイルの内容を確認
        with open(csv_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split("\n")
        self.assertEqual(lines[0], "date,unique_users")
        self.assertEqual(len(lines), 11)  # ヘッダー + 10日分
    
    def test_user_info_integration(self):
        """ユーザー情報の統合テストをテスト"""
        # テスト用のセッションを作成
        session_id = self.session_manager.create_new_session()
        
        # メタデータを設定
        session = self.session_manager.get_session(session_id)
        session.user_metadata = {
            "nickname": "太郎",
            "birthday": "5月10日",
            "age": 25,
            "likes": [
                {"category": "food", "item": "ラーメン", "confidence": 0.9, "timestamp": datetime.now().isoformat()}
            ],
            "dislikes": [
                {"category": "food", "item": "ピーマン", "confidence": 0.8, "timestamp": datetime.now().isoformat()}
            ]
        }
        
        # 会話履歴を追加
        for i in range(3):
            user_input = f"テストメッセージ {i}"
            assistant_response = f"テスト応答 {i}"
            self.session_manager.update_conversation_history(session_id, user_input, assistant_response)
            
            # 会話アクティビティを記録
            self.stats.record_session_activity(session_id, "interaction")
        
        # セッションを保存
        self.session_manager.save_session(session_id)
        
        # 管理者インターフェースの関数を抽出
        from admin_interface import load_user_info
        
        # ユーザー情報を読み込み
        user_info, conversation_data = load_user_info(session_id)
        
        # 結果を確認
        self.assertIsNotNone(user_info)
        self.assertEqual(user_info["session_id"], session_id)
        self.assertIn("user_metadata", user_info)
        self.assertEqual(user_info["user_metadata"]["nickname"], "太郎")
        
        # 会話履歴を確認
        self.assertIsNotNone(conversation_data)
        self.assertEqual(len(conversation_data), 3)
        
        # 統計情報を確認
        summary = self.stats.get_summary_statistics()
        self.assertGreaterEqual(summary["total_conversations"], 3)

if __name__ == "__main__":
    unittest.main()