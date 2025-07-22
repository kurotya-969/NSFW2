"""
管理者インターフェースのテスト
統計データの視覚化と管理者機能のテスト
"""

import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import gradio as gr
from admin_interface import create_admin_interface, check_admin_auth
from usage_statistics import UsageStatistics, initialize_usage_statistics
from affection_system import initialize_affection_system

class TestAdminInterface(unittest.TestCase):
    """管理者インターフェースのテストケース"""
    
    def setUp(self):
        """テスト前の準備"""
        # テスト用のディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        # 統計モジュールを初期化
        self.stats = UsageStatistics(self.test_dir)
        # アフェクションシステムを初期化
        self.session_manager, self.affection_tracker = initialize_affection_system(self.test_dir)
        
        # テスト用のセッションを作成
        self.session_id = self.session_manager.create_new_session()
        
        # テスト用の統計データを作成
        self._create_test_stats_data()
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テスト用のディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    def _create_test_stats_data(self):
        """テスト用の統計データを作成"""
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
        
        # 時間帯別データを設定
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
        
        # その他の統計データを設定
        stats_data["total_sessions"] = 100
        stats_data["total_conversations"] = 500
        stats_data["avg_session_duration"] = 600
        stats_data["avg_conversation_turns"] = 15
        
        self.stats._save_stats(stats_data)
    
    @patch('gradio.Blocks')
    @patch('gradio.Tab')
    @patch('gradio.Row')
    @patch('gradio.Column')
    @patch('gradio.Markdown')
    @patch('gradio.Dropdown')
    @patch('gradio.Button')
    @patch('gradio.JSON')
    @patch('gradio.Plot')
    @patch('gradio.Textbox')
    @patch('gradio.File')
    @patch('gradio.Dataframe')
    def test_admin_interface_creation(self, mock_dataframe, mock_file, mock_textbox, 
                                     mock_plot, mock_json, mock_button, mock_dropdown, 
                                     mock_markdown, mock_column, mock_row, mock_tab, mock_blocks):
        """管理者インターフェースの作成をテスト"""
        # Gradioコンポーネントをモック化
        mock_blocks_instance = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_blocks_instance
        
        mock_tab_instance = MagicMock()
        mock_tab.return_value.__enter__.return_value = mock_tab_instance
        
        mock_row_instance = MagicMock()
        mock_row.return_value.__enter__.return_value = mock_row_instance
        
        mock_column_instance = MagicMock()
        mock_column.return_value.__enter__.return_value = mock_column_instance
        
        # グローバル変数をパッチ
        with patch('admin_interface.get_usage_statistics', return_value=self.stats), \
             patch('admin_interface.get_session_manager', return_value=self.session_manager):
            
            # 管理者インターフェースを作成
            admin_interface = create_admin_interface()
            
            # Blocksが作成されたことを確認
            mock_blocks.assert_called_once()
            
            # タブが作成されたことを確認
            self.assertEqual(mock_tab.call_count, 2)  # 利用統計タブとユーザー情報タブ
            
            # 各種コンポーネントが作成されたことを確認
            mock_dropdown.assert_called()  # 期間選択ドロップダウン
            mock_button.assert_called()  # 更新ボタンなど
            mock_json.assert_called()  # 統計情報表示
            mock_plot.assert_called()  # グラフ表示
    
    @patch('matplotlib.pyplot.figure')
    def test_update_statistics(self, mock_figure):
        """統計情報の更新機能をテスト"""
        # matplotlibをモック化
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # グローバル変数をパッチ
        with patch('admin_interface.get_usage_statistics', return_value=self.stats):
            # 管理者インターフェースのupdate_statistics関数を抽出
            from admin_interface import update_statistics
            
            # 統計情報を更新
            summary, daily_fig, hourly_fig = update_statistics("過去7日間")
            
            # 結果を確認
            self.assertIsNotNone(summary)
            self.assertIsNotNone(daily_fig)
            self.assertIsNotNone(hourly_fig)
            
            # サマリー統計の内容を確認
            self.assertIn("total_sessions", summary)
            self.assertIn("total_conversations", summary)
            self.assertIn("avg_session_duration", summary)
            self.assertIn("avg_conversation_turns", summary)
            self.assertIn("total_unique_users", summary)
            self.assertIn("active_days", summary)
            self.assertIn("avg_users_per_day", summary)
            self.assertIn("most_active_hour", summary)
    
    def test_export_data(self):
        """データエクスポート機能をテスト"""
        # グローバル変数をパッチ
        with patch('admin_interface.get_usage_statistics', return_value=self.stats):
            # 管理者インターフェースのexport_data関数を抽出
            from admin_interface import export_data
            
            # 開始日と終了日を設定
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            # データをエクスポート
            csv_file = export_data(start_date, end_date)
            
            # 結果を確認
            self.assertIsNotNone(csv_file)
            self.assertTrue(os.path.exists(csv_file))
            
            # ファイルの内容を確認
            with open(csv_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.strip().split("\n")
            self.assertEqual(lines[0], "date,unique_users")
            self.assertGreaterEqual(len(lines), 2)  # ヘッダー + 少なくとも1日分
    
    def test_load_sessions(self):
        """セッション一覧の読み込み機能をテスト"""
        # グローバル変数をパッチ
        with patch('admin_interface.get_session_manager', return_value=self.session_manager):
            # 管理者インターフェースのload_sessions関数を抽出
            from admin_interface import load_sessions
            
            # セッション一覧を読み込み
            sessions = load_sessions()
            
            # 結果を確認
            self.assertIsNotNone(sessions)
            self.assertIsInstance(sessions, list)
            self.assertGreaterEqual(len(sessions), 1)
            self.assertIn(self.session_id, sessions)
    
    def test_load_user_info(self):
        """ユーザー情報の読み込み機能をテスト"""
        # セッションにメタデータを追加
        session = self.session_manager.get_session(self.session_id)
        session.user_metadata = {
            "nickname": "太郎",
            "birthday": "5月10日",
            "age": 25
        }
        self.session_manager.save_session(self.session_id)
        
        # 会話履歴を追加
        self.session_manager.update_conversation_history(
            self.session_id,
            "こんにちは、麻理！",
            "...ああ。何か用？"
        )
        
        # グローバル変数をパッチ
        with patch('admin_interface.get_session_manager', return_value=self.session_manager):
            # 管理者インターフェースのload_user_info関数を抽出
            from admin_interface import load_user_info
            
            # ユーザー情報を読み込み
            user_info, conversation_data = load_user_info(self.session_id)
            
            # 結果を確認
            self.assertIsNotNone(user_info)
            self.assertIsInstance(user_info, dict)
            self.assertIn("session_id", user_info)
            self.assertEqual(user_info["session_id"], self.session_id)
            self.assertIn("affection_level", user_info)
            self.assertIn("user_metadata", user_info)
            self.assertEqual(user_info["user_metadata"]["nickname"], "太郎")
            
            # 会話履歴を確認
            self.assertIsNotNone(conversation_data)
            self.assertIsInstance(conversation_data, list)
            self.assertGreaterEqual(len(conversation_data), 1)
            self.assertEqual(conversation_data[0][1], "こんにちは、麻理！")
            self.assertEqual(conversation_data[0][2], "...ああ。何か用？")
    
    def test_admin_auth(self):
        """管理者認証機能をテスト"""
        # 環境変数をパッチ
        with patch.dict(os.environ, {"ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "password"}):
            # 正しい認証情報でテスト
            self.assertTrue(check_admin_auth("admin", "password"))
            
            # 誤った認証情報でテスト
            self.assertFalse(check_admin_auth("admin", "wrong_password"))
            self.assertFalse(check_admin_auth("wrong_user", "password"))
            self.assertFalse(check_admin_auth("", ""))

if __name__ == "__main__":
    unittest.main()