"""
親密度表示システムの統合テスト
親密度ゲージ、段階変化通知、関係性詳細表示の統合テスト
"""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import gradio as gr
from affection_system import initialize_affection_system, get_session_manager, get_affection_tracker

class TestAffectionDisplayIntegration(unittest.TestCase):
    """親密度表示システムの統合テストケース"""
    
    def setUp(self):
        """テスト前の準備"""
        # テスト用のセッションディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        # アフェクションシステムを初期化
        self.session_manager, self.affection_tracker = initialize_affection_system(self.test_dir)
        # テスト用のセッションを作成
        self.session_id = self.session_manager.create_new_session()
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テスト用のディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    @patch('gradio.Slider')
    @patch('gradio.HTML')
    def test_affection_gauge_integration(self, mock_html, mock_slider):
        """親密度ゲージとUIコンポーネントの統合テスト"""
        # Gradioコンポーネントをモック化
        mock_slider_instance = MagicMock()
        mock_slider.return_value = mock_slider_instance
        
        mock_html_instance = MagicMock()
        mock_html.return_value = mock_html_instance
        
        # 親密度レベルを設定
        level = 55
        session = self.session_manager.get_session(self.session_id)
        session.affection_level = level
        self.session_manager.save_session(self.session_id)
        
        # 関係性情報を取得
        relationship_info = self.affection_tracker.get_mari_behavioral_state(level)
        
        # 親密度ゲージの更新関数（実際のアプリケーションから抽出）
        def update_session_info(session_id, previous_stage=None):
            """セッション情報表示を更新する関数（テスト用に簡略化）"""
            if not session_id:
                return None, 0, "不明", None, ""
            
            session = self.session_manager.get_session(session_id)
            if not session:
                return session_id, 0, "不明", None, ""
            
            affection_level = session.affection_level
            stage = self.affection_tracker.get_relationship_stage(affection_level)
            relationship_info = self.affection_tracker.get_mari_behavioral_state(affection_level)
            
            # 段階変化の検出
            stage_changed = previous_stage and previous_stage != stage
            notification_html = ""
            
            if stage_changed:
                # 段階変化通知のHTMLを生成
                notification_html = f"""
                <div class="stage-change-notification stage-{stage}">
                    {_get_stage_change_message(stage)}
                </div>
                """
            
            # 関係性詳細情報のHTMLを生成
            details_html = _generate_relationship_details_html(affection_level, stage, relationship_info)
            
            return session_id, affection_level, stage, notification_html, details_html
        
        # モック関数を追加
        def _get_stage_change_message(stage):
            messages = {
                "distant": "麻理の警戒心が少し和らいだようだ...",
                "cautious": "麻理はあなたに対して少し興味を持ち始めたようだ...",
                "friendly": "麻理はあなたに対して友好的な態度を見せ始めた！",
                "warm": "麻理はあなたに心を開き始めている...！",
                "close": "麻理はあなたを特別な存在として認めているようだ！"
            }
            return messages.get(stage, "麻理との関係性が変化した...")
        
        def _generate_relationship_details_html(affection_level, stage, relationship_info):
            # 関係性詳細情報のHTMLを生成（テスト用に簡略化）
            return f"""
            <div class="relationship-details">
                <h4>現在の関係性: {stage}</h4>
                <p>{relationship_info["description"]}</p>
            </div>
            """
        
        # 親密度ゲージの更新をシミュレート
        session_id, affection_level, stage, notification_html, details_html = update_session_info(self.session_id)
        
        # 結果を検証
        self.assertEqual(session_id, self.session_id)
        self.assertEqual(affection_level, level)
        self.assertEqual(stage, "friendly")
        self.assertEqual(notification_html, "")  # 段階変化なし
        self.assertIn("現在の関係性: friendly", details_html)
        self.assertIn(relationship_info["description"], details_html)
        
        # ステージ変化をシミュレート
        previous_stage = "friendly"
        self.session_manager.update_affection(self.session_id, 20)  # friendlyからwarmへ
        session_id, affection_level, stage, notification_html, details_html = update_session_info(self.session_id, previous_stage)
        
        # 結果を検証
        self.assertEqual(stage, "warm")
        self.assertIn("stage-change-notification stage-warm", notification_html)
        self.assertIn("麻理はあなたに心を開き始めている", notification_html)
    
    def test_affection_gauge_css_integration(self):
        """親密度ゲージのCSSスタイルが正しく設定されているかテスト"""
        # CSSファイルを読み込み
        with open('affection_gauge.css', 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        # 各ステージのスタイルが定義されているか確認
        stages = ["hostile", "distant", "cautious", "friendly", "warm", "close"]
        for stage in stages:
            self.assertIn(f".stage-{stage}", css_content, f"Style for stage '{stage}' not found in CSS")
        
        # 親密度ゲージのスタイルが定義されているか確認
        self.assertIn(".affection-gauge", css_content)
        self.assertIn(".slider-track", css_content)
        self.assertIn("linear-gradient", css_content)  # グラデーション
        
        # 段階変化通知のアニメーションが定義されているか確認
        self.assertIn("@keyframes stageChangeNotification", css_content)
        self.assertIn(".stage-change-notification", css_content)
        
        # 関係性詳細情報のスタイルが定義されているか確認
        self.assertIn(".relationship-details", css_content)
        self.assertIn(".next-stage-progress", css_content)
    
    def test_affection_gauge_js_integration(self):
        """親密度ゲージのJavaScriptが正しく機能するかテスト"""
        # JavaScriptファイルを読み込み
        with open('affection_gauge.js', 'r', encoding='utf-8') as f:
            js_content = f.read()
        
        # 必要な関数が定義されているか確認
        required_functions = [
            "getRelationshipStage",
            "getStageDisplayName",
            "getPointsToNextStage",
            "showStageChangeNotification",
            "updateRelationshipDetails",
            "initializeAndUpdateAffectionGauge",
            "createRelationshipStageDisplay"
        ]
        
        for func in required_functions:
            self.assertIn(f"function {func}", js_content, f"Function '{func}' not found in JavaScript")
        
        # ステージ変化の閾値が正しく設定されているか確認
        self.assertIn("affectionLevel <= 10", js_content)
        self.assertIn("affectionLevel <= 25", js_content)
        self.assertIn("affectionLevel <= 45", js_content)
        self.assertIn("affectionLevel <= 65", js_content)
        self.assertIn("affectionLevel <= 85", js_content)
        
        # 日本語表示名が定義されているか確認
        self.assertIn('"hostile": "敵対的"', js_content)
        self.assertIn('"distant": "距離を置く"', js_content)
        self.assertIn('"cautious": "慎重"', js_content)
        self.assertIn('"friendly": "友好的"', js_content)
        self.assertIn('"warm": "温かい"', js_content)
        self.assertIn('"close": "親密"', js_content)

if __name__ == "__main__":
    unittest.main()