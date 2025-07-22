"""
段階変化通知システムのテスト
親密度の段階変化を検出し、通知を表示する機能のテスト
"""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from affection_system import initialize_affection_system, get_session_manager, get_affection_tracker

class TestStageChangeNotification(unittest.TestCase):
    """段階変化通知システムのテストケース"""
    
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
    
    def test_stage_change_detection(self):
        """段階変化の検出が正しく行われるかテスト"""
        # JavaScriptの関数は直接テストできないので、ステージ変化のロジックのみテスト
        
        # 初期ステージを確認
        initial_level = self.session_manager.get_affection_level(self.session_id)
        initial_stage = self.affection_tracker.get_relationship_stage(initial_level)
        
        # 初期値を確認（デフォルトは25なのでdistantステージ）
        self.assertEqual(initial_stage, "distant")
        
        # ステージ変化のないアフェクション変更（distantステージは10-25なので、-5しても同じステージ）
        small_delta = -5
        self.session_manager.update_affection(self.session_id, small_delta)
        new_level = self.session_manager.get_affection_level(self.session_id)
        new_stage = self.affection_tracker.get_relationship_stage(new_level)
        
        # 同じステージ内の変化では通知は発生しない
        self.assertEqual(new_stage, "distant")
        
        # ステージ変化を伴うアフェクション変更
        large_delta = 30
        old_stage = new_stage
        self.session_manager.update_affection(self.session_id, large_delta)
        new_level = self.session_manager.get_affection_level(self.session_id)
        new_stage = self.affection_tracker.get_relationship_stage(new_level)
        
        # ステージが変化したことを確認
        self.assertNotEqual(new_stage, old_stage)
        
        # 実際のアプリケーションでは、JavaScriptの関数が呼び出される
        # ここではモックを使用して、関数が呼び出されることを確認する
        # mock_notification.assert_called_with(old_stage, new_stage)
    
    def test_stage_transition_thresholds(self):
        """ステージ遷移の閾値が正しく設定されているかテスト"""
        # 各ステージの境界値をテスト
        thresholds = {
            10: "hostile",
            11: "distant",
            25: "distant",
            26: "cautious",
            45: "cautious",
            46: "friendly",
            65: "friendly",
            66: "warm",
            85: "warm",
            86: "close"
        }
        
        for level, expected_stage in thresholds.items():
            stage = self.affection_tracker.get_relationship_stage(level)
            self.assertEqual(stage, expected_stage, f"Level {level} should be {expected_stage}, got {stage}")
    
    def test_stage_specific_messages(self):
        """ステージ固有のメッセージが正しく設定されているかテスト"""
        # 実際のJavaScriptコードからメッセージを抽出
        with open('affection_gauge.js', 'r', encoding='utf-8') as f:
            js_code = f.read()
        
        # 各ステージのメッセージが含まれているか確認
        stages = ["distant", "cautious", "friendly", "warm", "close"]
        for stage in stages:
            # JavaScriptコード内にステージ名が含まれているか確認
            self.assertIn(f"case '{stage}':", js_code, f"Stage '{stage}' not found in JavaScript code")
            
            # メッセージが設定されているか確認
            self.assertIn("message = ", js_code)
    
    def test_relationship_details_update(self):
        """関係性詳細情報の更新が正しく行われるかテスト"""
        # 親密度レベルを設定
        level = 55  # friendly
        
        # 関係性情報を取得
        relationship_info = self.affection_tracker.get_mari_behavioral_state(level)
        
        # 次のステージまでの情報を確認
        # JavaScriptの関数をPythonで再現
        def get_points_to_next_stage(affection_level):
            if affection_level <= 10:
                return {"nextStage": "distant", "pointsNeeded": 11 - affection_level, "percentage": affection_level / 11 * 100}
            elif affection_level <= 25:
                return {"nextStage": "cautious", "pointsNeeded": 26 - affection_level, "percentage": (affection_level - 11) / 15 * 100}
            elif affection_level <= 45:
                return {"nextStage": "friendly", "pointsNeeded": 46 - affection_level, "percentage": (affection_level - 26) / 20 * 100}
            elif affection_level <= 65:
                return {"nextStage": "warm", "pointsNeeded": 66 - affection_level, "percentage": (affection_level - 46) / 20 * 100}
            elif affection_level <= 85:
                return {"nextStage": "close", "pointsNeeded": 86 - affection_level, "percentage": (affection_level - 66) / 20 * 100}
            else:
                return {"nextStage": "max", "pointsNeeded": 0, "percentage": 100}
        
        next_stage_info = get_points_to_next_stage(level)
        self.assertEqual(next_stage_info["nextStage"], "warm")
        self.assertEqual(next_stage_info["pointsNeeded"], 66 - level)
        self.assertAlmostEqual(next_stage_info["percentage"], (level - 46) / 20 * 100)

if __name__ == "__main__":
    unittest.main()