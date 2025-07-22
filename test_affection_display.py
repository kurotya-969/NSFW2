"""
親密度表示システムのテスト
親密度ゲージの視覚的表示と段階変化通知のテスト
"""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from affection_system import initialize_affection_system, get_session_manager, get_affection_tracker

class TestAffectionDisplay(unittest.TestCase):
    """親密度表示システムのテストケース"""
    
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
    
    def test_affection_level_change(self):
        """親密度レベルの変更が正しく反映されるかテスト"""
        # 初期値を確認
        initial_level = self.session_manager.get_affection_level(self.session_id)
        self.assertEqual(initial_level, 25)  # デフォルト値は25
        
        # 親密度を更新
        delta = 10
        self.session_manager.update_affection(self.session_id, delta)
        
        # 更新後の値を確認
        new_level = self.session_manager.get_affection_level(self.session_id)
        self.assertEqual(new_level, initial_level + delta)
        
        # セッションから直接値を確認
        session = self.session_manager.get_session(self.session_id)
        self.assertEqual(session.affection_level, initial_level + delta)
    
    def test_relationship_stage_transition(self):
        """関係性ステージの遷移が正しく行われるかテスト"""
        # 初期ステージを確認
        initial_level = self.session_manager.get_affection_level(self.session_id)
        initial_stage = self.affection_tracker.get_relationship_stage(initial_level)
        self.assertEqual(initial_stage, "distant")  # 初期値25はdistantステージ
        
        # cautious (26-45) に更新
        self.session_manager.update_affection(self.session_id, 10)
        new_level = self.session_manager.get_affection_level(self.session_id)
        new_stage = self.affection_tracker.get_relationship_stage(new_level)
        self.assertEqual(new_stage, "cautious")
        
        # friendly (46-65) に更新
        self.session_manager.update_affection(self.session_id, 20)
        new_level = self.session_manager.get_affection_level(self.session_id)
        new_stage = self.affection_tracker.get_relationship_stage(new_level)
        self.assertEqual(new_stage, "friendly")
        
        # warm (66-85) に更新
        self.session_manager.update_affection(self.session_id, 20)
        new_level = self.session_manager.get_affection_level(self.session_id)
        new_stage = self.affection_tracker.get_relationship_stage(new_level)
        self.assertEqual(new_stage, "warm")
        
        # close (86-100) に更新
        self.session_manager.update_affection(self.session_id, 20)
        new_level = self.session_manager.get_affection_level(self.session_id)
        new_stage = self.affection_tracker.get_relationship_stage(new_level)
        self.assertEqual(new_stage, "close")
    
    def test_relationship_description(self):
        """関係性の説明が正しく取得できるかテスト"""
        # 各ステージの説明を確認
        stages = {
            5: "hostile",
            20: "distant",
            35: "cautious",
            55: "friendly",
            75: "warm",
            95: "close"
        }
        
        for level, expected_stage in stages.items():
            # 親密度を設定
            session = self.session_manager.get_session(self.session_id)
            session.affection_level = level
            self.session_manager.save_session(self.session_id)
            
            # ステージを確認
            stage = self.affection_tracker.get_relationship_stage(level)
            self.assertEqual(stage, expected_stage)
            
            # 説明を取得
            description = self.affection_tracker.get_relationship_description(level)
            self.assertIsNotNone(description)
            self.assertNotEqual(description, "")
            self.assertNotEqual(description, "関係性が不明確")
    
    def test_mari_behavioral_state(self):
        """麻理の行動状態が正しく取得できるかテスト"""
        # 親密度レベルを設定
        level = 55  # friendly
        
        # 行動状態を取得
        behavior = self.affection_tracker.get_mari_behavioral_state(level)
        
        # 基本的な構造を確認
        self.assertIn("stage", behavior)
        self.assertEqual(behavior["stage"], "friendly")
        
        self.assertIn("stage_traits", behavior)
        traits = behavior["stage_traits"]
        
        # 必要な特性が含まれているか確認
        expected_traits = ["openness", "trust", "vulnerability", "communication_style", 
                          "emotional_expression", "characteristic_phrases", "relationship_dynamics"]
        for trait in expected_traits:
            self.assertIn(trait, traits)
        
        # 基本的な人格特性が含まれているか確認
        self.assertIn("core_personality", behavior)
        self.assertIn("speech_patterns", behavior)
        self.assertIn("first_person", behavior)
        
        # 説明が含まれているか確認
        self.assertIn("description", behavior)
        self.assertIsNotNone(behavior["description"])
        self.assertNotEqual(behavior["description"], "")

if __name__ == "__main__":
    unittest.main()