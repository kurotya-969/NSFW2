"""
セッションストレージとユーザーメタデータの統合テスト
ユーザーメタデータの保存と読み込み機能のテスト
"""

import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock
from user_info_extractor import UserMetadata
from affection_system import initialize_affection_system, get_session_manager, get_affection_tracker
from session_storage import SessionStorage, UserSession

class TestSessionStorageWithMetadata(unittest.TestCase):
    """セッションストレージとユーザーメタデータの統合テストケース"""
    
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
    
    def test_save_and_load_metadata(self):
        """メタデータの保存と読み込みをテスト"""
        # セッションを取得
        session = self.session_manager.get_session(self.session_id)
        self.assertIsNotNone(session)
        
        # メタデータを作成
        metadata = {
            "nickname": "太郎",
            "birthday": "5月10日",
            "age": 25,
            "likes": [
                {"category": "food", "item": "ラーメン", "confidence": 0.9, "timestamp": datetime.now().isoformat()}
            ],
            "dislikes": [
                {"category": "food", "item": "ピーマン", "confidence": 0.8, "timestamp": datetime.now().isoformat()}
            ],
            "location": "東京都",
            "occupation": "エンジニア",
            "custom_fields": {"favorite_color": "青"}
        }
        
        # メタデータをセッションに設定
        session.user_metadata = metadata
        
        # セッションを保存
        saved = self.session_manager.save_session(self.session_id)
        self.assertTrue(saved)
        
        # セッションをメモリから削除
        del self.session_manager.current_sessions[self.session_id]
        
        # セッションを再読み込み
        loaded_session = self.session_manager.get_session(self.session_id)
        self.assertIsNotNone(loaded_session)
        
        # メタデータが正しく読み込まれたか確認
        self.assertTrue(hasattr(loaded_session, "user_metadata"))
        self.assertIsNotNone(loaded_session.user_metadata)
        
        loaded_metadata = loaded_session.user_metadata
        self.assertEqual(loaded_metadata["nickname"], "太郎")
        self.assertEqual(loaded_metadata["birthday"], "5月10日")
        self.assertEqual(loaded_metadata["age"], 25)
        self.assertEqual(loaded_metadata["location"], "東京都")
        self.assertEqual(loaded_metadata["occupation"], "エンジニア")
        self.assertEqual(loaded_metadata["custom_fields"]["favorite_color"], "青")
        
        self.assertGreaterEqual(len(loaded_metadata["likes"]), 1)
        self.assertEqual(loaded_metadata["likes"][0]["item"], "ラーメン")
        self.assertEqual(loaded_metadata["likes"][0]["category"], "food")
        
        self.assertGreaterEqual(len(loaded_metadata["dislikes"]), 1)
        self.assertEqual(loaded_metadata["dislikes"][0]["item"], "ピーマン")
        self.assertEqual(loaded_metadata["dislikes"][0]["category"], "food")
    
    def test_metadata_update_and_merge(self):
        """メタデータの更新と結合をテスト"""
        # セッションを取得
        session = self.session_manager.get_session(self.session_id)
        self.assertIsNotNone(session)
        
        # 初期メタデータを設定
        initial_metadata = {
            "nickname": "太郎",
            "birthday": "5月10日",
            "likes": [
                {"category": "food", "item": "ラーメン", "confidence": 0.9, "timestamp": datetime.now().isoformat()}
            ],
            "dislikes": []
        }
        
        session.user_metadata = initial_metadata
        self.session_manager.save_session(self.session_id)
        
        # 新しい情報を追加
        session = self.session_manager.get_session(self.session_id)
        metadata = session.user_metadata
        
        # 既存の情報を更新
        metadata["age"] = 25
        
        # 新しい好きなものを追加
        metadata["likes"].append({
            "category": "hobby", 
            "item": "読書", 
            "confidence": 0.8, 
            "timestamp": datetime.now().isoformat()
        })
        
        # 嫌いなものを追加
        metadata["dislikes"].append({
            "category": "food", 
            "item": "ピーマン", 
            "confidence": 0.8, 
            "timestamp": datetime.now().isoformat()
        })
        
        # セッションを保存
        session.user_metadata = metadata
        self.session_manager.save_session(self.session_id)
        
        # セッションをメモリから削除
        del self.session_manager.current_sessions[self.session_id]
        
        # セッションを再読み込み
        loaded_session = self.session_manager.get_session(self.session_id)
        loaded_metadata = loaded_session.user_metadata
        
        # 更新された情報を確認
        self.assertEqual(loaded_metadata["nickname"], "太郎")
        self.assertEqual(loaded_metadata["birthday"], "5月10日")
        self.assertEqual(loaded_metadata["age"], 25)
        
        # 好きなものリストを確認
        self.assertEqual(len(loaded_metadata["likes"]), 2)
        items = [like["item"] for like in loaded_metadata["likes"]]
        self.assertIn("ラーメン", items)
        self.assertIn("読書", items)
        
        # 嫌いなものリストを確認
        self.assertEqual(len(loaded_metadata["dislikes"]), 1)
        self.assertEqual(loaded_metadata["dislikes"][0]["item"], "ピーマン")
    
    def test_metadata_serialization(self):
        """メタデータのシリアライズとデシリアライズをテスト"""
        # UserMetadataオブジェクトを作成
        metadata = UserMetadata()
        metadata.nickname = "太郎"
        metadata.birthday = "5月10日"
        metadata.age = 25
        metadata.location = "東京都"
        metadata.occupation = "エンジニア"
        
        metadata.likes.append({
            "category": "food", 
            "item": "ラーメン", 
            "confidence": 0.9, 
            "timestamp": datetime.now().isoformat()
        })
        
        metadata.dislikes.append({
            "category": "food", 
            "item": "ピーマン", 
            "confidence": 0.8, 
            "timestamp": datetime.now().isoformat()
        })
        
        metadata.custom_fields["favorite_color"] = "青"
        
        # 辞書に変換
        metadata_dict = metadata.to_dict()
        
        # 辞書からオブジェクトに戻す
        restored_metadata = UserMetadata.from_dict(metadata_dict)
        
        # 変換前後で値が一致するか確認
        self.assertEqual(restored_metadata.nickname, metadata.nickname)
        self.assertEqual(restored_metadata.birthday, metadata.birthday)
        self.assertEqual(restored_metadata.age, metadata.age)
        self.assertEqual(restored_metadata.location, metadata.location)
        self.assertEqual(restored_metadata.occupation, metadata.occupation)
        
        self.assertEqual(len(restored_metadata.likes), len(metadata.likes))
        self.assertEqual(restored_metadata.likes[0]["item"], metadata.likes[0]["item"])
        self.assertEqual(restored_metadata.likes[0]["category"], metadata.likes[0]["category"])
        
        self.assertEqual(len(restored_metadata.dislikes), len(metadata.dislikes))
        self.assertEqual(restored_metadata.dislikes[0]["item"], metadata.dislikes[0]["item"])
        self.assertEqual(restored_metadata.dislikes[0]["category"], metadata.dislikes[0]["category"])
        
        self.assertEqual(restored_metadata.custom_fields["favorite_color"], metadata.custom_fields["favorite_color"])
    
    def test_session_file_format(self):
        """セッションファイルのフォーマットをテスト"""
        # セッションを取得
        session = self.session_manager.get_session(self.session_id)
        
        # メタデータを設定
        metadata = {
            "nickname": "太郎",
            "birthday": "5月10日",
            "age": 25
        }
        
        session.user_metadata = metadata
        
        # セッションを保存
        self.session_manager.save_session(self.session_id)
        
        # セッションファイルのパスを取得
        session_file = os.path.join(self.test_dir, f"{self.session_id}.json")
        
        # ファイルが存在するか確認
        self.assertTrue(os.path.exists(session_file))
        
        # ファイルを読み込み
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # ファイル形式を確認
        self.assertIn("user_id", session_data)
        self.assertEqual(session_data["user_id"], self.session_id)
        
        self.assertIn("affection_level", session_data)
        self.assertIn("conversation_history", session_data)
        self.assertIn("session_start_time", session_data)
        self.assertIn("last_interaction", session_data)
        
        # メタデータが含まれているか確認
        self.assertIn("user_metadata", session_data)
        self.assertEqual(session_data["user_metadata"]["nickname"], "太郎")
        self.assertEqual(session_data["user_metadata"]["birthday"], "5月10日")
        self.assertEqual(session_data["user_metadata"]["age"], 25)

if __name__ == "__main__":
    unittest.main()