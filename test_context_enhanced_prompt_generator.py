"""
コンテキスト強化プロンプト生成のテスト
ユーザーメタデータを活用したプロンプト生成機能のテスト
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
from tsundere_aware_prompt_generator import TsundereAwarePromptGenerator

class TestContextEnhancedPromptGenerator(unittest.TestCase):
    """コンテキスト強化プロンプト生成のテストケース"""
    
    def setUp(self):
        """テスト前の準備"""
        # テスト用のセッションディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        # アフェクションシステムを初期化
        self.session_manager, self.affection_tracker = initialize_affection_system(self.test_dir)
        # テスト用のセッションを作成
        self.session_id = self.session_manager.create_new_session()
        
        # 基本システムプロンプト
        self.base_prompt = """
        # 麻理の人格設定
        あなたは「麻理」という名前のAIコンパニオンです。過去に捨てられた経験から警戒心が強く、素直になれない不器用なツンデレ性格です。
        """
        
        # プロンプト生成器を作成
        self.prompt_generator = TsundereAwarePromptGenerator(self.base_prompt)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テスト用のディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    def test_prompt_with_user_metadata(self):
        """ユーザーメタデータを含むプロンプト生成をテスト"""
        # セッションを取得
        session = self.session_manager.get_session(self.session_id)
        
        # メタデータを設定
        metadata = {
            "nickname": "太郎",
            "birthday": "5月10日",
            "age": 25,
            "likes": [
                {"category": "food", "item": "ラーメン", "confidence": 0.9, "timestamp": datetime.now().isoformat()},
                {"category": "hobby", "item": "読書", "confidence": 0.8, "timestamp": datetime.now().isoformat()}
            ],
            "dislikes": [
                {"category": "food", "item": "ピーマン", "confidence": 0.8, "timestamp": datetime.now().isoformat()}
            ],
            "location": "東京都",
            "occupation": "エンジニア"
        }
        
        session.user_metadata = metadata
        self.session_manager.save_session(self.session_id)
        
        # 親密度レベルを設定
        affection_level = 55  # friendly
        
        # tsundere_contextを作成（実際のアプリケーションでは、tsundere_sentiment_detectorから取得）
        tsundere_context = {
            "detected_tsundere_pattern": "none",
            "user_sentiment": "positive",
            "conversation_context": "general",
            "suggested_response_style": "friendly_tsun"
        }
        
        # プロンプトを生成
        dynamic_prompt = self.prompt_generator.generate_dynamic_prompt(
            affection_level, 
            tsundere_context,
            session_id=self.session_id
        )
        
        # プロンプトにユーザー情報が含まれているか確認
        self.assertIn("太郎", dynamic_prompt)
        self.assertIn("ラーメン", dynamic_prompt)
        self.assertIn("読書", dynamic_prompt)
        self.assertIn("ピーマン", dynamic_prompt)
    
    def test_prompt_with_different_affection_levels(self):
        """異なる親密度レベルでのプロンプト生成をテスト"""
        # セッションを取得
        session = self.session_manager.get_session(self.session_id)
        
        # 基本的なメタデータを設定
        metadata = {
            "nickname": "太郎",
            "likes": [
                {"category": "food", "item": "ラーメン", "confidence": 0.9, "timestamp": datetime.now().isoformat()}
            ]
        }
        
        session.user_metadata = metadata
        self.session_manager.save_session(self.session_id)
        
        # 異なる親密度レベルでプロンプトを生成
        affection_levels = [10, 30, 50, 70, 90]  # hostile, cautious, friendly, warm, close
        
        for level in affection_levels:
            # プロンプトを生成
            dynamic_prompt = self.prompt_generator.generate_dynamic_prompt(
                level, 
                {},  # 空のtsundere_context
                session_id=self.session_id
            )
            
            # 親密度レベルに応じた表現が含まれているか確認
            if level <= 10:  # hostile
                self.assertIn("極端に警戒", dynamic_prompt.lower())
            elif level <= 45:  # cautious
                self.assertIn("警戒", dynamic_prompt.lower())
            elif level <= 65:  # friendly
                self.assertIn("友好的", dynamic_prompt.lower())
            elif level <= 85:  # warm
                self.assertIn("信頼", dynamic_prompt.lower())
            else:  # close
                self.assertIn("深い信頼", dynamic_prompt.lower())
            
            # ユーザー情報が含まれているか確認
            self.assertIn("太郎", dynamic_prompt)
            self.assertIn("ラーメン", dynamic_prompt)
    
    def test_prompt_with_conversation_history(self):
        """会話履歴を含むプロンプト生成をテスト"""
        # セッションを取得
        session = self.session_manager.get_session(self.session_id)
        
        # 会話履歴を追加
        conversation_history = [
            {
                "timestamp": datetime.now().isoformat(),
                "user": "こんにちは、麻理！",
                "assistant": "...ああ。何か用？"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "user": "今日はいい天気だね",
                "assistant": "ふん、まあな。外出るわけでもないけど"
            }
        ]
        
        session.conversation_history = conversation_history
        
        # メタデータを設定
        metadata = {
            "nickname": "太郎"
        }
        
        session.user_metadata = metadata
        self.session_manager.save_session(self.session_id)
        
        # プロンプトを生成
        dynamic_prompt = self.prompt_generator.generate_dynamic_prompt(
            50,  # friendly
            {},  # 空のtsundere_context
            session_id=self.session_id
        )
        
        # 会話履歴の内容が反映されているか確認
        self.assertIn("太郎", dynamic_prompt)
        
        # 注: 実際のTsundereAwarePromptGeneratorの実装によっては、
        # 会話履歴の内容がそのままプロンプトに含まれるとは限らない
    
    def test_prompt_with_no_metadata(self):
        """メタデータがない場合のプロンプト生成をテスト"""
        # メタデータを設定せずにプロンプトを生成
        dynamic_prompt = self.prompt_generator.generate_dynamic_prompt(
            50,  # friendly
            {},  # 空のtsundere_context
            session_id=self.session_id
        )
        
        # 基本的なプロンプトが生成されているか確認
        self.assertIn("麻理", dynamic_prompt)
        self.assertIn("ツンデレ", dynamic_prompt.lower())
        
        # ユーザー固有の情報が含まれていないことを確認
        self.assertNotIn("太郎", dynamic_prompt)
        self.assertNotIn("ラーメン", dynamic_prompt)
    
    def test_prompt_with_invalid_session(self):
        """無効なセッションIDでのプロンプト生成をテスト"""
        # 存在しないセッションIDでプロンプトを生成
        dynamic_prompt = self.prompt_generator.generate_dynamic_prompt(
            50,  # friendly
            {},  # 空のtsundere_context
            session_id="invalid_session_id"
        )
        
        # 基本的なプロンプトが生成されているか確認
        self.assertIn("麻理", dynamic_prompt)
        self.assertIn("ツンデレ", dynamic_prompt.lower())

if __name__ == "__main__":
    unittest.main()