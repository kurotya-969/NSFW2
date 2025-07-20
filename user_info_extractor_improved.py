"""
ユーザー情報抽出モジュール（改善版）
会話からユーザーの重要な個人情報を抽出し、セッションに保存する
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# グローバル変数の定義
SUDACHI_AVAILABLE = False

# 形態素解析のためのライブラリをインポート
try:
    import sudachipy
    from sudachipy import tokenizer
    from sudachipy import dictionary
    SUDACHI_AVAILABLE = True
except ImportError:
    logging.warning("SudachiPyがインストールされていません。基本的な正規表現のみで抽出を行います。")

class UserMetadata:
    """ユーザーメタデータモデル"""
    
    def __init__(self):
        self.nickname: Optional[str] = None
        self.birthday: Optional[str] = None
        self.age: Optional[int] = None
        self.likes: List[Dict[str, Any]] = []  # {category: "food", item: "ラーメン", confidence: 0.9, timestamp: "2023-01-01"}
        self.dislikes: List[Dict[str, Any]] = []
        self.location: Optional[str] = None
        self.occupation: Optional[str] = None
        self.custom_fields: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """メタデータを辞書に変換"""
        return {
            "nickname": self.nickname,
            "birthday": self.birthday,
            "age": self.age,
            "likes": self.likes,
            "dislikes": self.dislikes,
            "location": self.location,
            "occupation": self.occupation,
            "custom_fields": self.custom_fields
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserMetadata':
        """辞書からメタデータを作成"""
        metadata = cls()
        metadata.nickname = data.get("nickname")
        metadata.birthday = data.get("birthday")
        metadata.age = data.get("age")
        metadata.likes = data.get("likes", [])
        metadata.dislikes = data.get("dislikes", [])
        metadata.location = data.get("location")
        metadata.occupation = data.get("occupation")
        metadata.custom_fields = data.get("custom_fields", {})
        return metadata

class UserInfoExtractor:
    """ユーザー情報抽出クラス（改善版）"""
    
    def __init__(self):
        # SudachiPyの初期化
        self.tokenizer_obj = None
        self.tokenizer_mode = None
        
        if SUDACHI_AVAILABLE:
            try:
                self.tokenizer_obj = dictionary.Dictionary().create()
                self.tokenizer_mode = tokenizer.Tokenizer.SplitMode.C
            except Exception as e:
                logging.warning(f"SudachiPyの初期化に失敗しました: {str(e)}")
        
        # 抽出パターン定義
        self.patterns = {
            "nickname": [
                r"(?:私|僕|俺|わたし|ぼく|おれ)(?:の名前|は|が)(?:、|\s)?([^\s、。,\.]{1,10})(?:です|だよ|だ|っていう|って言う|と言います|といいます)",
                r"(?:私|僕|俺|わたし|ぼく|おれ)(?:のことは|のことを)?(?:、|\s)?([^\s、。,\.]{1,10})(?:って|と)(?:呼んで|読んで|言って)",
                r"(?:名前|なまえ)(?:は|が)(?:、|\s)?([^\s、。,\.]{1,10})(?:です|だよ|だ|っていう|って言う|と言います|といいます)"
            ],
            "birthday": [
                r"(?:誕生日|たんじょうび)(?:は|が)(?:、|\s)?(\d{1,2})(?:月)(\d{1,2})(?:日)",
                r"(\d{1,2})月(\d{1,2})日(?:が|は)(?:私|僕|俺|わたし|ぼく|おれ)の(?:誕生日|たんじょうび)"
            ],
            "age": [
                r"(?:私|僕|俺|わたし|ぼく|おれ)(?:は|が)(?:、|\s)?(\d{1,2})(?:歳|才)",
                r"(\d{1,2})(?:歳|才)(?:です|だよ|だ)"
            ],
            "likes": [
                r"(?:私|僕|俺|わたし|ぼく|おれ)(?:は|が)(?:、|\s)?([^\s、。,\.]{1,20})(?:が好き|が大好き|は好き|は大好き|が趣味|を愛して)",
                r"([^\s、。,\.]{1,20})(?:が|は)(?:大好き|好き|お気に入り)",
                r"([^\s、。,\.]{1,20})(?:をよく|をいつも)(?:食べ|飲|見|読|聴|聞|遊|楽しみ|やっ)",
                r"([^\s、。,\.]{1,20})(?:にはまって|が趣味で|を楽しんで)"
            ],
            "dislikes": [
                r"(?:私|僕|俺|わたし|ぼく|おれ)(?:は|が)(?:、|\s)?([^\s、。,\.]{1,20})(?:が嫌い|は嫌い|が苦手|は苦手)",
                r"([^\s、。,\.]{1,20})(?:が|は)(?:嫌い|苦手|ダメ|だめ)",
                r"([^\s、。,\.]{1,20})(?:は避けて|は食べられ|は飲め|は見|は読め|は聴け|は聞け)(?:ない|ません)"
            ],
            "location": [
                r"(?:私|僕|俺|わたし|ぼく|おれ)(?:は|が)(?:、|\s)?([^\s、。,\.]{1,10}(?:都|道|府|県|市|区|町|村))(?:に住んで|に住む|に在住|出身)",
                r"([^\s、。,\.]{1,10}(?:都|道|府|県|市|区|町|村))(?:に住んで|に住む|に在住|出身)(?:です|だよ|だ)",
                r"(?:住まい|住所|居住地)(?:は|が)(?:、|\s)?([^\s、。,\.]{1,10}(?:都|道|府|県|市|区|町|村))"
            ],
            "occupation": [
                r"(?:私|僕|俺|わたし|ぼく|おれ)(?:は|が)(?:、|\s)?([^\s、。,\.]{1,15}(?:会社員|学生|大学生|高校生|中学生|小学生|エンジニア|プログラマ|デザイナー|先生|教師|医者|看護師|公務員|自営業|フリーランス|主婦|主夫))",
                r"(?:職業|しょくぎょう|仕事)(?:は|が)(?:、|\s)?([^\s、。,\.]{1,15})",
                r"([^\s、。,\.]{1,15}(?:会社|企業|学校|大学|病院|事務所))(?:で働いて|に勤めて|に所属して)"
            ]
        }
        
        # カテゴリ定義（拡張版）
        self.categories = {
            "likes": [
                {"name": "food", "keywords": ["食べ物", "料理", "ラーメン", "寿司", "カレー", "パスタ", "ピザ", "肉", "魚", "野菜", "フルーツ", "デザート", "スイーツ", "チョコレート", "ケーキ", "アイス", "和食", "洋食", "中華", "イタリアン", "フレンチ", "メキシカン", "エスニック", "朝食", "昼食", "夕食", "おやつ", "お菓子"]},
                {"name": "drink", "keywords": ["飲み物", "コーヒー", "紅茶", "お茶", "ジュース", "水", "酒", "ビール", "ワイン", "カクテル", "ウイスキー", "日本酒", "焼酎", "ソーダ", "炭酸", "ミルク", "スムージー", "ドリンク"]},
                {"name": "hobby", "keywords": ["趣味", "読書", "映画", "音楽", "ゲーム", "スポーツ", "旅行", "料理", "写真", "絵", "ダンス", "歌", "釣り", "登山", "ハイキング", "キャンプ", "ガーデニング", "DIY", "手芸", "編み物", "裁縫", "工作", "プログラミング", "ブログ", "SNS", "動画", "配信", "収集", "コレクション"]},
                {"name": "anime", "keywords": ["アニメ", "漫画", "ゲーム", "キャラクター", "声優", "コスプレ", "同人", "二次元", "ライトノベル", "ジャンプ", "少年", "少女", "青年", "成人向け"]},
                {"name": "music", "keywords": ["音楽", "歌", "バンド", "アーティスト", "ライブ", "コンサート", "楽器", "ピアノ", "ギター", "ドラム", "ベース", "バイオリン", "フルート", "サックス", "トランペット", "ボーカル", "作曲", "作詞", "編曲", "DJ", "クラシック", "ジャズ", "ロック", "ポップ", "ヒップホップ", "レゲエ", "メタル", "フォーク", "民謡"]},
                {"name": "other", "keywords": []}
            ],
            "dislikes": [
                {"name": "food", "keywords": ["食べ物", "料理", "ラーメン", "寿司", "カレー", "パスタ", "ピザ", "肉", "魚", "野菜", "フルーツ", "デザート", "スイーツ", "チョコレート", "ケーキ", "アイス", "和食", "洋食", "中華", "イタリアン", "フレンチ", "メキシカン", "エスニック", "朝食", "昼食", "夕食", "おやつ", "お菓子"]},
                {"name": "situation", "keywords": ["混雑", "騒音", "暑さ", "寒さ", "雨", "雪", "風", "人混み", "待ち時間", "遅刻", "渋滞", "行列", "満員電車", "通勤", "通学", "残業", "徹夜", "早起き", "夜更かし", "二日酔い", "病気", "怪我", "痛み", "疲れ", "眠気", "空腹", "喉の渇き", "汗", "臭い", "汚れ", "埃", "カビ", "虫", "ゴキブリ", "蚊", "蜂", "蜘蛛", "ネズミ"]},
                {"name": "activity", "keywords": ["運動", "スポーツ", "勉強", "仕事", "家事", "掃除", "洗濯", "料理", "買い物", "外出", "旅行", "飛行機", "電車", "バス", "車", "自転車", "歩き", "走り", "泳ぎ", "登り", "降り", "会議", "プレゼン", "電話", "メール", "SNS", "ゲーム", "テレビ", "映画", "読書", "音楽", "ダンス", "歌", "演奏", "絵", "写真", "動画", "配信"]},
                {"name": "other", "keywords": []}
            ]
        }
    
    def extract_info(self, user_input: str) -> Dict[str, Any]:
        """
        ユーザー入力から情報を抽出（改善版）
        
        Args:
            user_input: ユーザーの入力テキスト
            
        Returns:
            抽出された情報の辞書
        """
        extracted_info = {}
        
        # 形態素解析による抽出（SudachiPyが利用可能な場合）
        if self.tokenizer_obj is not None:
            extracted_info.update(self._extract_with_morphological_analysis(user_input))
        
        # 正規表現による抽出
        for info_type, patterns in self.patterns.items():
            for pattern in patterns:
                # 誕生日は特殊なパターンなのでsearchを使用
                if info_type == "birthday":
                    matches = re.search(pattern, user_input)
                    if matches and len(matches.groups()) >= 2:
                        month = matches.group(1)
                        day = matches.group(2)
                        extracted_info[info_type] = f"{month}月{day}日"
                else:
                    # その他の情報はsearchを使用
                    matches = re.search(pattern, user_input)
                    if matches:
                        if info_type in ["likes", "dislikes"]:
                            # 好き嫌いはカテゴリ分類して保存
                            item = matches.group(1)
                            category = self._categorize_item(item, info_type)
                            confidence = self._calculate_confidence(item, info_type)
                            
                            if info_type not in extracted_info:
                                extracted_info[info_type] = []
                            
                            extracted_info[info_type].append({
                                "category": category,
                                "item": item,
                                "confidence": confidence,
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            # その他の情報はそのまま保存（最後のマッチを優先）
                            extracted_info[info_type] = matches.group(1)
        
        return extracted_info
    
    def _extract_with_morphological_analysis(self, user_input: str) -> Dict[str, Any]:
        """
        形態素解析を使用した情報抽出
        
        Args:
            user_input: ユーザーの入力テキスト
            
        Returns:
            抽出された情報の辞書
        """
        if self.tokenizer_obj is None:
            return {}
        
        extracted_info = {}
        tokens = self.tokenizer_obj.tokenize(user_input, self.tokenizer_mode)
        
        # 形態素解析結果から情報を抽出
        # 例: 好きなものと嫌いなものの抽出
        likes = []
        dislikes = []
        
        for i, token in enumerate(tokens):
            # 好きなものの抽出
            if i < len(tokens) - 1 and token.surface() != "私" and token.surface() != "僕" and token.surface() != "俺":
                next_token = tokens[i + 1]
                if next_token.surface() in ["好き", "大好き", "お気に入り", "趣味"]:
                    item = token.surface()
                    category = self._categorize_item(item, "likes")
                    confidence = self._calculate_confidence(item, "likes")
                    
                    if "likes" not in extracted_info:
                        extracted_info["likes"] = []
                    
                    extracted_info["likes"].append({
                        "category": category,
                        "item": item,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # 嫌いなものの抽出
            if i < len(tokens) - 1 and token.surface() != "私" and token.surface() != "僕" and token.surface() != "俺":
                next_token = tokens[i + 1]
                if next_token.surface() in ["嫌い", "苦手", "ダメ", "だめ"]:
                    item = token.surface()
                    category = self._categorize_item(item, "dislikes")
                    confidence = self._calculate_confidence(item, "dislikes")
                    
                    if "dislikes" not in extracted_info:
                        extracted_info["dislikes"] = []
                    
                    extracted_info["dislikes"].append({
                        "category": category,
                        "item": item,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    })
        
        return extracted_info
    
    def _categorize_item(self, item: str, info_type: str) -> str:
        """
        アイテムをカテゴリに分類（改善版）
        
        Args:
            item: 分類するアイテム
            info_type: 情報タイプ（"likes" または "dislikes"）
            
        Returns:
            カテゴリ名
        """
        if info_type not in self.categories:
            return "other"
        
        # 最も高いスコアのカテゴリを選択
        best_category = "other"
        best_score = 0
        
        for category in self.categories[info_type]:
            score = self._calculate_category_score(item, category["keywords"])
            if score > best_score:
                best_score = score
                best_category = category["name"]
        
        return best_category
    
    def _calculate_category_score(self, item: str, keywords: List[str]) -> float:
        """
        アイテムとカテゴリキーワードの類似度スコアを計算
        
        Args:
            item: 分類するアイテム
            keywords: カテゴリキーワードのリスト
            
        Returns:
            類似度スコア（0.0〜1.0）
        """
        if not keywords:
            return 0.0
        
        # 完全一致
        if item in keywords:
            return 1.0
        
        # 部分一致
        for keyword in keywords:
            if keyword in item or item in keyword:
                # 部分一致の場合、一致部分の長さに応じてスコアを計算
                match_length = min(len(keyword), len(item))
                return match_length / max(len(keyword), len(item)) * 0.8
        
        # 文字レベルの類似度（簡易版）
        best_similarity = 0.0
        for keyword in keywords:
            # 共通の文字数を計算
            common_chars = set(item) & set(keyword)
            if common_chars:
                similarity = len(common_chars) / max(len(set(item)), len(set(keyword))) * 0.5
                best_similarity = max(best_similarity, similarity)
        
        return best_similarity
    
    def _calculate_confidence(self, item: str, info_type: str) -> float:
        """
        抽出した情報の信頼度を計算
        
        Args:
            item: 抽出したアイテム
            info_type: 情報タイプ
            
        Returns:
            信頼度（0.0〜1.0）
        """
        # 基本的な信頼度
        base_confidence = 0.8
        
        # アイテムの長さに応じた調整
        if len(item) < 2:
            # 短すぎるアイテムは信頼度を下げる
            base_confidence *= 0.7
        elif len(item) > 10:
            # 長すぎるアイテムも信頼度を下げる
            base_confidence *= 0.9
        
        # カテゴリ分類の結果に応じた調整
        if info_type in ["likes", "dislikes"]:
            category = self._categorize_item(item, info_type)
            if category != "other":
                # 明確なカテゴリに分類できた場合は信頼度を上げる
                base_confidence *= 1.1
        
        # 信頼度の範囲を0.0〜1.0に制限
        return min(1.0, max(0.0, base_confidence))
    
    def update_metadata(self, metadata: UserMetadata, extracted_info: Dict[str, Any]) -> UserMetadata:
        """
        抽出した情報でメタデータを更新
        
        Args:
            metadata: 既存のメタデータ
            extracted_info: 抽出された情報
            
        Returns:
            更新されたメタデータ
        """
        # 単一値フィールドの更新
        for field in ["nickname", "birthday", "age", "location", "occupation"]:
            if field in extracted_info and extracted_info[field]:
                # 年齢は数値に変換
                if field == "age" and isinstance(extracted_info[field], str):
                    try:
                        extracted_info[field] = int(extracted_info[field])
                    except ValueError:
                        pass
                
                setattr(metadata, field, extracted_info[field])
        
        # リストフィールドの更新
        for field in ["likes", "dislikes"]:
            if field in extracted_info and extracted_info[field]:
                current_list = getattr(metadata, field)
                
                for new_item in extracted_info[field]:
                    # 既存のアイテムと重複チェック
                    duplicate = False
                    for existing_item in current_list:
                        if existing_item["item"] == new_item["item"]:
                            # 既存のアイテムを更新（信頼度が高い方を優先）
                            if new_item["confidence"] > existing_item["confidence"]:
                                existing_item["confidence"] = new_item["confidence"]
                                existing_item["category"] = new_item["category"]
                            
                            existing_item["timestamp"] = new_item["timestamp"]
                            duplicate = True
                            break
                    
                    # 重複がなければ追加
                    if not duplicate:
                        current_list.append(new_item)
        
        return metadata

def extract_and_update_user_info(user_input: str, session_id: str, session_manager) -> Tuple[bool, Dict[str, Any]]:
    """
    ユーザー入力から情報を抽出し、セッションメタデータを更新
    
    Args:
        user_input: ユーザーの入力テキスト
        session_id: セッションID
        session_manager: セッション管理オブジェクト
        
    Returns:
        Tuple of (情報が更新されたかどうか, 抽出された情報)
    """
    # セッションを取得
    session = session_manager.get_session(session_id)
    if not session:
        logging.warning(f"Session {session_id} not found for user info extraction")
        return False, {}
    
    # メタデータを取得または初期化
    if hasattr(session, "user_metadata") and session.user_metadata:
        if isinstance(session.user_metadata, dict):
            metadata = UserMetadata.from_dict(session.user_metadata)
        else:
            metadata = session.user_metadata
    else:
        metadata = UserMetadata()
    
    # 情報を抽出
    extractor = UserInfoExtractor()
    extracted_info = extractor.extract_info(user_input)
    
    # 抽出された情報がなければ終了
    if not extracted_info:
        return False, {}
    
    # メタデータを更新
    updated_metadata = extractor.update_metadata(metadata, extracted_info)
    
    # セッションに保存
    session.user_metadata = updated_metadata.to_dict()
    session_manager.save_session(session_id)
    
    logging.info(f"Updated user metadata for session {session_id}: {json.dumps(extracted_info, ensure_ascii=False)}")
    
    return True, extracted_info