"""
好感度を手動で設定するためのユーティリティスクリプト
"""

import os
import sys
import json
from pathlib import Path

def set_affection_level(session_id, new_level):
    """
    指定されたセッションの好感度レベルを設定する
    
    Args:
        session_id: セッションID
        new_level: 新しい好感度レベル (0-100)
    
    Returns:
        bool: 成功したかどうか
    """
    # セッションディレクトリのパスを取得
    sessions_dir = Path("sessions")
    
    # セッションファイルのパスを構築
    session_file = sessions_dir / f"{session_id}.json"
    
    # ファイルが存在するか確認
    if not session_file.exists():
        print(f"エラー: セッションファイル {session_file} が見つかりません")
        return False
    
    try:
        # セッションファイルを読み込む
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        
        # 現在の好感度を表示
        old_level = session_data.get("affection_level", 0)
        print(f"現在の好感度: {old_level}")
        
        # 好感度を更新
        session_data["affection_level"] = max(0, min(100, int(new_level)))
        
        # 更新したデータを保存
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"好感度を {old_level} から {session_data['affection_level']} に更新しました")
        return True
    
    except Exception as e:
        print(f"エラー: {str(e)}")
        return False

def list_sessions():
    """利用可能なセッションを一覧表示する"""
    sessions_dir = Path("sessions")
    
    if not sessions_dir.exists():
        print("セッションディレクトリが見つかりません")
        return
    
    session_files = list(sessions_dir.glob("*.json"))
    
    if not session_files:
        print("セッションファイルが見つかりません")
        return
    
    print("利用可能なセッション:")
    for session_file in session_files:
        session_id = session_file.stem
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                affection = session_data.get("affection_level", "不明")
                last_interaction = session_data.get("last_interaction", "不明")
                print(f"セッションID: {session_id}, 好感度: {affection}, 最終更新: {last_interaction}")
        except:
            print(f"セッションID: {session_id}, データ読み込みエラー")

if __name__ == "__main__":
    # コマンドライン引数をチェック
    if len(sys.argv) == 1 or sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print("使用方法:")
        print("  python set_affection.py list                  - 利用可能なセッションを一覧表示")
        print("  python set_affection.py <session_id> <level>  - 指定したセッションの好感度を設定")
        print("例:")
        print("  python set_affection.py list")
        print("  python set_affection.py 123e4567-e89b-12d3-a456-426614174000 85")
        sys.exit(0)
    
    # セッション一覧を表示
    if sys.argv[1] == "list":
        list_sessions()
        sys.exit(0)
    
    # 好感度を設定
    if len(sys.argv) >= 3:
        session_id = sys.argv[1]
        try:
            new_level = int(sys.argv[2])
            if new_level < 0 or new_level > 100:
                print("好感度は0から100の間で指定してください")
                sys.exit(1)
            
            set_affection_level(session_id, new_level)
        except ValueError:
            print("好感度は整数で指定してください")
            sys.exit(1)
    else:
        print("セッションIDと好感度を指定してください")
        sys.exit(1)