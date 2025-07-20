"""
管理者インターフェースモジュール
ユーザー統計情報の表示と分析のための管理者向けインターフェース
"""

import gradio as gr
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import io
import base64

# 統計情報モジュールのインポート
from usage_statistics import get_usage_statistics
from affection_system import get_session_manager

def create_admin_interface():
    """
    管理者インターフェースを作成
    
    Returns:
        Gradioインターフェース
    """
    with gr.Blocks(title="麻理AI管理者パネル", theme=gr.themes.Soft()) as admin_interface:
        gr.Markdown("# 麻理AI管理者パネル")
        
        with gr.Tab("利用統計"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 利用統計")
                    
                    # 期間選択
                    period_selector = gr.Dropdown(
                        choices=["過去7日間", "過去30日間", "過去90日間", "全期間"],
                        value="過去30日間",
                        label="期間"
                    )
                    
                    # 統計情報更新ボタン
                    refresh_btn = gr.Button("統計情報を更新", variant="primary")
                    
                    # 基本統計情報
                    summary_stats = gr.JSON(label="基本統計情報")
                    
                with gr.Column(scale=2):
                    # グラフ表示
                    daily_users_chart = gr.Plot(label="日別ユニークユーザー数")
                    hourly_distribution_chart = gr.Plot(label="時間帯別アクティビティ")
            
            with gr.Row():
                # データエクスポート
                export_start_date = gr.Textbox(
                    label="開始日 (YYYY-MM-DD)",
                    value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                )
                export_end_date = gr.Textbox(
                    label="終了日 (YYYY-MM-DD)",
                    value=datetime.now().strftime("%Y-%m-%d")
                )
                export_btn = gr.Button("CSVエクスポート")
                export_result = gr.File(label="エクスポート結果")
        
        with gr.Tab("ユーザー情報"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## ユーザー情報")
                    
                    # セッションID検索
                    session_id_input = gr.Textbox(label="セッションID")
                    search_btn = gr.Button("検索")
                    
                    # セッション一覧
                    session_list = gr.Dropdown(label="アクティブセッション")
                    load_sessions_btn = gr.Button("セッション一覧を更新")
                
                with gr.Column(scale=2):
                    # ユーザー情報表示
                    user_info = gr.JSON(label="ユーザー情報")
                    
                    # 会話履歴表示
                    conversation_history = gr.Dataframe(
                        headers=["時間", "ユーザー", "アシスタント"],
                        label="会話履歴"
                    )
        
        # 関数定義
        def update_statistics(period):
            """
            統計情報を更新
            
            Args:
                period: 表示期間
                
            Returns:
                Tuple of (summary_stats, daily_users_chart, hourly_distribution_chart)
            """
            try:
                stats = get_usage_statistics()
                if not stats:
                    return {"error": "統計情報モジュールが初期化されていません"}, None, None
                
                # 期間に応じた日数を設定
                days = {
                    "過去7日間": 7,
                    "過去30日間": 30,
                    "過去90日間": 90,
                    "全期間": 365  # 最大1年
                }.get(period, 30)
                
                # 基本統計情報を取得
                summary = stats.get_summary_statistics()
                
                # 日別ユーザー数を取得
                daily_users = stats.get_daily_users(days)
                
                # 時間帯別分布を取得
                hourly_distribution = stats.get_hourly_distribution()
                
                # 日別ユーザー数のグラフを作成
                daily_fig = create_daily_users_chart(daily_users)
                
                # 時間帯別分布のグラフを作成
                hourly_fig = create_hourly_distribution_chart(hourly_distribution)
                
                return summary, daily_fig, hourly_fig
                
            except Exception as e:
                logging.error(f"統計情報の更新に失敗しました: {str(e)}")
                return {"error": f"統計情報の更新に失敗しました: {str(e)}"}, None, None
        
        def create_daily_users_chart(daily_users):
            """
            日別ユーザー数のグラフを作成
            
            Args:
                daily_users: 日別ユーザー数の辞書
                
            Returns:
                matplotlib図
            """
            fig, ax = plt.subplots(figsize=(10, 6))
            
            dates = list(daily_users.keys())
            counts = list(daily_users.values())
            
            # 日付でソート
            sorted_data = sorted(zip(dates, counts), key=lambda x: x[0])
            sorted_dates = [item[0] for item in sorted_data]
            sorted_counts = [item[1] for item in sorted_data]
            
            ax.bar(sorted_dates, sorted_counts, color='skyblue')
            ax.set_xlabel('日付')
            ax.set_ylabel('ユニークユーザー数')
            ax.set_title('日別ユニークユーザー数')
            
            # X軸の日付ラベルを調整
            if len(sorted_dates) > 10:
                plt.xticks(rotation=45, ha='right')
                # 日付ラベルを間引く
                step = max(1, len(sorted_dates) // 10)
                for i, label in enumerate(ax.xaxis.get_ticklabels()):
                    if i % step != 0:
                        label.set_visible(False)
            
            plt.tight_layout()
            return fig
        
        def create_hourly_distribution_chart(hourly_distribution):
            """
            時間帯別分布のグラフを作成
            
            Args:
                hourly_distribution: 時間帯別分布の辞書
                
            Returns:
                matplotlib図
            """
            fig, ax = plt.subplots(figsize=(10, 6))
            
            hours = [int(h) for h in hourly_distribution.keys()]
            counts = list(hourly_distribution.values())
            
            # 時間でソート
            sorted_data = sorted(zip(hours, counts))
            sorted_hours = [f"{item[0]}時" for item in sorted_data]
            sorted_counts = [item[1] for item in sorted_data]
            
            ax.bar(sorted_hours, sorted_counts, color='lightgreen')
            ax.set_xlabel('時間帯')
            ax.set_ylabel('アクティビティ数')
            ax.set_title('時間帯別アクティビティ分布')
            
            plt.xticks(rotation=45, ha='right')
            # 時間ラベルを間引く
            step = max(1, len(sorted_hours) // 12)
            for i, label in enumerate(ax.xaxis.get_ticklabels()):
                if i % step != 0:
                    label.set_visible(False)
            
            plt.tight_layout()
            return fig
        
        def export_data(start_date, end_date):
            """
            データをCSVとしてエクスポート
            
            Args:
                start_date: 開始日
                end_date: 終了日
                
            Returns:
                CSVファイル
            """
            try:
                stats = get_usage_statistics()
                if not stats:
                    return None
                
                csv_data = stats.export_data_csv(start_date, end_date)
                
                # CSVファイルを作成
                filename = f"mari_stats_{start_date}_to_{end_date}.csv"
                temp_path = os.path.join(os.path.dirname(__file__), filename)
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(csv_data)
                
                return temp_path
                
            except Exception as e:
                logging.error(f"データのエクスポートに失敗しました: {str(e)}")
                return None
        
        def load_sessions():
            """
            アクティブなセッション一覧を読み込む
            
            Returns:
                セッションIDのリスト
            """
            try:
                session_manager = get_session_manager()
                if not session_manager:
                    return []
                
                session_ids = session_manager.list_sessions()
                
                # セッション情報を取得して、最終アクセス日時でソート
                session_info = []
                for session_id in session_ids:
                    session = session_manager.get_session(session_id)
                    if session:
                        try:
                            last_interaction = datetime.fromisoformat(session.last_interaction)
                            session_info.append((session_id, last_interaction))
                        except (ValueError, TypeError):
                            session_info.append((session_id, datetime.min))
                
                # 最終アクセス日時の新しい順にソート
                sorted_sessions = sorted(session_info, key=lambda x: x[1], reverse=True)
                
                # セッションIDのみを返す
                return [session_id for session_id, _ in sorted_sessions]
                
            except Exception as e:
                logging.error(f"セッション一覧の読み込みに失敗しました: {str(e)}")
                return []
        
        def load_user_info(session_id):
            """
            ユーザー情報を読み込む
            
            Args:
                session_id: セッションID
                
            Returns:
                Tuple of (user_info, conversation_history)
            """
            try:
                session_manager = get_session_manager()
                if not session_manager or not session_id:
                    return {}, []
                
                session = session_manager.get_session(session_id)
                if not session:
                    return {"error": f"セッション {session_id} が見つかりません"}, []
                
                # ユーザー情報を取得
                user_info = {
                    "session_id": session_id,
                    "affection_level": session.affection_level,
                    "session_start_time": session.session_start_time,
                    "last_interaction": session.last_interaction
                }
                
                # ユーザーメタデータがあれば追加
                if hasattr(session, "user_metadata") and session.user_metadata:
                    user_info["user_metadata"] = session.user_metadata
                
                # 会話履歴を取得
                conversation_data = []
                for entry in session.conversation_history:
                    if 'timestamp' in entry and 'user' in entry and 'assistant' in entry:
                        conversation_data.append([
                            entry['timestamp'],
                            entry['user'],
                            entry['assistant']
                        ])
                
                return user_info, conversation_data
                
            except Exception as e:
                logging.error(f"ユーザー情報の読み込みに失敗しました: {str(e)}")
                return {"error": f"ユーザー情報の読み込みに失敗しました: {str(e)}"}, []
        
        # イベントハンドラの設定
        refresh_btn.click(
            update_statistics,
            inputs=[period_selector],
            outputs=[summary_stats, daily_users_chart, hourly_distribution_chart]
        )
        
        export_btn.click(
            export_data,
            inputs=[export_start_date, export_end_date],
            outputs=[export_result]
        )
        
        load_sessions_btn.click(
            load_sessions,
            inputs=[],
            outputs=[session_list]
        )
        
        search_btn.click(
            load_user_info,
            inputs=[session_id_input],
            outputs=[user_info, conversation_history]
        )
        
        session_list.change(
            load_user_info,
            inputs=[session_list],
            outputs=[user_info, conversation_history]
        )
        
        # 初期データ読み込み
        admin_interface.load(
            update_statistics,
            inputs=[period_selector],
            outputs=[summary_stats, daily_users_chart, hourly_distribution_chart]
        )
        
        admin_interface.load(
            load_sessions,
            inputs=[],
            outputs=[session_list]
        )
    
    return admin_interface

# 管理者認証
def check_admin_auth(username, password):
    """
    管理者認証を行う
    
    Args:
        username: ユーザー名
        password: パスワード
        
    Returns:
        認証成功の場合はTrue、失敗の場合はFalse
    """
    # 環境変数から管理者認証情報を取得
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_password = os.environ.get("ADMIN_PASSWORD", "password")
    
    return username == admin_username and password == admin_password