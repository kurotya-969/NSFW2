# ベースイメージとしてPython 3.11-slimを選択
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txtを先にコピーして、依存関係のレイヤーをキャッシュさせる
COPY requirements.txt .

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# Renderが環境変数PORTでポートを指定してくるので、それに合わせる
EXPOSE $PORT

# UvicornでFastAPIアプリケーションを直接実行
# シェル形式でCMDを記述し、$PORT環境変数を展開させる
CMD uvicorn app:app --host 0.0.0.0 --port $PORT