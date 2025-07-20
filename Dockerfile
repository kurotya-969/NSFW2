FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 静的ファイルディレクトリが確実に存在するようにする
RUN mkdir -p /app/assets
COPY assets/ /app/assets/

# 明示的にポート10000を設定
ENV PORT=10000
EXPOSE 10000

# 固定ポートを使用して起動
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]