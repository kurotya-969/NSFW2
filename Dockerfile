FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render が PORT を注入するため明示的に環境変数指定しなくてOKだが、念のためデフォルト値も定義
ENV PORT=10000
EXPOSE 10000

# PORT を shell 展開するには sh -c の中で $PORT を囲う
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT}"

