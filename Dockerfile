FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# NumPy の互換性対応（PyTorch未対応）
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

# exec形式で CMD を記述（プロセス管理のため）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
