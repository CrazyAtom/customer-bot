import os

from chromadb.config import Settings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
USER_AGENT = os.getenv('USER_AGENT')

# 모델 설정
GEMINI_MODEL = "gemini-pro"
EMBEDDING_MODEL = "models/embedding-001"

# 데이터 설정
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0

# API 타임아웃 설정
API_TIMEOUT = 120  # 120초

# Google API 설정
GOOGLE_API_SETTINGS = {
    "retry_count": 3,
    "timeout": API_TIMEOUT,
    "backoff_factor": 0.3
}

# ChromaDB 설정
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)