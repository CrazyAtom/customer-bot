import logging
import os
import ssl
import time
from pathlib import Path
from typing import List, Optional, Union

import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from config.settings import CHROMA_SETTINGS

# ChromaDB 설정에 SSL 컨텍스트 추가
CHROMA_SETTINGS = {
    "anonymized_telemetry": False,
    "allow_reset": True,
    "is_persistent": True,
    "persist_directory": "./chroma_db",
    "ssl": {
        "verify_ssl": False,
        "ca_certs": certifi.where()
    }
}

class WebScraper:
    def __init__(self):
        self.setup_chrome_options()

    def setup_chrome_options(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")

    def get_content(self, url: str) -> Optional[str]:
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(url)

            # 동적 콘텐츠 로딩 대기
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)  # 추가 대기

            page_source = driver.page_source
            driver.quit()

            # BeautifulSoup으로 파싱
            soup = BeautifulSoup(page_source, 'html.parser')

            # 다양한 선택자로 콘텐츠 찾기
            selectors = [
                'main',
                'article',
                'div.content',
                '#__next',
                'div[role="main"]',
                '.article-content',
                '.post-content',
                '.entry-content',
                '#content'
            ]

            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    return "\n".join([elem.get_text(separator='\n', strip=True)
                                    for elem in elements])

            # 선택자로 찾지 못한 경우 본문 추정
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if paragraphs:
                return "\n".join([p.get_text(strip=True) for p in paragraphs])

            return None

        except Exception as e:
            logging.error(f"스크래핑 중 오류 발생: {str(e)}")
            return None

class DocumentLoader:
    @staticmethod
    def load_from_file(file_path: str) -> Optional[str]:
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logging.error(f"파일을 찾을 수 없습니다: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                logging.error("파일이 비어있습니다")
                return None

            return content

        except Exception as e:
            logging.error(f"파일 로딩 중 오류 발생: {str(e)}")
            return None

class ContentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", "!", "?", "？", "！", " ", ""]
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def process_content(self, content: str, source: str) -> Optional[any]:
        try:
            # 문서 생성과 분할
            docs = [Document(page_content=content, metadata={"source": source})]
            all_splits = self.text_splitter.split_documents(docs)

            if not all_splits:
                logging.error("텍스트 청크를 생성할 수 없습니다")
                return None

            # ChromaDB 디렉토리 생성
            os.makedirs(CHROMA_SETTINGS['persist_directory'], exist_ok=True)

            # 벡터스토어 생성
            vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=self.embeddings,
                persist_directory=CHROMA_SETTINGS['persist_directory'],
                client_settings=CHROMA_SETTINGS
            )

            return vectorstore.as_retriever()

        except Exception as e:
            logging.error(f"콘텐츠 처리 중 오류 발생: {str(e)}")
            logging.exception("상세 오류:")
            return None

def load_and_process_data(source: str, source_type: str = "url") -> Optional[any]:
    """
    URL 또는 파일에서 데이터를 로드하고 처리합니다.

    Args:
        source (str): URL 또는 파일 경로
        source_type (str): "url" 또는 "file"

    Returns:
        Optional[any]: 처리된 데이터의 retriever 또는 None
    """
    try:
        content = None
        if source_type == "url":
            scraper = WebScraper()
            content = scraper.get_content(source)
        elif source_type == "file":
            loader = DocumentLoader()
            content = loader.load_from_file(source)
        else:
            logging.error(f"지원하지 않는 소스 타입: {source_type}")
            return None

        if not content:
            logging.error("콘텐츠를 로드할 수 없습니다")
            return None

        processor = ContentProcessor()
        return processor.process_content(content, source)

    except Exception as e:
        logging.error(f"데이터 로딩 및 처리 중 오류 발생: {str(e)}")
        logging.exception("상세 오류:")
        return None