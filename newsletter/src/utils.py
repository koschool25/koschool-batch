import urllib.parse

from typing import Optional

from newspaper import Article
from langchain_community.document_loaders import WebBaseLoader


def extract_content(link: str) -> str:
    """
    WebBaseLoader : 본문이 아닌 내용까지 같이 추출 
    -> article에서 에러가 발생하면 WebBaseLoader 이용
    """
    try:
        article = Article(link)
        article.download()
        article.parse()
        if len(article.text.strip()) > 0:
            return article.text
    except Exception:
        pass
        
    loader = WebBaseLoader(link)
    docs = loader.load()
    return docs[0].page_content


def is_similar_content(
    content_embedding: list[float], 
    previous_content_embeddings: list[list[float]], 
    threshold: float = 0.95
) -> bool:
    """이전에 검색된 뉴스 본문과 코사인 유사도가 0.95 이상이면 유사하다 판단 -> 저장 X"""
    if not previous_content_embeddings:
        return False
        
    similarities = cosine_similarity([content_embedding], previous_content_embeddings)
    return similarities.max() >= threshold


def extract_and_decode_bing_url(bing_url: str) -> Optional[str]:
    # Bing news RSS에서 원본 링크 추출
    parsed_url = urllib.parse.urlparse(bing_url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    
    if 'url' in query_params:
        return urllib.parse.unquote(query_params['url'][0])
    return None