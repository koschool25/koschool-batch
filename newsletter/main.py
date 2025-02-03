import os
import time
import random
import requests
import urllib.parse
import threading

import openai
import pymysql

from tqdm import tqdm
from typing import Optional
from enum import Enum
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel, Field
from newspaper import Article
from langchain_community.document_loaders import WebBaseLoader
from apscheduler.schedulers.background import BackgroundScheduler

from dotenv import load_dotenv
load_dotenv()


def get_db_connection():
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        database=os.environ["MYSQL_DATABASE"],   
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
    )


class NewsletterExtractionError(Exception):
    pass


class NewsletterCategoryEnum(Enum):
    ENERGY = "에너지"
    MATERIALS = "소재"
    INDUSTRIALS = "산업"
    CONSUMER_DISCRETIONARY = "임의소비재"
    CONSUMER_STAPLES = "필수소비재"
    HEALTHCARE = "헬스케어"
    FINANCIALS = "금융"
    INFORMATION_TECHNOLOGY = "정보기술"
    COMMUNICATION = "통신"
    UTILITIES = "유틸리티"
    REAL_ESTATE = "부동산"


class SearchedNewsletter(BaseModel):
    title: str
    link: str
    publish_date: datetime


class NewsletterQuiz(BaseModel):
    question: str
    answer: str
    explanation: str
    wrong_answer1: str
    wrong_answer2: str
    wrong_answer3: str


class NewsletterQuizEntity(BaseModel):
    level: int
    question: str
    answer: str
    explanation: str
    wrong_answer1: str
    wrong_answer2: str
    wrong_answer3: str


class ExtractedNewsletter(BaseModel):
    summary: str
    relative_stock: Optional[str] = None
    quiz: NewsletterQuiz

    def to_newsletter_quiz_entity(self) -> NewsletterQuizEntity:
        return NewsletterQuizEntity(
            level=3,
            question=self.quiz.question,
            answer=self.quiz.answer,
            explanation=self.quiz.explanation,
            wrong_answer1=self.quiz.wrong_answer1,
            wrong_answer2=self.quiz.wrong_answer2,
            wrong_answer3=self.quiz.wrong_answer3,
        )


class NewsletterEntity(BaseModel):
    title: str
    category: str
    summary: str
    link: str
    stock: str
    likes: int = Field(default_factory=lambda: random.randint(10, 200))
    date: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Seoul")).date())
    
    @classmethod
    def make_newsletter(
        cls, 
        category: NewsletterCategoryEnum,
        searched_newsletter: SearchedNewsletter, 
        extracted_newsletter: ExtractedNewsletter
    ):
        return cls(
            title=searched_newsletter.title,
            category=category,
            summary=extracted_newsletter.summary,
            link=searched_newsletter.link,
            stock=extracted_newsletter.relative_stock,
        )


def extract_content(link: str) -> str:
    """
    WebBaseLoader : 본문이 아닌 내용까지 첨부됨 
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
        
    # Article 추출 실패시 WebBaseLoader 사용
    loader = WebBaseLoader(link)
    docs = loader.load()
    return docs[0].page_content


class TextEmbedder:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding


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
    
    # 'url' 파라미터가 존재하면 디코딩하여 반환
    if 'url' in query_params:
        return urllib.parse.unquote(query_params['url'][0])
    return None


class NewsletterExtractor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)


    def extract(self, content: str) -> ExtractedNewsletter:
        system_prompt = f"""당신은 경제 뉴스 분석 전문가입니다. 주어진 경제 뉴스 기사를 분석하여 아래 형식에 맞게 정보를 추출해주세요.:
---
JSON 각 필드별 요구사항:
- summary: 뉴스 요약내용
- relative_stock: 뉴스와 관련된 기업 이름
---
<핵심 내용 요약>
- 글머리 기호(•)를 사용하여 기사의 주요 포인트를 최대 6개의 불렛 포인트로 요약해주세요.
- 가장 중요한 내용부터 순서대로 작성해주세요.
- 수치가 있으면 언급해주세요.
- ~음 으로 문장을 마무리해주세요.
---
<관련 기업 추출>
- 기사 내용과 관련된 기업 1개를 선정하여 기업 이름을 작성해주세요.
- 관련된 기업이 없다면, "없음"을 작성하세요.
---
<뉴스 관련 문제 출제>
JSON 각 필드별 요구사항:
- quiz.question: 지정된 레벨에 맞는 명확하고 간단한 주관식 질문
- quiz.answer: 정확한 정답 (주관식 답안으로 사용)
- quiz.explanation: 해당 레벨 수준에 맞는 상세한 설명과 관련 맥락
- quiz.multipleChoices: 4개의 선택지 배열 (첫 번째 요소가 반드시 정답)
- quiz.keyword: 퀴즈 핵심 키워드 (이전 키워드와 중복되지 않아야 함)
---
퀴즈 작성 지침:
1. 한국어를 사용
2. 뉴스 기사와 관련된 퀴즈를 출제하고, 되도록 경제 및 투자와 관련된 문제를 출제
3. 주관식 답을 맞출 수 있도록 되도록 정답은 단어 및 용어 위주로 출제
4. 문제에 정답이 포함되지 않도록 출제
5. 설명은 뉴스 기사를 근거로 학습자가 이해할 수 있는 수준으로 작성
6. JSON 형식이 올바르게 유지되어야 함
---
<뉴스 기사>
{content}
---
주어진 경제 뉴스 기사를 분석하여 뉴스 기사 요약문, 관련 종목, 뉴스 관련 퀴즈를 한국어로 생성해주세요.
"""
        
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0,
            response_format=ExtractedNewsletter,
        )

        result = completion.choices[0].message
        if result.refusal:
            raise NewsletterExtractionError(f"뉴스레터 정보 추출에 실패하였습니다.: {result.refusal}")
        else:
            extracted_newsletter = result.parsed

            return extracted_newsletter


def search_category_newsletters(category: NewsletterCategoryEnum) -> list[SearchedNewsletter]:
    """Retrieve the company's latest news and return them in Dataframe format"""
    search_url = f"https://www.bing.com/news/search?q={category}%20경제%20뉴스&format=rss"
    print(f"search_url: {search_url}")
    
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")
    category_newsletters = []
    top_n = 20
    for i, item in enumerate(items[:top_n]):
        pubDate = item.pubDate.text
        try:
            newsletter = SearchedNewsletter(
                title=item.title.text,
                link=extract_and_decode_bing_url(item.link.text),
                publish_date=datetime.strptime(
                    pubDate, "%a, %d %b %Y %H:%M:%S %Z"
                ),
            )
        except Exception as e:
            print(f"뉴스 검색 저장 에러 발생: {e}")
            continue
        category_newsletters.append(newsletter)

    category_newsletters.sort(key=lambda x: x.publish_date, reverse=True)

    return category_newsletters


def extract_category_newsletters(category: NewsletterCategoryEnum):
    try:
        connection = get_db_connection()
        # 카테고리 뉴스 검색
        searched_newsletters = search_category_newsletters(category)
        print(f"{category} 뉴스 검색 개수 : {len(searched_newsletters)}")

        n_newsletter = int(os.getenv("N_NEWSLETTER", 3))
        n_saved_newsletters = 0
        for searched_newsletter in searched_newsletters:
            link = searched_newsletter.link

            openai_api_key = os.environ["OPENAI_API_KEY"]
            previous_content_embeddings = []

            # 뉴스 본문 추출
            content = extract_content(link)

            # 뉴스 본문 임베딩 생성
            embedder = TextEmbedder(openai_api_key)
            content_embedding = embedder.get_embedding(content)
            if is_similar_content(content_embedding, previous_content_embeddings):
                print("유사한 뉴스레터가 이미 존재합니다")
                continue
            else:    
                # 뉴스 요약문, 관련 종목
                extractor = NewsletterExtractor(openai_api_key)
                try:
                    extracted_newsletter = extractor.extract(content)
                    newsletter_entity = NewsletterEntity.make_newsletter(
                        category,
                        searched_newsletter,
                        extracted_newsletter,
                    )
                    newsletter_quiz_entity = extracted_newsletter.to_newsletter_quiz_entity()
                    insert_newsletter_with_quiz(connection, newsletter_entity, newsletter_quiz_entity)
                    n_saved_newsletters += 1
                except openai.RateLimitError as e:
                    print("OpenAI API Rate limit 도달... 60초 후에 작업 이어서 진행")
                    time.sleep(60)
                    continue

                if n_saved_newsletters >= n_newsletter:
                    break

                previous_content_embeddings.append(content_embedding)
    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
    finally:
        connection.close()


def insert_newsletter_with_quiz(connection, newsletter_entity: NewsletterEntity, newsletter_quiz_entity: NewsletterQuizEntity):
    try:
        with connection.cursor() as cursor:
            newsletter_insert_query = """
            INSERT INTO newsletter (title, category, summary, link, stock, likes, date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            newsletter_quiz_insert_query = """
            INSERT INTO quiz (level, newsletter_id, question, answer, explanation, wrong_answer1, wrong_answer2, wrong_answer3)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            newsletter_values = (
                newsletter_entity.title,
                newsletter_entity.category,
                newsletter_entity.summary,
                newsletter_entity.link,
                newsletter_entity.stock,
                newsletter_entity.likes,
                newsletter_entity.date
            )
            cursor.execute(newsletter_insert_query, newsletter_values)
            newsletter_id = cursor.lastrowid

            newsletter_quiz_values = (
                newsletter_quiz_entity.level,
                newsletter_id,
                newsletter_quiz_entity.question,
                newsletter_quiz_entity.answer,
                newsletter_quiz_entity.explanation,
                newsletter_quiz_entity.wrong_answer1,
                newsletter_quiz_entity.wrong_answer2,
                newsletter_quiz_entity.wrong_answer3
            )
            cursor.execute(newsletter_quiz_insert_query, newsletter_quiz_values)
            
            connection.commit()
            print("뉴스레터 데이터가 성공적으로 저장되었습니다.")

    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
        raise e


def clear_newsletter_table():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            delete_query = "DELETE FROM newsletter"
            cursor.execute(delete_query)
            connection.commit()
            print("newsletter 테이블의 모든 데이터가 삭제되었습니다.")
    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
    finally:
        connection.close()


def get_all_newsletters():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            select_query = """
            SELECT title, category, summary, link, stock, likes, date 
            FROM newsletter
            ORDER BY date DESC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            
            print("\n=== 뉴스레터 목록 ===")
            for row in results:
                print(f"\n제목: {row[0]}")
                print(f"카테고리: {row[1]}")
                print(f"요약: {row[2]}")
                print(f"링크: {row[3]}")
                print(f"주식: {row[4]}")
                print(f"좋아요 수: {row[5]}")
                print(f"날짜: {row[6]}")
                print("-" * 50)
            
            print(f"\n총 {len(results)}개의 뉴스레터가 있습니다.")
            
    except pymysql.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
    finally:
        connection.close()


def main():
    threads = []
    for category in tqdm(NewsletterCategoryEnum):
        print(f"{category.value} 뉴스 저장 시작...")
        thread = threading.Thread(target=extract_category_newsletters, args=(category.value,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=main,
        trigger='cron',
        hour=23,
        minute=0,
        second=0,
        args=[]
    )

    try:
        scheduler.start()
        print("스케줄러 실행...")
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("스케줄러 종료...")
