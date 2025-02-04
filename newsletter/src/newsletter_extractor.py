import os
import time
import requests
import openai
import pymysql

from tqdm import tqdm
from typing import Optional
from enum import Enum
from datetime import datetime
from bs4 import BeautifulSoup
from openai import OpenAI

from models import (
    ExtractedNewsletter, 
    NewsletterCategoryEnum, 
    SearchedNewsletter, 
    NewsletterEntity, 
    NewsletterQuizGenerationError, 
    NewsletterExtractionError, 
    GeneratedNewsletterQuiz
)
from utils import extract_and_decode_bing_url, extract_content, is_similar_content
from database import get_db_connection, insert_newsletter_with_quiz
from embedding import TextEmbedder

from dotenv import load_dotenv
load_dotenv()


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


    def generate_newsletter_quiz(self, content: str) -> GeneratedNewsletterQuiz:
        system_prompt = f"""당신은 경제 뉴스 분석 및 퀴즈 생성 전문가입니다. 주어진 경제 뉴스 기사를 바탕으로 아래 형식에 맞게 퀴즈를 출제해주세요.:
---
<뉴스 관련 문제 출제>
JSON 각 필드별 요구사항:
- question: 지정된 레벨에 맞는 명확하고 간단한 주관식 질문
- answer: 정확한 정답 (주관식 답안으로 사용)
- explanation: 해당 레벨 수준에 맞는 상세한 설명과 관련 맥락
- multipleChoices: 4개의 선택지 배열 (첫 번째 요소가 반드시 정답)
- keyword: 퀴즈 핵심 키워드 (이전 키워드와 중복되지 않아야 함)
---
퀴즈 작성 지침:
1. 한국어를 사용
2. 뉴스 기사와 관련된 퀴즈를 출제하고, 되도록 경제 및 투자와 관련된 문제를 출제
3. 지엽적인 문제가 나오지 않도록 뉴스 전반적인 내용을 바탕으로 문제를 출제
4. 주관식 답을 맞출 수 있도록 되도록 정답은 단어 및 용어 위주로 출제
5. 문제에 정답이 포함되지 않도록 출제
6. 설명은 뉴스 기사를 근거로 학습자가 이해할 수 있는 수준으로 작성
7. JSON 형식이 올바르게 유지되어야 함
---
<뉴스 기사>
{content}
---
주어진 경제 뉴스 기사를 분석하여 뉴스 관련 퀴즈를 한국어로 생성해주세요.
"""

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0,
            response_format=GeneratedNewsletterQuiz,
        )

        result = completion.choices[0].message
        if result.refusal:
            raise NewsletterQuizGenerationError(f"뉴스레터 퀴즈 생성에 실패하였습니다.: {result.refusal}")
        else:
            generated_newsletter_quiz = result.parsed

            return generated_newsletter_quiz


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

            # 뉴스 요약문, 관련 종목, 퀴즈 추출
            extractor = NewsletterExtractor(openai_api_key)
            try:
                extracted_newsletter = extractor.extract(content)
                generated_newsletter_quiz = extractor.generate_newsletter_quiz(content)
                newsletter_entity = NewsletterEntity.make_newsletter(
                    category,
                    searched_newsletter,
                    extracted_newsletter,
                )
                newsletter_quiz_entity = generated_newsletter_quiz.to_newsletter_quiz_entity()
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